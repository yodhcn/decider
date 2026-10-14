"""The contract this image promises, asserted where no GPU is available.

Run at build time (`RUN python /app/scripts/check_image.py`) and again in the CI smoke test, against the image
that was loaded from the artifact.  One file, so the two cannot drift apart.

Needs no GPU: every check is about what is on disk.  `import torch` does not touch libcuda.so.1, and
`decider.serve` only builds the engine in its FastAPI lifespan, so importing it is cheap and GPU-free.

  python /app/scripts/check_image.py     -> exit 0, or exit 1 naming every check that failed
"""
import glob
import importlib.metadata as md
import importlib.util
import os
import subprocess
import sys
import sysconfig

# Derived, not hardcoded, so the same file works wherever scripts/ is copied to (in the image it is /app/scripts
# and the working directory is /app).
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)

# The CUDA version this image is built against.  Driver 550.90.12 on the target host caps it at 12.4, which is
# why the cu124 index.  A CPU torch wheel would run the whole engine on the CPU and decider.serve would not say
# a word -- that silent fallback is the failure this file exists to catch.
EXPECTED_CUDA = (12, 4)

# What a client of upstream's server needs.  A renamed or dropped route breaks it, quietly, at the client.
ROUTES = {"/decide", "/v1/systemone", "/health", "/v1/models", "/stats"}

# Only the host driver may be unresolved, by exact name, plus the driver's own libnvidia* prefix.  A `libcuda*`
# prefix would also let libcudart.so.12 through, and that one is exactly what this check is for.
DRIVER_LIBS = ("libcuda.so.1", "libnvcuvid.so.1")


def check_torch_is_the_cuda_build():
    import torch
    cuda = torch.version.cuda or ""
    assert tuple(int(x) for x in cuda.split(".")[:2]) == EXPECTED_CUDA, (
        f"torch is not the cu{EXPECTED_CUDA[0]}{EXPECTED_CUDA[1]} build: torch={torch.__version__} "
        f"cuda={cuda!r}.  A CPU torch runs the engine on the CPU without saying so")
    return f"torch {torch.__version__}, built for CUDA {cuda}"


def check_cuda_libraries_resolve():
    """Everything torch's CUDA path needs must resolve *here*; only the driver is allowed to be absent."""
    purelib = sysconfig.get_paths()["purelib"]
    libs = sorted(glob.glob(purelib + "/torch/lib/*.so*") + glob.glob(purelib + "/nvidia/*/lib/*.so*"))
    assert libs, f"no torch shared libraries under {purelib} -- is torch installed at all?"
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(sorted({os.path.dirname(x) for x in libs})))
    out = subprocess.run(["ldd", *libs], capture_output=True, text=True, env=env).stdout
    missing = sorted({line.split()[0] for line in out.splitlines() if "not found" in line})
    bad = [m for m in missing if not (m in DRIVER_LIBS or m.startswith("libnvidia"))]
    assert not bad, (f"this image cannot resolve {bad}; the allowed-absent set is exactly "
                     f"{list(DRIVER_LIBS)} plus libnvidia* (host driver)")
    return f"{len(libs)} libraries, driver-only absences: {missing or 'none'}"


def check_server_imports_with_every_route():
    assert importlib.util.find_spec("decider.serve") is not None, (
        "decider.serve is not importable -- was /opt/decider[serve] installed?")
    from decider.serve import app
    paths = {r.path for r in app.routes}
    missing = ROUTES - paths
    assert not missing, f"decider.serve is missing {sorted(missing)}; it has {sorted(paths)}"
    return f"5 routes as upstream: {sorted(ROUTES)}"


def check_hybrid_layer_dependencies():
    """Qwen3.5's Gated DeltaNet layers go through flash-linear-attention on Triton.  Miss either and the first
    forward on the A40 fails -- at request time, where it is expensive to diagnose, not here."""
    absent = [m for m in ("triton", "fla") if importlib.util.find_spec(m) is None]
    assert not absent, f"missing module(s) {absent}: the Qwen3.5 hybrid layers cannot run a forward"
    return "triton + flash-linear-attention (module `fla`) present"


def check_dependency_versions():
    got = {p: md.version(p) for p in ("decider-ai", "transformers", "flash-linear-attention")}
    major = int(got["transformers"].split(".")[0])
    assert major >= 5, f"transformers {got['transformers']} does not satisfy decider-ai's transformers>=5"
    return "  ".join(f"{k}={v}" for k, v in got.items())


def check_no_weights_are_baked_in():
    """The image ships the runtime, never the checkpoint: mounting the model is the caller's decision, and 4 GB
    of safetensors baked into a layer could not be swapped without a rebuild."""
    hits = [p for root in ("/models", APP)
            for ext in ("safetensors", "gguf")
            for p in glob.glob(f"{root}/**/*.{ext}", recursive=True)]
    assert not hits, f"the image carries weights: {hits[:5]} -- mount the checkpoint instead"
    return f"none under /models or {APP}"


def check_entrypoint_script():
    """The documented way in is `docker run <image> scripts/serve.sh <model> <port>`, resolved against WORKDIR
    /app.  An unexecutable file, a bash typo, or a leftover script from another engine all fail there."""
    path = os.path.join(HERE, "serve.sh")
    assert os.path.exists(path), f"{path} is not in the image"
    assert os.access(path, os.X_OK), f"{path} is not executable -- `docker run ... scripts/serve.sh` would fail"
    text = open(path).read()
    assert "decider.serve:app" in text, f"{path} does not start decider.serve:app"
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"{path}: not valid bash: {r.stderr.strip()}"
    return "scripts/serve.sh present, executable, bash-syntax clean, runs decider.serve:app"


def check_cache_dirs_are_writable():
    """Both are env-configured mount points.  Building the layer without them, or mounting a read-only volume,
    turns into a confusing failure much later: a re-download, or a Triton compile that never caches."""
    dirs = [os.environ.get("HF_HOME", "/hf-cache"), os.environ.get("TRITON_CACHE_DIR", "/triton-cache")]
    for d in dirs:
        assert os.path.isdir(d), f"{d} does not exist (the Dockerfile creates it; mount a volume here)"
        probe = os.path.join(d, ".write-probe")
        try:
            with open(probe, "w") as fh:
                fh.write("")
        finally:
            if os.path.exists(probe):
                os.remove(probe)
    return "writable: " + ", ".join(dirs)


CHECKS = (
    ("torch is the cu124 build", check_torch_is_the_cuda_build),
    ("cuda libraries resolve", check_cuda_libraries_resolve),
    ("decider.serve routes", check_server_imports_with_every_route),
    ("hybrid-layer deps", check_hybrid_layer_dependencies),
    ("dependency versions", check_dependency_versions),
    ("entrypoint script", check_entrypoint_script),
    ("cache dirs writable", check_cache_dirs_are_writable),
    ("no baked-in weights", check_no_weights_are_baked_in),
)


def main():
    width = max(len(name) for name, _ in CHECKS)
    failed = []
    for name, fn in CHECKS:
        try:
            detail = fn()
        except Exception as e:                                    # noqa: BLE001 - report every failure, not the first
            failed.append(name)
            print(f"FAIL {name:<{width}}  {type(e).__name__}: {e}", flush=True)
        else:
            print(f"ok   {name:<{width}}  {detail}", flush=True)
    if failed:
        print(f"\nERROR: {len(failed)} of {len(CHECKS)} checks failed: {failed}", file=sys.stderr, flush=True)
        return 1
    print(f"\nOK: {len(CHECKS)} checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
