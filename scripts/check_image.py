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
import re
import subprocess
import sys
import sysconfig

# Derived, not hardcoded, so the same file works wherever scripts/ is copied to (in the image it is /app/scripts
# and the working directory is /app).
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)

# The CUDA *major* this image must stay on.  Minor is free: CUDA 12.x is minor-version compatible, so a cu126
# wheel runs on the 550.90.12 driver that caps the toolkit at 12.4.  CUDA 13 is not free -- it needs driver
# >= 580 -- and a CPU torch wheel would run the whole engine on the CPU without decider.serve saying a word.
EXPECTED_CUDA_MAJOR = 12

# flash-linear-attention's own floor, used only if it cannot be read out of fla (see _fla_triton_floor).
# Below it, fla does not fail: it caches a 'cpu' platform and transformers switches to a reference kernel.
MIN_TRITON = "3.3.0"

# What a client of upstream's server needs.  A renamed or dropped route breaks it, quietly, at the client.
ROUTES = {"/decide", "/v1/systemone", "/health", "/v1/models", "/stats"}

# Only the host driver may be unresolved, by exact name, plus the driver's own libnvidia* prefix.  A `libcuda*`
# prefix would also let libcudart.so.12 through, and that one is exactly what this check is for.
DRIVER_LIBS = ("libcuda.so.1", "libnvcuvid.so.1")


def _vtuple(s):
    """`"3.3.1+gitabc"` -> `(3, 3, 1)`.  Comparing these beats importing packaging to compare three numbers."""
    nums = re.findall(r"\d+", s.split("+")[0])
    return tuple(int(n) for n in (nums + ["0", "0", "0"])[:3])


def _subprocess_python(code):
    """Run a snippet in a fresh interpreter and return its stdout.

    Used for anything that imports torch-adjacent libraries at module scope: fla's device detection runs on
    import and prints a warning that only means "this machine has no GPU", which in a build log reads like a
    defect.  Keeping it out of process keeps the report honest.
    """
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (r.stderr.strip().splitlines() or ["no output"])[-1]
    return r.stdout.strip()


def check_torch_is_the_cuda_build():
    import torch
    cuda = torch.version.cuda or ""
    assert cuda, (f"torch {torch.__version__} is a CPU build: the engine would run on the CPU and "
                  f"decider.serve would not mention it")
    major, minor = (int(x) for x in cuda.split(".")[:2])
    assert major == EXPECTED_CUDA_MAJOR, (
        f"torch is built for CUDA {cuda}, but this image has to stay on CUDA {EXPECTED_CUDA_MAJOR}.x: the host "
        f"driver is 550.90.12, and CUDA 13 requires a >= 580 driver")
    local = torch.__version__.split("+")[1] if "+" in torch.__version__ else ""
    assert local == f"cu{major}{minor}", (
        f"torch {torch.__version__} reports CUDA {cuda} but its wheel is tagged {local!r}: the wheel was built "
        f"against a different toolkit than it claims")
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
    forward on the A40 fails -- at request time, where it is expensive to diagnose, not here.

    Presence only.  Being importable says nothing about being *usable*: triton 3.2.0 imported fine and still
    sent every hybrid layer to a reference kernel.  That is the next check's job.
    """
    absent = [m for m in ("triton", "fla") if importlib.util.find_spec(m) is None]
    assert not absent, f"missing module(s) {absent}: the Qwen3.5 hybrid layers cannot run a forward"
    return "triton + flash-linear-attention (module `fla`) present"


def _fla_triton_floor():
    """fla's minimum triton, read out of its source.

    It cannot be imported: `required_triton_version` is a local inside check_environments(), not a module
    constant.  Falls back to MIN_TRITON if fla reorganises, because the check has to have a floor either way.
    """
    try:
        src = _subprocess_python(
            "import inspect;from fla.utils import _device;print(inspect.getsource(_device.check_environments))")
    except AssertionError:
        return MIN_TRITON, f"fla floor unreadable, using the documented {MIN_TRITON}"
    m = re.search(r"required_triton_version\s*=\s*package_version\.parse\([\"']([\d.]+)[\"']\)", src)
    if not m:
        return MIN_TRITON, f"fla floor not found in its source, using the documented {MIN_TRITON}"
    return m.group(1), None


def check_triton_meets_fla_floor():
    """The check whose absence cost a real GPU run.

    `find_spec("fla")` passed, the build went green, and the first A40 forward threw: triton was 3.2.0, below
    fla 0.5.2's declared floor of 3.3.  fla does not raise on that -- and that is the problem.  On 3.2 its
    `get_available_device()` swallowed whatever the platform probe raised, warned "roll back to CPU", @cache'd
    the answer, and transformers quietly took the O(N^2) reference path.  The image looked correct and behaved
    like a different model on different hardware.
    """
    import triton
    floor, note = _fla_triton_floor()
    got = triton.__version__
    assert _vtuple(got) >= _vtuple(floor), (
        f"triton {got} is below {floor}, the floor flash-linear-attention declares: on a GPU this does not "
        f"error, it silently falls back to the reference O(N^2) kernel -- slow, and OOM at 4B.  Take a torch "
        f"whose triton is >= {floor} (cu126 + torch 2.7.1 gives 3.3.1)")
    return f"triton {got} >= {floor}" + (f" ({note})" if note else " (fla's own floor)")


def check_conv_kernel_installed():
    """The same class of silent fallback: without causal_conv1d transformers prints one line and uses its
    reference conv for every Qwen3.5 hybrid layer.  Imported for real, in a subprocess -- a wheel that installs
    but cannot load is exactly what a `find_spec` check would have called healthy."""
    v = _subprocess_python(
        "import causal_conv1d as c;print(getattr(c, '__version__', '') or 'installed')")
    assert v, "causal_conv1d imported but reported nothing"
    return f"causal_conv1d {v}"


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


def _triton_cache_probe():
    """The per-key directory Triton has just created -- asked of Triton, not modelled from its API.

    The obvious query lies, twice over.  triton 3.2 and 3.3 -- which is what a torch 2.6/cu124 build and this
    image's 2.7/cu126 build pin -- have no knobs module at all, and their default_cache_dir() is only the
    *fallback*: it returns $TRITON_HOME/.triton/cache whatever TRITON_CACHE_DIR says, so reporting it as
    "where triton caches" fails a correct image.  triton >= 3.4 does expose knobs.cache.dir, but guessing which
    generation is installed is the mistake to avoid, not to make twice.  What every generation shares is the
    cache manager, and its constructor is the code that actually turns the setting into a directory -- so ask
    that one, and let it create the directory as proof.
    """
    from triton.runtime.cache import get_cache_manager
    manager = get_cache_manager("00")              # any hex key; its encoding names the per-key subdirectory
    per_key = getattr(manager, "cache_dir", None)
    assert per_key, (f"{type(manager).__name__} reports no cache_dir: TRITON_CACHE_MANAGER is set to a manager "
                     f"this check cannot probe")
    return per_key


def check_caches_point_where_we_set_them():
    """Three ways this breaks, none of them visible to a check that reads the variable back and trusts it.  An
    empty TRITON_CACHE_DIR is not "no caching": on 3.2 it silently falls back to ~/.triton/cache, and on the
    knobs releases it makes every compile raise.  A library that stopped honouring the variable sends the
    kernels outside the mounted volume, where the next container rebuilds them.  A read-only volume fails here
    instead of at the first request."""
    want = (os.environ.get("TRITON_CACHE_DIR") or "").strip()
    assert want, ("TRITON_CACHE_DIR is unset or empty, which Triton does not read as 'no caching': 3.2 falls "
                  "back to $TRITON_HOME/.triton/cache and the knobs releases raise on every compile")
    per_key = _triton_cache_probe()
    try:
        assert per_key.startswith(want.rstrip("/") + "/"), (
            f"triton compiles into {os.path.dirname(per_key)!r}, outside TRITON_CACHE_DIR={want!r}: the mounted "
            f"volume would not cover the kernels and every container would rebuild them")
    finally:
        try:
            os.rmdir(per_key)                      # empty or not: only removes a directory we just created
        except OSError:
            pass
    detail = f"triton -> {os.path.dirname(per_key)}"

    try:
        import huggingface_hub.constants as hf
    except ImportError:                            # only reachable if the hub client is not installed at all
        return detail
    hub = getattr(hf, "HF_HUB_CACHE", None)
    if hub is None:
        return detail
    home = (os.environ.get("HF_HOME") or "/hf-cache").rstrip("/")
    assert hub == home or hub.startswith(home + "/"), (
        f"huggingface_hub caches to {hub}, outside HF_HOME={home!r}: the checkpoint would be re-downloaded "
        f"every container even with /hf-cache mounted")
    return f"{detail}; hub -> {hub}"


CHECKS = (
    ("torch is a CUDA 12 build", check_torch_is_the_cuda_build),
    ("cuda libraries resolve", check_cuda_libraries_resolve),
    ("decider.serve routes", check_server_imports_with_every_route),
    ("hybrid-layer deps", check_hybrid_layer_dependencies),
    ("triton meets fla floor", check_triton_meets_fla_floor),
    ("conv kernel", check_conv_kernel_installed),
    ("dependency versions", check_dependency_versions),
    ("entrypoint script", check_entrypoint_script),
    ("cache dirs writable", check_cache_dirs_are_writable),
    ("caches honour the env", check_caches_point_where_we_set_them),
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
