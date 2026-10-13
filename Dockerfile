# syntax=docker/dockerfile:1
#
# decider, deployed through llama.cpp (GGUF).
#
#   upstream ...... https://github.com/Mapika/decider
#   documented path https://github.com/Mapika/decider#quick-start   ->  "GGUF (llama.cpp)"
#                   pip install "decider-ai[gguf]"  +  CMAKE_ARGS="-DGGML_CUDA=on"
#                   (this image adds the "serve" extra on top, for the HTTP API)
#
# IN the image
#   * the upstream source at DECIDER_REF, installed as decider-ai[gguf,serve]
#   * llama-cpp-python (>=0.3.35,<0.4) built from source with -DGGML_CUDA=on against CUDA 12.4.1, sm_86 (A40)
#   * Python 3.12 on Ubuntu 22.04
#   * scripts/{serve.sh,serve_gguf.py}: the HTTP API on the GGUF engine (fastapi/uvicorn come from the serve extra)
#
# NOT in the image
#   * no .gguf weights, no tokenizer, no decider_config.json -- mount the model folder (see README.md)
#
# Build
#   docker build -t decider-gguf:cu12.4.1 .
#
# Run -- HTTP API, on llama.cpp.  Upstream's scripts/serve.sh starts decider.serve, which is the *torch* engine and
# cannot read a GGUF ("The HTTP server does not serve GGUF files"), so this repo ships its own GGUF server.
# A self-contained model folder is read in place and needs no cache volume -- nothing goes to the Hub.  The
# /hf-cache volume is only for the other form, a Hub repo id (see README section 5).
#   docker run --rm --gpus all -p 8000:8000 -v /srv/decider-4b-GGUF:/models \
#     decider-gguf:cu12.4.1 scripts/serve.sh /models/decider-4b-v2.1-Q4_K_M.gguf 8000
#
# Run -- upstream's demo, or your own code (the model folder holds the .gguf, the tokenizer and decider_config.json)
#   docker run --rm --gpus all -v /srv/decider-4b-GGUF:/models decider-gguf:cu12.4.1 \
#     python -m decider.infer /models/decider-4b-v2.1-Q4_K_M.gguf
#
# The default command is `tail -f /dev/null`. A command after the image name replaces it entirely, so the
# caller decides what actually runs.

# CUDA 12.4 has no Ubuntu 24.04 base image (24.04 starts at 12.5.1), so Ubuntu 22.04 it is.
ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.12
ARG DECIDER_REPO=https://github.com/Mapika/decider
ARG DECIDER_REF=main
# A40 = Ampere = sm_86. One arch instead of nvcc's whole default list. `native`, `all` or `80;86;89;90` also work.
ARG LLAMA_CUDA_ARCH=86
# llama.cpp does the CUDA work on this path -- both for `decider.infer` and for scripts/serve_gguf.py -- and torch
# here only does small host-side tensor ops on the readout.  The CPU wheel is therefore enough, and it keeps a
# second CUDA user space (torch's bundled nvidia-*-cu12 libs) out of the image.
# Nothing in this image needs CUDA torch, but a caller who wants to run the torch engine (decider.serve) next to
# it can build with https://download.pytorch.org/whl/cu124 -- the newest CUDA torch the target driver allows.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC=torch

# ---------------------------------------------------------------------------------------------------------------------
# Stage 1: builder. nvcc and a host compiler come from the -devel image and from build-essential.
# ---------------------------------------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ARG DECIDER_REPO
ARG DECIDER_REF
ARG LLAMA_CUDA_ARCH
ARG TORCH_INDEX
ARG TORCH_SPEC

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04 ships Python 3.10, decider-ai requires >=3.11 and upstream CI uses 3.12, so deadsnakes it is.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      gnupg \
      ninja-build \
      software-properties-common; \
    add-apt-repository -y ppa:deadsnakes/ppa; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      "python${PYTHON_VERSION}" \
      "python${PYTHON_VERSION}-dev" \
      "python${PYTHON_VERSION}-venv"; \
    rm -rf /var/lib/apt/lists/*; \
    "python${PYTHON_VERSION}" -m venv /opt/venv; \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

# torch first: decider-ai requires the bare, unpinned `torch`, so a wheel that is already installed is reused and
# the rest of the dependency graph cannot pull a different build in behind it.
RUN set -eux; \
    /opt/venv/bin/pip install --index-url "${TORCH_INDEX}" "${TORCH_SPEC}"

# Upstream source, shallow. `git init` + `fetch --depth 1` accepts a branch, a tag and a commit SHA alike
# (`git clone --branch` does not take a SHA).
RUN set -eux; \
    git init -q /opt/decider; \
    git -C /opt/decider remote add origin "${DECIDER_REPO}"; \
    git -C /opt/decider fetch -q --depth 1 origin "${DECIDER_REF}"; \
    git -C /opt/decider checkout -q --detach FETCH_HEAD; \
    rm -rf /opt/decider/.git

# decider-ai[gguf,serve]: the package + llama-cpp-python>=0.3.35,<0.4 + fastapi/uvicorn[standard]/httpx.
# CMAKE_ARGS is how llama-cpp-python's build forwards options to llama.cpp's CMake, the same way the upstream
# README documents -DGGML_CUDA=on.
#   GGML_NATIVE=OFF            do not tune the binary for the build host's CPU (the runner is not the A40 host)
#   CMAKE_CUDA_ARCHITECTURES   build for the target arch only
# The serve extra is what scripts/serve_gguf.py runs on; note that it also makes upstream's decider.serve
# importable, but that one serves the torch engine and still cannot read a GGUF.
RUN set -eux; \
    export CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCH}"; \
    /opt/venv/bin/pip install "/opt/decider[gguf,serve]"

# The upstream has no lockfile, so the resolver's choices are only pinned down here.  Printed into the build log
# and, in the workflow, into the artifact manifest.
RUN /opt/venv/bin/python -c "import importlib.metadata as m;print(' '.join(p+'='+m.version(p) for p in ('decider-ai','torch','transformers','flash-linear-attention','llama-cpp-python','fastapi','uvicorn')))"

# Self-check for what the install step alone cannot promise: that the CUDA backend really landed in the built
# llama.cpp. A missing backend is a silent CPU fallback -- the install succeeds either way -- so it is asserted
# here, again on the image that ships (the runtime stage) and again in the workflow smoke test.
#
# `import llama_cpp` cannot be run bare in a build. Built with GGML_CUDA, llama.cpp links the CUDA *driver*
# (libcuda.so.1, for the VMM pool) and no CUDA image carries that library: the host driver injects it at run time,
# through the NVIDIA container runtime. What the -devel image does carry is the driver *stub* (libcuda.so, 70 KB,
# under /usr/local/cuda*/.../lib/stubs -- the directory its LIBRARY_PATH already points at). Aliasing that to
# libcuda.so.1 satisfies the loader without a GPU: importing resolves the driver's entry points and calls none of
# them. The alias is made under /tmp in the builder, and the builder is copied nowhere.
RUN set -eux; \
    lib="$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/llama_cpp/lib"; \
    test -d "$lib" || { echo "not a directory: $lib (venv copy or PYTHON_VERSION changed?)" >&2; exit 1; }; \
    ls -l "$lib"; \
    ls "$lib"/libggml-cuda.so*; \
    readelf -d "$lib"/*.so* 2>/dev/null | grep -F '[libcuda.so.1]'; \
    stub="$(ls /usr/local/cuda*/lib64/stubs/libcuda.so /usr/local/cuda*/targets/*/lib/stubs/libcuda.so 2>/dev/null | head -1)"; \
    test -n "$stub"; \
    mkdir -p /tmp/cuda-stub; \
    ln -sf "$stub" /tmp/cuda-stub/libcuda.so.1; \
    LD_LIBRARY_PATH=/tmp/cuda-stub /opt/venv/bin/python -c "import os, llama_cpp, torch, fastapi, uvicorn; from decider.infer import Decider; from decider.engine_gguf import GGUFEngine; from decider.serve import app; print('OK llama_cpp='+llama_cpp.__version__+' torch='+torch.__version__+' fastapi='+fastapi.__version__+' uvicorn='+uvicorn.__version__+' at '+os.path.dirname(llama_cpp.__file__))"; \
    rm -rf /tmp/cuda-stub

# ---------------------------------------------------------------------------------------------------------------------
# Stage 2: runtime. `-runtime` keeps cudart/cuBLAS (llama.cpp links them) and drops nvcc, the headers and the
# static libraries. Both stages share CUDA / Ubuntu / Python, so the venv's absolute paths and ABI stay valid.
# ---------------------------------------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION
ARG DECIDER_REF

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    HF_HOME=/hf-cache

LABEL org.opencontainers.image.title="decider-gguf" \
      org.opencontainers.image.description="decider (Mapika/decider) GGUF decisions through llama.cpp on CUDA 12.4.1; weights are not included" \
      org.opencontainers.image.source="https://github.com/Mapika/decider" \
      org.opencontainers.image.revision="${DECIDER_REF}" \
      org.opencontainers.image.licenses="Apache-2.0"

# libgomp1: llama.cpp's OpenMP runtime. The -runtime image carries no gcc, so nothing pulls it in by itself.
# /hf-cache is the Hugging Face cache (HF_HOME). A local model folder never writes it -- the Hub is only touched
# when the model is passed as a repo id, which is the one case where mounting a volume here pays for itself.
# Not a VOLUME on purpose: nothing should be downloaded into a layer that a caller forgets to persist.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      gnupg \
      libgomp1 \
      software-properties-common; \
    add-apt-repository -y ppa:deadsnakes/ppa; \
    apt-get update; \
    apt-get install -y --no-install-recommends "python${PYTHON_VERSION}"; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p "${HF_HOME}" /models

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/decider /opt/decider
COPY scripts /app/scripts

WORKDIR /app

# Three checks on the image that ships. None of them needs a GPU:
#   * the CUDA backend is in the venv that was copied over -- llama.cpp falls back to the CPU in silence.
#   * every shared library the llama.cpp libs need resolves *here*, and the CUDA driver is the only one that does
#     not. The driver is meant to be missing (the host injects it at run time), so it is allowed by name -- and by
#     exact name: `libcuda*` would also match libcudart.so.12, which must be there and is exactly the kind of
#     thing this check is for. Anything else unresolved breaks the build. This is also why this stage cannot
#     `import llama_cpp` either -- same driver, same reason as the builder stage above.
#   * the GGUF HTTP server this repo adds imports with all five routes -- catches a serve extra that did not land
#     or a route upstream renamed. The import is cheap: the model is only loaded in the FastAPI lifespan.
RUN set -eux; \
    lib="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/llama_cpp/lib"; \
    test -d "$lib" || { echo "not a directory: $lib (venv copy or PYTHON_VERSION changed?)" >&2; exit 1; }; \
    ls -l "$lib"; \
    ls "$lib"/libggml-cuda.so*; \
    python -c "import glob, subprocess as sp, sysconfig; lib=sysconfig.get_paths()['purelib']+'/llama_cpp/lib'; libs=sorted(glob.glob(lib+'/*.so*')); assert libs, 'no shared libraries under '+lib; out=sp.run(['ldd']+libs, capture_output=True, text=True).stdout; missing=sorted({l.split()[0] for l in out.splitlines() if 'not found' in l}); print('unresolved shared libraries:', missing); driver=('libcuda.so.1','libnvcuvid.so.1'); bad=[m for m in missing if not (m in driver or m.startswith('libnvidia'))]; assert not bad, 'this image cannot resolve: '+repr(bad)"; \
    chmod +x /app/scripts/serve.sh; \
    bash -n /app/scripts/serve.sh; \
    python -c "import sys; sys.path.insert(0, '/app/scripts'); import serve_gguf; paths=sorted(r.path for r in serve_gguf.app.routes); missing={'/decide','/v1/systemone','/health','/v1/models','/stats'}-set(paths); assert not missing, 'missing routes: '+repr(missing); print('serve routes', paths)"

# Informational only (docker run -P would publish it); scripts/serve.sh defaults to 0.0.0.0:8000.
EXPOSE 8000

# The CUDA images ship ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"], which would run a caller's command as
# an argument to that script. Clearing it is what makes `docker run <image> <command>` replace the whole
# command line. It does not touch GPU access: device injection happens in the container runtime, before exec.
ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]
