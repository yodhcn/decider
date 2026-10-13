# syntax=docker/dockerfile:1
#
# decider, deployed through llama.cpp (GGUF).
#
#   upstream ...... https://github.com/Mapika/decider
#   documented path https://github.com/Mapika/decider#quick-start   ->  "GGUF (llama.cpp)"
#                   pip install "decider-ai[gguf]"  +  CMAKE_ARGS="-DGGML_CUDA=on"
#
# IN the image
#   * the upstream source at DECIDER_REF, installed as decider-ai[gguf]
#   * llama-cpp-python (>=0.3.35,<0.4) built from source with -DGGML_CUDA=on against CUDA 12.4.1, sm_86 (A40)
#   * Python 3.12 on Ubuntu 22.04
#
# NOT in the image
#   * no .gguf weights, no tokenizer, no decider_config.json -- mount the model folder (see README.md)
#
# Build
#   docker build -t decider-gguf:cu12.4.1 .
#
# Run (the model folder holds the .gguf, the tokenizer and decider_config.json)
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
# llama.cpp does the CUDA work on this path; torch here only does small CPU tensor ops on the readout, so the CPU
# wheel is enough and keeps a second CUDA user space (torch's bundled nvidia-*-cu12 libs) out of the image.
# For the CUDA torch build (needed by the torch engine / decider.serve):
#   --build-arg TORCH_INDEX=https://pypi.org/simple
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

# decider-ai[gguf] = the package + llama-cpp-python>=0.3.35,<0.4. CMAKE_ARGS is how llama-cpp-python's build
# forwards options to llama.cpp's CMake, the same way the upstream README documents -DGGML_CUDA=on.
#   GGML_NATIVE=OFF            do not tune the binary for the build host's CPU (the runner is not the A40 host)
#   CMAKE_CUDA_ARCHITECTURES   build for the target arch only
RUN set -eux; \
    export CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCH}"; \
    /opt/venv/bin/pip install "/opt/decider[gguf]"

# Self-check for what the install step alone cannot promise: that the CUDA backend really landed in the built
# llama.cpp (a missing backend is a silent CPU fallback, not an install error) and that the package imports.
RUN set -eux; \
    /opt/venv/bin/python -c "import glob,os,llama_cpp,torch;from decider.infer import Decider;from decider.engine_gguf import GGUFEngine;lib=os.path.join(os.path.dirname(llama_cpp.__file__),'lib');cuda=sorted(glob.glob(os.path.join(lib,'*cuda*')));assert cuda,'llama.cpp was built without a CUDA backend: '+lib+' holds '+repr(sorted(os.listdir(lib)));print('OK torch='+torch.__version__+' cuda_backend='+repr([os.path.basename(p) for p in cuda]))"

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
# /hf-cache is the Hugging Face cache (HF_HOME), so a caller can mount a volume and reuse it.
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

WORKDIR /app

# The CUDA images ship ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"], which would run a caller's command as
# an argument to that script. Clearing it is what makes `docker run <image> <command>` replace the whole
# command line. It does not touch GPU access: device injection happens in the container runtime, before exec.
ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]
