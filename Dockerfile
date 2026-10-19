# syntax=docker/dockerfile:1
#
# decider on the torch engine -- the server upstream ships, not a reimplementation of it.
#
#   upstream ....... https://github.com/Mapika/decider
#   weights ........ NOT in the image.  A full checkpoint (safetensors + tokenizer + decider_config.json),
#                    2B bf16 about 4 GB, 4B bf16 8.4 GB, from a Hub repo id or a mounted folder.
#   engine ......... decider.engine_v2.EngineV2: bf16, shape-bucketed CUDA graphs, optional torch.compile
#                    and FP8, shared-prefix scoring, dynamic batching (decider.batching.plan_batches).
#   server ......... decider.serve:app
#                    POST /decide  POST /v1/systemone  GET /health /v1/models /stats
#   in the image ... the upstream source at DECIDER_REF, installed as decider-ai[serve]
#                    Python 3.12, torch 2.7.1 from the cu126 index (triton 3.3.1), causal_conv1d
#                    scripts/{serve.sh,check_image.py}
#
# Why python:3.12-slim and not an nvidia/cuda base.  Everything CUDA this image needs arrives through pip:
# torch is installed from https://download.pytorch.org/whl/cu126, which pulls cudart / cuBLAS / cuDNN / NCCL /
# nvrtc into site-packages, and Triton -- what flash-linear-attention's Qwen3.5 Gated DeltaNet kernels run on
# -- ships its own ptxas, so no nvcc is involved either.  An nvidia/cuda base would stack a *second*,
# separately versioned CUDA user space on top of the one torch already carries, for no gain.  The one CUDA
# component no base image has is libcuda.so.1: the host driver injects it at container start through the
# NVIDIA container runtime.
#
# Why cu126 and not cu124, although the host driver is 550.90.12 -- the CUDA 12.4 *toolkit* driver.  CUDA 12.x
# is minor-version compatible: an application built against any 12.x toolkit runs on any driver >= 525, so the
# driver is not what constrains this image.  Triton is.  flash-linear-attention 0.5.2 declares triton >= 3.3
# (fla/utils/_device.py, warning only), and the cu124 index stops at torch 2.6.0, which pins triton 3.2.0.
# Below that floor the damage is quiet: fla's get_available_device() swallows whatever the platform probe
# raised on 3.2, warns "roll back to CPU", @cache's that answer for the rest of the process, and transformers
# silently switches to the reference O(N^2) gated-delta-rule -- on an A40 that is both slow and an
# out-of-memory at 4B, which is exactly how the first real run died.  torch 2.7.1 (cu126) pins triton 3.3.1,
# and triton 3.3 still compiles sm_86 with ptxas 12.4.99 -- the same ptxas as 3.2.0, so the generated kernels
# do not change, only the floor is met.  cu128 (triton 3.4, ptxas 12.8) also runs on this driver, but it moves
# the codegen for no benefit here; CUDA 13 would need driver >= 580 and is out.
#
# There is nothing to compile here -- no llama.cpp, no CUDA kernels -- so the build is one stage and one
# index.  flash-linear-attention's Triton kernels JIT-compile against the target GPU the first time a hybrid
# layer runs them; that is a *run-time* cost on the machine that owns the GPU, which is why TRITON_CACHE_DIR
# points at a path meant to be mounted and kept (see below).
#
# Build
#   docker build -t decider-torch:cu126 .
#
# Run -- HTTP API, model from the Hub.  Without the /hf-cache volume every fresh container re-downloads the
# checkpoint; without /triton-cache every fresh container re-JITs the kernels.
#   docker run --rm --gpus all -p 8000:8000 \
#     -v decider-hf:/hf-cache -v decider-triton:/triton-cache \
#     decider-torch:cu126 scripts/serve.sh Mapika/decider-2b 8000
#
# Run -- a local checkpoint folder: nothing is downloaded and nothing needs to persist.
#   docker run --rm --gpus all -p 8000:8000 -v /srv/decider-2b:/models/decider-2b:ro \
#     decider-torch:cu126 scripts/serve.sh /models/decider-2b 8000
#
# Run -- upstream's own scripts/serve.sh, which is also in the image at /opt/decider/scripts/serve.sh.  It
# works unmodified here because this image presets the two things that assume upstream's dev checkout:
# DECIDER_HOST (its default 127.0.0.1 is unreachable through a published port) and UVICORN (its default
# .venv312/bin/uvicorn only exists after upstream's own `uv venv`).
#   docker run --rm --gpus all -p 8000:8000 -v decider-hf:/hf-cache \
#     decider-torch:cu126 /opt/decider/scripts/serve.sh Mapika/decider-2b 8000
#
# The default command is `tail -f /dev/null`; a command after the image name replaces it entirely.

ARG PYTHON_VERSION=3.12
ARG DECIDER_REPO=https://github.com/Mapika/decider
ARG DECIDER_REF=main
# decider-ai requires a bare, unpinned `torch`, so the wheel is installed first and the rest of the
# dependency graph reuses it instead of pulling a different build in behind it.  2.7.1 is the oldest wheel
# that pins triton >= 3.3 (it asks for 3.3.1); 2.8.0 would ask for 3.4.0 and compile sm_86 with ptxas 12.8.
# cu124 tops out at 2.6.0, whose triton 3.2.0 is below flash-linear-attention's floor.
# Pin exactly: --build-arg TORCH_SPEC=torch==2.7.1; whatever lands is printed into the build log.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_SPEC=torch==2.7.1
# causal_conv1d: no sdist builds without nvcc, so this is the prebuilt wheel, matched to the exact
# (torch minor, cpython, C++ ABI) triple that ended up installed.  Bump the tag to move versions.
ARG CCV_VERSION=1.7.0
ARG CCV_TAG=v1.7.0

FROM python:${PYTHON_VERSION}-slim-bookworm

ARG DECIDER_REPO
ARG DECIDER_REF
ARG TORCH_INDEX
ARG TORCH_SPEC
ARG CCV_VERSION
ARG CCV_TAG

# HF_HOME and TRITON_CACHE_DIR are not library defaults -- they are this image's choice of where the two caches
# live, and the volumes below mount exactly these paths.  Left alone, huggingface_hub would use
# ~/.cache/huggingface and Triton $TRITON_HOME/.triton/cache, i.e. /root/.triton/cache as root: both inside the
# container's writable layer, so both die with the container and a long mount: nothing notices.
# scripts/check_image.py asks each library where it actually caches rather than reading these back, so a library
# that stops honouring one of them fails the build instead of silently dropping the cache on the floor.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/hf-cache \
    TRITON_CACHE_DIR=/triton-cache \
    DECIDER_MODEL=Mapika/decider-2b \
    DECIDER_HOST=0.0.0.0 \
    UVICORN=uvicorn

# libgomp1 is torch's OpenMP runtime and the Debian slim base does not carry it.  curl+ca-certificates are
# for the source tarball; git is deliberately not installed (see the source step).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libgomp1; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" /models

# torch first, from the CUDA index.  This is the whole CUDA user space this image will ever have.
RUN set -eux; \
    pip install --index-url "${TORCH_INDEX}" "${TORCH_SPEC}"

# Upstream source, shallow, without git.  GitHub serves archive/<ref>.tar.gz for a branch, a tag and a commit
# SHA alike, so the first URL covers the documented "branch, tag or SHA" contract; the refs/heads and
# refs/tags fallbacks are for a ref that contains a slash.  Nothing is fetched at run time.
RUN set -eux; \
    mkdir -p /opt/decider; \
    ok=""; \
    for url in \
      "${DECIDER_REPO%/}/archive/${DECIDER_REF}.tar.gz" \
      "${DECIDER_REPO%/}/archive/refs/heads/${DECIDER_REF}.tar.gz" \
      "${DECIDER_REPO%/}/archive/refs/tags/${DECIDER_REF}.tar.gz"; do \
      if curl -fsSL --retry 3 --retry-delay 2 "$url" -o /tmp/src.tar.gz; then ok="$url"; break; fi; \
      echo "not an archive: $url" >&2; \
    done; \
    test -n "$ok" || { echo "no archive for ${DECIDER_REPO} @ ${DECIDER_REF}" >&2; exit 1; }; \
    echo "source: $ok"; \
    tar -xzf /tmp/src.tar.gz --strip-components=1 -C /opt/decider; \
    rm -f /tmp/src.tar.gz; \
    test -f /opt/decider/pyproject.toml

# decider-ai[serve]: the package + fastapi/uvicorn[standard]/httpx for decider.serve.  The gguf extra is not
# installed -- this image is the torch path, and llama.cpp would be dead weight.  Nothing compiles: the
# package is pure Python and its dependencies ship wheels.
RUN set -eux; \
    pip install "/opt/decider[serve]"

# causal_conv1d, the fourth piece of the Qwen3.5 hybrid layer and the one that has to be binary-matched: the
# wheel is pinned to a torch minor, a cpython tag and torch's C++ ABI, so the name is derived here instead of
# written down.  A wrong name is a 404, and a wrong ABI is an import error the check_image step would catch --
# either way better than the alternative, which is transformers quietly using its reference conv.
#   torch >= 2.7 wheels are CXX11_ABI=1; the other spelling is tried once rather than assumed.
RUN set -eux; \
    torch_xy="$(python -c 'import torch;print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')"; \
    abi="$(python -c 'import torch;print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"; \
    py="$(python -c 'import sys;print("cp%d%d" % sys.version_info[:2])')"; \
    base="https://github.com/Dao-AILab/causal-conv1d/releases/download/${CCV_TAG}"; \
    ok=""; \
    for a in "$abi" "$(if [ "$abi" = TRUE ]; then echo FALSE; else echo TRUE; fi)"; do \
      url="${base}/causal_conv1d-${CCV_VERSION}+cu12torch${torch_xy}cxx11abi${a}-${py}-${py}-linux_x86_64.whl"; \
      if pip install --no-deps "$url"; then ok="$url"; break; fi; \
      echo "no wheel: $url" >&2; \
    done; \
    test -n "$ok" || { echo "no causal_conv1d ${CCV_VERSION} wheel for torch ${torch_xy} / ${py} / ${abi}" >&2; exit 1; }; \
    echo "causal_conv1d: $ok"

# The upstream has no lockfile, so the resolver's choices are only pinned down here.  Printed into the build
# log and, in the workflow, into the artifact manifest.
RUN python -c "import importlib.metadata as m,sys;print('python='+sys.version.split()[0]);print(' '.join(p+'='+m.version(p) for p in ('decider-ai','torch','transformers','flash-linear-attention','triton','causal_conv1d','numpy','fastapi','uvicorn')))"

COPY scripts /app/scripts

# The contract this image promises, asserted where no GPU is available.  scripts/check_image.py is the single
# source of truth -- including the syntax check on serve.sh: the build runs it here and the workflow runs the
# same file in the image it loads back from the artifact, so the two cannot drift apart.
RUN set -eux; \
    chmod +x /app/scripts/serve.sh; \
    python /app/scripts/check_image.py

LABEL org.opencontainers.image.title="decider-torch" \
      org.opencontainers.image.description="decider (Mapika/decider) on the torch engine: decider.serve over CUDA 12.6 / cu126, torch 2.7.1, triton 3.3.1; weights are not included" \
      org.opencontainers.image.source="https://github.com/Mapika/decider" \
      org.opencontainers.image.revision="${DECIDER_REF}" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Informational only (docker run -P would publish it); scripts/serve.sh defaults to 0.0.0.0:8000.
EXPOSE 8000

# The default command is a no-op: it keeps the container up so `docker exec` can drive it, and a command
# after the image name replaces it entirely.
ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]
