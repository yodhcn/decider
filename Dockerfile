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
#   in the image ... decider-ai[serve] at DECIDER_VERSION, the release wheel from PyPI (not the git tree)
#                    torch 2.7.1 / CUDA 12.6 / cuDNN 9 and triton 3.3.1 from the base image, causal_conv1d
#                    scripts/{serve.sh,check_image.py}
#
# Why pytorch/pytorch, and not python:3.x-slim plus a pip-installed torch.  torch and the CUDA user space under
# it are one unit: cudart, cuBLAS, cuDNN, NCCL and cuDNN's kernel selection are versioned with the torch build,
# and a resolver choosing nvidia-*-cu12 wheels to sit under a torch wheel is a combination nobody tested
# together.  pytorch/pytorch *is* the combination its maintainers build, test and publish as one.  So this image
# starts from it and never installs torch itself -- the decider install step fails the build if anything in
# decider's dependency graph replaces it.
#
# Why the released wheel and not the git tree.  decider-ai is on PyPI, and its wheel is `py3-none-any` with
# decider.serve, decider.engine_v2 and the rest of the package inside it -- the only things the repository has
# that the wheel does not are scripts/ and docs/, and this image carries its own entry script.  So the tarball
# step bought nothing and cost something: a URL that can 404, a ref that moves, a second copy of the source to
# keep in step with the wheel, and no statement of which release the image actually runs.  Pinning
# DECIDER_VERSION to an exact release makes the image reproducible from the tag alone, and check_image.py
# asserts that the version installed is the one asked for rather than trusting pip's resolver.
#
# -runtime and not -devel: the compiler toolchain is only needed to compile CUDA extensions, and nothing here
# compiles.  causal_conv1d arrives as a prebuilt wheel, and flash-linear-attention's kernels are compiled at run
# time by triton, which carries its own ptxas.
#
# What that costs, plainly: the image is several GB larger than a slim base plus torch wheels, because the base
# carries the whole CUDA 12.6 runtime and a conda environment; and Python comes from that environment rather than
# from a version this Dockerfile picks.  decider-ai requires >= 3.11, so that is fine, and check_image.py asserts
# it rather than trusting the tag -- the same version also names the causal_conv1d wheel, which is derived at
# build time.  Everything else is unchanged: no weights, two volumes, upstream's own server.
#
# Why this torch, now that decider is pip-installed.  Pip does not get to pick it: decider-ai declares a bare
# `torch`, and flash-linear-attention's floors for it (torch >= 2.7, triton >= 3.3) sit inside
# `fla-core[cuda]` -- an extra nobody asks for -- so the resolver enforces neither.  Whatever PYTORCH_IMAGE
# ships is the whole answer, which is why check_image.py asserts both floors instead of trusting pip.
#
# Tag census at this revision, taken from the wheels' own metadata rather than a blog post:
#   2.7.1  -> triton 3.3.1, cudart 12.6.77, cuDNN 9.5.1    (this image)
#   2.8.0  -> triton 3.4.0, cudart 12.8.90, cuDNN 9.10.2   (also runs on this driver)
#   2.9.x  -> triton 3.5.x, same 12.8;  2.10.0 -> triton 3.6.0, same 12.8
#   2.14.x -> cu13 (nvidia-cudnn-cu13)                     (needs driver >= 580: out)
# causal_conv1d 1.7.0 publishes prebuilt wheels for cu12torch2.7 and cu12torch2.8 (cp311-cp313, abiTRUE) and
# that is what really caps the choice: above 2.8 there is no conv wheel to match, and without one transformers
# silently drops to its reference convolution in every hybrid layer.
#
# So the choice is 2.7.1 or 2.8.0, and cu126/2.7.1 stays.  CUDA 12.x is minor-version compatible, so the
# 550.90.12 driver is not the constraint; triton is, and triton 3.3 compiles sm_86 with ptxas 12.4.99 -- the
# same ptxas as the 3.2.0 that produced the first real run -- so moving to 3.3 changes the floor and nothing
# else.  2.8.0 buys headroom over two floors that have not moved, and charges for it by moving sm_86 codegen to
# ptxas 12.8.  When a fla release does raise a floor, that check fails the build and the fix is one documented
# build arg, not a rewrite.
#
# And why not cu124 + torch 2.6.0, which is what this image used to be: fla needs triton >= 3.3 and cu124 stops
# at torch 2.6.0 / triton 3.2.0.  Below the floor the damage is quiet -- fla's get_available_device() swallows
# the probe's exception, warns "roll back to CPU", @cache's that answer, and transformers switches to the
# reference O(N^2) gated-delta-rule: slow, and an OOM at 4B, which is exactly how the first real run died.
#
# There is still nothing to compile, so the build stays one stage.  flash-linear-attention's Triton kernels
# JIT-compile against the target GPU the first time a hybrid layer runs them; that is a *run-time* cost on the
# machine that owns the GPU, which is why TRITON_CACHE_DIR points at a path meant to be mounted and kept.
#
# Build
#   docker build -t decider-torch:cu126 .
#   docker build --build-arg DECIDER_VERSION=1.9.0 -t decider-torch:cu126-1.9.0 .
#   docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -t decider-torch:cu128 .
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
# scripts/serve.sh is this repository's, not upstream's: the wheel does not ship the repository's scripts/, and
# this one presets DECIDER_HOST=0.0.0.0 (upstream's 127.0.0.1 is unreachable through a published port) and
# warns when a checkpoint has no decider_config.json.
#
# The default command is `tail -f /dev/null`; a command after the image name replaces it entirely.

# The base tag names torch, the CUDA minor and cuDNN at once, so it is the one place those versions live.
# Overriding it is supported -- check_image.py still holds the line on what has to be true (CUDA 12.x, triton at
# or above flash-linear-attention's floor, torch's CUDA libraries resolvable).
ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
# The decider release, exact.  Not a range: the point of pinning is that the artifact names the release it
# runs, and check_image.py compares the installed distribution against this value.
ARG DECIDER_VERSION=1.9.0
# causal_conv1d: no sdist builds without nvcc, so this is the prebuilt wheel, matched to the exact
# (torch minor, cpython, C++ ABI) triple that ended up installed.  Bump the tag to move versions.
ARG CCV_VERSION=1.7.0
ARG CCV_TAG=v1.7.0

FROM ${PYTORCH_IMAGE}

ARG DECIDER_VERSION
ARG CCV_VERSION
ARG CCV_TAG

# HF_HOME and TRITON_CACHE_DIR are not library defaults -- they are this image's choice of where the two caches
# live, and the volumes below mount exactly these paths.  Left alone, huggingface_hub would use
# ~/.cache/huggingface and Triton $TRITON_HOME/.triton/cache, i.e. /root/.triton/cache as root: both inside the
# container's writable layer, so both die with the container and a long mount: nothing notices.
# scripts/check_image.py asks each library where it actually caches rather than reading these back, so a library
# that stops honouring one of them fails the build instead of silently dropping the cache on the floor.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/hf-cache \
    TRITON_CACHE_DIR=/triton-cache \
    DECIDER_MODEL=Mapika/decider-2b \
    DECIDER_HOST=0.0.0.0 \
    UVICORN=uvicorn
# Read back by check_image.py (installed dist == declared release) and by `docker inspect`.  decider itself
# defines no DECIDER_VERSION, so this cannot collide with a runtime knob.
ENV DECIDER_VERSION=${DECIDER_VERSION}

# One apt step, and it is not optional.  torch's libnccl.so.2 lists libibverbs.so.1, librdmacm.so.1 and
# libmlx5.so.1 as DT_NEEDED -- NVIDIA links NCCL's InfiniBand transport in rather than dlopen-ing it -- so
# libtorch_cuda.so cannot be loaded without them.  torch defers that load until the first CUDA call, and the
# symptom is not an ImportError: it is torch.cuda.is_available() returning False and the engine quietly running
# on the CPU, which is the exact silent downgrade this whole image is built to avoid.  A single A40 never touches
# the IB transport; the dynamic linker cannot know that.  No compiler, no headers, no -dev packages: ~3 MB of
# runtime libraries, and the apt lists are removed in the same layer.
#
# Everything else an apt step used to install is either already in the base or caught a line later: the ubuntu
# base carries ca-certificates (and curl, which this Dockerfile no longer needs), and libgomp1 -- torch's OpenMP
# runtime -- is resolved by the `import torch` below before pip runs at all.  If a future base drops either, that
# import fails here with the library's name, and check_image.py's ldd sweep would fail on `libgomp.so.1: not
# found` rather than letting the first forward on the GPU discover it.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libibverbs1 \
      librdmacm1 \
      ibverbs-providers; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" /models

# Two things, one layer: say what the base gave us, then install decider on top of it.
#
# The probe fails here rather than two gigabytes of pip later if PYTORCH_IMAGE was overridden with a CPU build.
# Nothing installs torch after this line: the comparison around the install asserts that decider's dependency
# graph left it alone, which is what makes "torch comes from the base image" a fact instead of a hope.
#
# decider-ai[serve] from PyPI: the package + fastapi/uvicorn[standard]/httpx for decider.serve.  The gguf extra
# is not installed -- this image is the torch path, and llama.cpp would be dead weight.  `==` and not a range,
# so the layer is a function of DECIDER_VERSION alone; the wheel is pure python (py3-none-any), so there is no
# build backend and no compiler involved.
#
# decider-ai declares a bare, unpinned `torch`, which the base image already satisfies: pip installs what is
# missing and leaves torch where it is.  What it *does* pull and the base does not have is transformers (its own
# floor is >= 5) and flash-linear-attention -- neither is pinned, so a rebuild moves them; check_image.py prints
# the versions that landed instead of pretending they are fixed.
RUN set -eux; \
    python -c "import sys, torch; print('base: torch', torch.__version__, 'cuda', torch.version.cuda, 'python', sys.version.split()[0]); sys.exit(0 if torch.version.cuda else 1)"; \
    before="$(python -c 'import torch;print(torch.__version__)')"; \
    pip install "decider-ai[serve]==${DECIDER_VERSION}"; \
    after="$(python -c 'import torch;print(torch.__version__)')"; \
    [ "$before" = "$after" ] || { \
      echo "installing decider-ai replaced torch ($before -> $after); the base image's torch is the whole point of using pytorch/pytorch" >&2; \
      exit 1; \
    }; \
    echo "torch untouched by the decider install: $after"

# causal_conv1d, the fourth piece of the Qwen3.5 hybrid layer and the one that has to be binary-matched: the
# wheel is pinned to a torch minor, a cpython tag and torch's C++ ABI, so the name is derived here instead of
# written down.  A wrong name is a 404, and a wrong ABI is an import error the check_image step would catch --
# either way better than the alternative, which is transformers quietly using its reference conv.
#   torch >= 2.7 wheels are CXX11_ABI=1 and only TRUE is published for cu12torch2.7; the other spelling is
#   tried once rather than assumed, so a base with a different ABI still resolves.
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

# DECIDER_VERSION pins decider, but its own dependencies are loose floors (transformers>=5, flash-linear-
# attention, numpy, torch), so what actually landed is worth printing.  check_image.py does that a few lines
# below -- one report, in the same step that asserts them, instead of a separate layer that only echoes.

COPY scripts /app/scripts

# The contract this image promises, asserted where no GPU is available.  scripts/check_image.py is the single
# source of truth -- including the syntax check on serve.sh: the build runs it here and the workflow runs the
# same file in the image it loads back from the artifact, so the two cannot drift apart.
RUN set -eux; \
    chmod +x /app/scripts/serve.sh; \
    python /app/scripts/check_image.py

LABEL org.opencontainers.image.title="decider-torch" \
      org.opencontainers.image.description="decider (Mapika/decider) on the torch engine: decider.serve on pytorch/pytorch 2.7.1 / CUDA 12.6 / triton 3.3.1; weights are not included" \
      org.opencontainers.image.source="https://github.com/Mapika/decider" \
      org.opencontainers.image.version="${DECIDER_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Informational only (docker run -P would publish it); scripts/serve.sh defaults to 0.0.0.0:8000.
EXPOSE 8000

# The default command is a no-op: it keeps the container up so `docker exec` can drive it, and a command
# after the image name replaces it entirely.
ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]
