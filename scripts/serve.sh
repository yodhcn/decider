#!/usr/bin/env bash
# decider over HTTP, on llama.cpp (GGUF).   scripts/serve.sh [model] [port]
#
#   model  a .gguf file, a folder holding one, or a Hub repo id   (default: $DECIDER_MODEL, else /models/decider.gguf)
#   port   listen port                                            (default: 8000)
#
#   docker run --rm --gpus all -p 8000:8000 -v decider-hf:/hf-cache \
#     decider-gguf:cu12.4.1 scripts/serve.sh Mapika/decider-4b-GGUF 8000
#
# Upstream's own scripts/serve.sh starts decider.serve, which is the *torch* engine and cannot read a GGUF
# checkpoint.  This image is built around llama.cpp, so it ships scripts/serve_gguf.py instead: the same routes
# and the same wire format (POST /decide, POST /v1/systemone, GET /health, /v1/models, /stats), served by
# decider.infer.Decider on GGUFEngine.
#
# Knobs (see the header of scripts/serve_gguf.py for the full list):
#   DECIDER_MODEL, DECIDER_GGUF_FILE, DECIDER_N_CTX, DECIDER_N_GPU_LAYERS, DECIDER_N_THREADS,
#   DECIDER_MAX_STATE_TOKENS, DECIDER_MAX_FWD_TOKENS, DECIDER_MAX_QUEUE, DECIDER_LOG_LEVEL
#
# DECIDER_HOST defaults to 0.0.0.0 rather than 127.0.0.1, unlike the upstream script: 127.0.0.1 inside a
# container is not reachable through a published port, so that default only looks like a hung port mapping.
# There is NO authentication -- publish the port to a trusted network only, or set DECIDER_HOST=127.0.0.1 and
# reach the API through `docker exec`.
#
# No arguments are forwarded to uvicorn on purpose: more than one worker would put several processes behind one
# llama.cpp context, which is exactly what this server serialises instead.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
model="${1:-${DECIDER_MODEL:-/models/decider.gguf}}"
port="${2:-8000}"

export DECIDER_MODEL="$model"
export DECIDER_PORT="$port"
export DECIDER_HOST="${DECIDER_HOST:-0.0.0.0}"

echo "[serve.sh] model=$DECIDER_MODEL host=$DECIDER_HOST port=$port" >&2
exec python "$here/serve_gguf.py"
