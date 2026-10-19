#!/usr/bin/env bash
# decider over HTTP, on the torch engine.   scripts/serve.sh [model] [port]
#
#   model  a Hub repo id or a local checkpoint folder   (default: $DECIDER_MODEL, else Mapika/decider-2b)
#   port   listen port                                   (default: 8000)
#
# Runs upstream's server -- `uvicorn decider.serve:app` -- on decider.engine_v2.EngineV2: bf16, shape-bucketed
# CUDA graphs, dynamic batching (decider.batching.plan_batches), shared-prefix scoring.  Same five routes as
# upstream: POST /decide, POST /v1/systemone, GET /health /v1/models /stats.
#
#   docker run --rm --gpus all -p 8000:8000 \
#     -v decider-hf:/hf-cache -v decider-triton:/triton-cache \
#     decider-torch:cu126 scripts/serve.sh Mapika/decider-2b 8000
#
# DECIDER_HOST defaults to 0.0.0.0, unlike upstream's script: 127.0.0.1 inside a container is not reachable
# through a published port, so that default only looks like a hung port mapping.  There is NO authentication
# -- publish the port to a trusted network only, or set DECIDER_HOST=127.0.0.1 and reach the API through
# `docker exec`.
#
# No argument is forwarded to uvicorn on purpose: more than one worker would put several processes behind one
# GPU, each loading its own copy of the checkpoint, which is the opposite of what the batcher is for.
#
# Volumes that pay for themselves:
#   /hf-cache       HF_HOME.  A Hub repo id is about 4 GB for the 2B in bf16 and 8.4 GB for the 4B; without
#                   this volume every fresh container downloads it again.  A local folder never touches it.
#   /triton-cache   TRITON_CACHE_DIR.  flash-linear-attention's Triton kernels JIT-compile against the target
#                   GPU and the cubins are machine-specific, so they cannot be baked into the image.
#
# A cold start is not quick even with both volumes: the kernels compile once per GPU, and decider.serve then
# captures its whole CUDA graph grid before it answers anything (docs/SERVING.md).  DECIDER_WARMUP=0 skips the
# capture and runs every request eager.
set -euo pipefail

model="${1:-${DECIDER_MODEL:-Mapika/decider-2b}}"
port="${2:-8000}"

export DECIDER_MODEL="$model"
export DECIDER_HOST="${DECIDER_HOST:-0.0.0.0}"

case "$model" in
  *.gguf)
    echo "[serve.sh] '$model' is a GGUF file and this image runs the torch engine, which cannot read one." >&2
    echo "[serve.sh] Pass the checkpoint folder (safetensors + tokenizer) or its Hub repo id instead." >&2
    exit 2
    ;;
esac

echo "[serve.sh] model=$model host=$DECIDER_HOST port=$port uvicorn=$(command -v uvicorn || echo MISSING)" >&2

# decider.serve reads decider_config.json (layout, temperatures, schema_first) through a try/except that
# swallows every error, so a checkpoint without one runs at temperature 1.0 in the plain layout and reports
# nothing at all.  Say something instead.  It is legitimately absent for a stock checkpoint read with
# DECIDER_LAYOUT=chat, and a mistake for a decider model that ships one.
if [ "${DECIDER_CONFIG_CHECK:-1}" = "1" ]; then
  python - "$model" <<'PY' >&2
import json, os, sys
keys = ("layout", "temperature", "temperature_by_type", "schema_first", "isolated_levels", "neutralize_none")
m = sys.argv[1]
try:
    if os.path.isdir(m):
        path = os.path.join(m, "decider_config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(m, "decider_config.json")
    cfg = json.load(open(path))
except Exception as e:
    print(f"[serve.sh] no decider_config.json for {m} ({type(e).__name__}): layout=plain, temperature=1.0, "
          "no temperature_by_type, schema cache off -- and decider.serve will not report this.", file=sys.stderr)
    print("[serve.sh] Expected for a stock checkpoint read with DECIDER_LAYOUT=chat; a mistake for a decider "
          "model that ships one.", file=sys.stderr)
else:
    print("[serve.sh] decider_config.json: "
          + json.dumps({k: cfg[k] for k in keys if k in cfg}, ensure_ascii=False), file=sys.stderr)
PY
fi

exec uvicorn decider.serve:app --host "$DECIDER_HOST" --port "$port"
