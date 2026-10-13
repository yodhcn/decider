"""decider over HTTP, on the llama.cpp / GGUF engine.

Upstream decider.serve is the *torch* engine (decider.engine_v2.EngineV2, hard-wired in its _start()) and does
not serve GGUF -- its own module docstring says "The HTTP server does not serve GGUF files", and docs/SERVING.md
never mentions GGUF.  This is the GGUF counterpart, for the path this image is built around: the same routes and
the same wire format, so a client written against decider.serve works unchanged, but every forward runs in
llama.cpp through decider.engine_gguf.GGUFEngine, driven by decider.infer.Decider.

Routes (as decider.serve):
  POST /decide        {"context": str, "schema": {...}}    -> typed decisions, all questions in one row
  POST /v1/systemone  {"state": ..., "questions": {...}}   -> the TypeSafe / Jev wire format
  GET  /health  /v1/models  /stats

Left out on purpose, because they need the torch engine:
  * the schema cache (decider.schema_engine; Decider.schema() raises NotImplementedError for GGUF)
  * CUDA graphs, torch.compile and FP8
  * multi-row batching: GGUFEngine defaults to n_seq_max=1, one row per llama_decode, which is the repeatable
    path its measurements were made on (n_seq_max > 1 moves a row's probabilities with its neighbours)
  * the batching/merging planner of decider.batching: one request is scored at a time on one worker

Environment:
  DECIDER_MODEL             .gguf file, a folder holding one, or a Hub repo id  (/models/decider.gguf)
  DECIDER_GGUF_FILE         the .gguf inside that folder/repo; only needed when there is more than one
  DECIDER_HOST              listen address                               (0.0.0.0 -- see scripts/serve.sh)
  DECIDER_PORT              8000
  DECIDER_N_CTX             40960   llama.cpp context, i.e. the longest row in tokens
  DECIDER_N_GPU_LAYERS      -1      -1 offloads every layer (CUDA build + --gpus all), 0 runs on the CPU
  DECIDER_N_THREADS         unset   llama.cpp CPU threads
  DECIDER_MAX_STATE_TOKENS  32768   /v1/systemone state cut, the value decider.serve uses
  DECIDER_MAX_FWD_TOKENS    65536   token budget of one scoring call
  DECIDER_DECIDE_MAX_CTX    1536    /decide context cap, the value decider.serve uses
  DECIDER_MAX_QUEUE         256     requests admitted and not yet answered, over the whole server
  DECIDER_LOG_LEVEL         info    uvicorn log level
"""
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL = os.environ.get("DECIDER_MODEL", "/models/decider.gguf")
GGUF_FILE = os.environ.get("DECIDER_GGUF_FILE") or None
HOST = os.environ.get("DECIDER_HOST", "0.0.0.0")
PORT = int(os.environ.get("DECIDER_PORT", "8000"))
N_CTX = int(os.environ.get("DECIDER_N_CTX", "40960"))
N_GPU_LAYERS = int(os.environ.get("DECIDER_N_GPU_LAYERS", "-1"))
N_THREADS = os.environ.get("DECIDER_N_THREADS")
MAX_STATE_TOKENS = int(os.environ.get("DECIDER_MAX_STATE_TOKENS", "32768"))
MAX_FWD_TOKENS = int(os.environ.get("DECIDER_MAX_FWD_TOKENS", "65536"))
DECIDE_MAX_CTX = int(os.environ.get("DECIDER_DECIDE_MAX_CTX", "1536"))
MAX_QUEUE = int(os.environ.get("DECIDER_MAX_QUEUE", "256"))

d = None          # decider.infer.Decider
ex = None         # one worker thread: the GGUF engine owns one context and one llama_batch, so forwards serialise
outstanding = 0
stats = dict(requests=0, decisions=0, errors=0, rejected=0)


# ---- model resolution ---------------------------------------------------------------------------------------
def _only_gguf(names, where):
    found = sorted(n for n in names if n.endswith(".gguf"))
    if len(found) != 1:
        raise SystemExit(f"{where} holds {len(found)} .gguf files {found}; set DECIDER_GGUF_FILE to choose one "
                         "(guessing would silently serve a different quantization)")
    return found[0]


def resolve_source(model, gguf_file):
    """-> (path, gguf_file) to hand to Decider, plus a label for the logs.

    A Hub model is passed on as repo id + gguf_file, not as the downloaded cache path: Decider derives the
    tokenizer folder from that path, and a file inside the Hub cache would leave the repo's decider_config.json
    (prompt layout, temperatures) behind.
    """
    if str(model).endswith(".gguf"):
        return str(model), None, str(model)
    if os.path.isdir(model):
        f = gguf_file or _only_gguf(os.listdir(model), model)
        return os.path.join(model, f), None, os.path.join(model, f)
    from huggingface_hub import HfApi
    f = gguf_file or _only_gguf(HfApi().list_repo_files(model), model)
    return model, f, f"{model}/{f}"


def _load():
    from decider.infer import Decider
    path, gguf_file, label = resolve_source(MODEL, GGUF_FILE)
    opts = dict(n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS)
    if N_THREADS:
        opts["n_threads"] = int(N_THREADS)
    print(f"[serve-gguf] loading {label} n_ctx={N_CTX} n_gpu_layers={N_GPU_LAYERS}", flush=True)
    return Decider(path, gguf_file=gguf_file, gguf_options=opts), label


def _info():
    """What /health and /stats report about the loaded model."""
    eng = getattr(d, "eng", None)
    cfg = getattr(eng, "cfg", None) or {}
    return dict(model=d.name, backend=cfg.get("backend", "llama.cpp"), gguf=cfg.get("gguf"), n_ctx=cfg.get("n_ctx"),
                n_seq_max=cfg.get("n_seq_max"), layout=d.layout, temperature=d.T, temperature_by_type=d.T_by_type)


# ---- workers ------------------------------------------------------------------------------------------------
def _decide_sync(context, schema):
    return d.decide_json(context, schema, max_ctx_tokens=DECIDE_MAX_CTX)


def _systemone_sync(state, questions, independent, layout):
    return d.system_one(state, questions, independent=independent, max_state_tokens=MAX_STATE_TOKENS,
                        max_fwd_tokens=MAX_FWD_TOKENS, layout=layout)


async def _run(fn, *args):
    """Call fn on the single GGUF worker.  One context and one llama_batch are shared, so a forward is not
    overlapped with another; a request arriving while the worker is busy waits its turn, bounded by MAX_QUEUE."""
    global outstanding
    if outstanding >= MAX_QUEUE:
        stats["rejected"] += 1
        raise HTTPException(503, f"server busy: {outstanding} requests queued, the limit is {MAX_QUEUE} "
                                 "(DECIDER_MAX_QUEUE); retry later")
    outstanding += 1
    try:
        return await asyncio.get_running_loop().run_in_executor(ex, fn, *args)
    finally:
        outstanding -= 1


def _http_error(e):
    """A caller error, not a server error.  As in decider.serve a ValueError/KeyError out of the request layer is
    a 422 that names the expected form; a row longer than the context is the 413 that GGUFEngine._groups raises
    for, and the 2..MAX_OPTIONS assert of Decider._decide_items is a caller error too."""
    msg = str(e) or type(e).__name__
    return HTTPException(413 if "exceeds n_ctx" in msg else 422, msg)


@asynccontextmanager
async def lifespan(app):
    global d, ex
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gguf")
    loop = asyncio.get_running_loop()
    d, label = await loop.run_in_executor(ex, _load)
    print("[serve-gguf] ready " + json.dumps(dict(_info(), source=label), default=str), flush=True)
    yield
    ex.shutdown(wait=False, cancel_futures=True)
    ex = None


app = FastAPI(title="decider-gguf", lifespan=lifespan)


# ---- routes, wire-compatible with decider.serve --------------------------------------------------------------
class Req(BaseModel):
    context: str
    schema_: object = None                  # any JSON value: Decider._check_schema gives the 422
    model_config = {"populate_by_name": True}

    def __init__(self, **kw):
        if "schema" in kw:
            kw["schema_"] = kw.pop("schema")
        super().__init__(**kw)


class S1Req(BaseModel):
    state: object
    questions: dict
    model: str | None = None                # accepted and ignored, as in decider.serve
    independent: bool = True
    layout: str | None = None               # "schema_first" runs uncached here (no prefix cache in GGUFEngine)


@app.post("/decide")
async def decide(r: Req):
    try:
        out = await _run(_decide_sync, r.context, r.schema_)
    except (ValueError, KeyError, AssertionError) as e:
        stats["errors"] += 1
        raise _http_error(e) from None
    stats["requests"] += 1
    stats["decisions"] += len(out)
    return out


@app.post("/v1/systemone")
async def systemone(r: S1Req):
    try:
        out = await _run(_systemone_sync, r.state, r.questions, r.independent, r.layout)
    except (ValueError, KeyError, AssertionError) as e:
        stats["errors"] += 1
        raise _http_error(e) from None
    stats["requests"] += 1
    stats["decisions"] += len(out.get("answers", {}))
    return out


@app.get("/health")
async def health():
    ok = d is not None and ex is not None
    return dict(ok=ok, device=getattr(d, "dev", None), **( _info() if ok else {}))


@app.get("/v1/models")
async def models():
    return {"models": [{"name": d.name, "backend": "llama.cpp",
                        "description": "decider on llama.cpp (GGUF): one-pass typed decisions with calibrated probabilities"}]}


@app.get("/stats")
async def get_stats():
    eng = getattr(d, "eng", None)
    return dict(stats, outstanding_requests=outstanding, engine=getattr(eng, "stats", None),
                gguf=(eng.cfg.get("gguf") if eng else None),
                limits=dict(max_queue=MAX_QUEUE, n_ctx=N_CTX, max_state_tokens=MAX_STATE_TOKENS))


def main():
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.environ.get("DECIDER_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
