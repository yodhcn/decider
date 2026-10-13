# decider-gguf

把 [Mapika/decider](https://github.com/Mapika/decider) 的 **llama.cpp / GGUF** 部署路径打成 Docker 镜像，
并补上一个走 llama.cpp 的 **HTTP API**；用 GitHub Actions 在 ubuntu runner 上构建、导出 `tar.gz` 上传为
**workflow artifact**（不推任何 registry）。

目标宿主机：A40 / 驱动 550.90.12 / CUDA 12.4。

## 镜像里有什么

- 上游源码（`DECIDER_REF`）以 `decider-ai[gguf,serve]` 安装
- `llama-cpp-python >=0.3.35,<0.4`：**从源码编译**，`CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=86"`
- `scripts/serve.sh` + `scripts/serve_gguf.py`：**本仓库补的 GGUF HTTP 服务**（见下）
- Python 3.12 / Ubuntu 22.04 / `nvidia/cuda:12.4.1-runtime`

**不含**模型文件：`.gguf` 权重、tokenizer、`decider_config.json` 都由你挂载。

## 为什么 HTTP API 要自己补一个

上游的 HTTP 服务是 **torch 引擎**，读不了 GGUF。这不是配置问题，是写死的：`decider/serve.py:401` 的
`_start()` 里直接 `from decider.engine_v2 import EngineV2`，没有任何引擎选择；`docs/SERVING.md` 通篇不提
GGUF。上游 README 自己把两条都写明了：

> The GGUF files are not chat models: loading one in `llama-cli` or Ollama gives a text model, not decisions.

> The HTTP server does not serve GGUF files

原因是 decider 的读出头不是「让模型接着往下写」，而是把 answer slot 位置的 logits 投影到选项字母行上，
这一步只有 `decider/engine_gguf.py` 走 llama-cpp-python 的低层 C API（`llama_decode` + `llama_get_logits_ith`）
做。所以「llama.cpp 部署」只能是 `decider.infer.Decider` + `GGUFEngine`，起 `llama-server` 只会得到一个普通文本模型。

于是本仓库补了 `scripts/serve_gguf.py`：**路由和 wire 格式与 `decider.serve` 完全一致**，但每次 forward 都
在 llama.cpp 里，由 `Decider` + `GGUFEngine` 驱动。按 `decider.serve` 写的客户端不用改。

刻意没有搬过来的部分（都需要 torch 引擎）：

| | |
| --- | --- |
| schema cache（`decider.schema_engine`） | `Decider.schema()` 对 GGUF 直接 `raise NotImplementedError` |
| CUDA graphs / `torch.compile` / FP8 | 没有 torch 前向，这些参数在 GGUF 引擎上不存在 |
| 多行打包（`n_seq_max > 1`） | `GGUFEngine` 默认 `n_seq_max=1`：一次 `llama_decode` 一行，这是它测量时用的可复现路径；`n_seq_max > 1` 会让一行的概率随邻居漂移（BF16 达 0.02、Q4_K_M 达 0.16） |
| `decider.batching` 的合并规划 | 本服务同一时刻只算一个请求（单 worker 串行） |

## 拿到镜像

Actions → `docker-build-artifact` → Run workflow。产物：

```
decider-gguf-cu12.4.1-main-sm86-amd64.tar.gz
decider-gguf-cu12.4.1-main-sm86-amd64.tar.gz.sha256
decider-gguf-cu12.4.1-main-sm86-amd64.manifest.txt
```

```bash
sha256sum -c decider-gguf-cu12.4.1-main-sm86-amd64.tar.gz.sha256
gzip -dc decider-gguf-cu12.4.1-main-sm86-amd64.tar.gz | docker load
docker images | grep decider-gguf
```

artifact 是镜像的**唯一副本**（没有推 registry），默认保留 30 天，及时下载留存。

## 本地构建

```bash
docker build -t decider-gguf:cu12.4.1 .
docker build -t decider-gguf:cu12.4.1 --build-arg DECIDER_REF=v1.9.0
```

## 跑起来

### 1. 准备模型目录（镜像不含模型）

`.gguf` 和它的 tokenizer、`decider_config.json` 必须放在同一个目录（上游 `Decider` 就是这么解析的）。
两个官方 GGUF 仓库都带 Q4_K_M / Q8_0 / BF16 三个量化，所以**要显式指定文件名**：

```bash
huggingface-cli download Mapika/decider-4b-GGUF --local-dir /srv/decider-4b-GGUF
# /srv/decider-4b-GGUF/ 里应有 decider-4b-v2.1-Q4_K_M.gguf + tokenizer 文件 + decider_config.json
```

### 2. HTTP API（llama.cpp）

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /srv/decider-4b-GGUF:/models -v decider-hf:/hf-cache \
  decider-gguf:cu12.4.1 \
  scripts/serve.sh /models/decider-4b-v2.1-Q4_K_M.gguf 8000
```

`scripts/serve.sh [model] [port]`，`model` 可以是本地 `.gguf` 路径、装着 `.gguf` 的目录、或 HF 仓库 ID。
默认监听 `0.0.0.0`（不是上游脚本的 `127.0.0.1` —— 容器里的 `127.0.0.1` 穿不过 published port）。
**没有任何鉴权**，端口只发布给可信网络。

```bash
curl -s localhost:8000/health | python -m json.tool
curl -s localhost:8000/v1/models | python -m json.tool
```

`POST /decide` —— 一次前向，所有问题打成一个 prompt 行：

```bash
curl -s localhost:8000/decide -H 'content-type: application/json' -d '{
  "context": "My card was charged twice for the same purchase.",
  "schema": {
    "Which team?":      {"type": "choice", "options": ["billing", "technical", "sales"]},
    "Needs a refund?":  {"type": "bool"},
    "Urgency?":         {"type": "scale", "legend": {"0": "none", "1": "low", "2": "high"}}
  }
}' | python -m json.tool
```

```json
{
  "Which team?":     {"choice": "billing", "confidence": 0.97, "type": "choice", "probabilities": {"billing": 0.98, "sales": 0.01, "technical": 0.01}},
  "Needs a refund?": {"noul": 0.99, "type": "noul"},
  "Urgency?":        {"score": 1.42, "confidence": 0.71, "type": "scale", "legend": {"0": "none", "1": "low", "2": "high"}, "probabilities": {"0": 0.05, "1": 0.48, "2": 0.47}}
}
```

`POST /v1/systemone` —— TypeSafe / Jev 的 wire 格式，每个问题自己一行（`independent`，默认 `true`）：

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"ticket": {"subject": "double charge", "messages": [{"text": "charged twice"}]}},
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team should handle this?", "criteria": ["billing", "technical", "sales"]},
    "escalate": {"type": "noul", "instructions": "Does a human need to look at this today?"},
    "severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["none", "low", "medium", "high"]}
  }
}' | python -m json.tool
```

```json
{
  "model": "decider-2.1",
  "answers": {
    "team":     {"type": "choice", "choice": "billing", "confidence": 0.96, "x_p_max": 0.98, "certainty": 0.91, "probabilities": {"billing": 0.98, "technical": 0.01, "sales": 0.01}},
    "escalate": {"type": "noul", "noul": 0.93},
    "severity": {"type": "score", "score": 1.37, "confidence": 0.66, "x_p_max": 0.45, "certainty": 0.42, "legend": {"0": "none", "1": "low", "2": "medium", "3": "high"}, "probabilities": {"0": 0.03, "1": 0.44, "2": 0.45, "3": 0.08}}
  },
  "usage": {"input_tokens": 214, "output_tokens": 0}
}
```

`state` 可以是字符串、对象或数组（对象按 JSON 紧凑序列化，长数组会写入 `_index` 便于模型查表）；
`questions` 的完整格式见上游 `decider/systemone.py` 的模块注释。错误语义与上游一致：schema / 问题不合法是
**422**，一行超出 `n_ctx` 是 **413**，队列打满是 **503**。其余：`GET /health`、`GET /v1/models`、`GET /stats`。

### 3. 跑上游自带的 demo（三组 routing + 一个 10 字段 typed schema）

`python -m decider.infer <本地 .gguf 路径>` 是上游 `decider/infer.py` 的 `__main__`，拿本地 `.gguf` 直接可跑：

```bash
docker run --rm --gpus all -v /srv/decider-4b-GGUF:/models decider-gguf:cu12.4.1 \
  python -m decider.infer /models/decider-4b-v2.1-Q4_K_M.gguf
```

### 4. 跑自己的逻辑

```bash
docker run --rm --gpus all -it -v /srv/decider-4b-GGUF:/models decider-gguf:cu12.4.1 bash
```

```python
from decider.infer import Decider
d = Decider("/models/decider-4b-v2.1-Q4_K_M.gguf")
d.decide("My card was charged twice.", [{"question": "Which team?", "options": ["billing", "technical", "sales"]}])
d.system_one(state, questions)
```

容器默认命令是 `tail -f /dev/null`。**镜像名之后的命令会整体替换默认命令**；docker 自己的参数必须写在镜像名之前
（`docker run <img> python x.py --gpus all` 会把 `--gpus all` 当成 x.py 的参数），命令不经过 shell，要 `&&`、重定向或变量展开得显式套 `sh -c`。

`scripts/serve.sh` 与 `scripts/serve_gguf.py` 都在 `WORKDIR /app` 下，所以镜像名之后写 `scripts/serve.sh ...`
即可（绝对路径 `/app/scripts/serve.sh` 等价）。

### 5. 不落盘模型：仓库 ID + 缓存卷

仓库 ID 直接交给 `Decider`（它自己去 Hub 拉 `.gguf`、tokenizer 和 `decider_config.json`）：

```bash
docker run --rm --gpus all -p 8000:8000 -v decider-hf:/hf-cache \
  -e DECIDER_GGUF_FILE=decider-4b-v2.1-Q4_K_M.gguf \
  decider-gguf:cu12.4.1 scripts/serve.sh Mapika/decider-4b-GGUF 8000
```

（`-e` 和 `-v` 必须写在镜像名**之前**；写在后面会被当成 `scripts/serve.sh` 的参数。）

首次联网写入 `/hf-cache`（= 镜像里的 `HF_HOME`），之后可加 `-e HF_HUB_OFFLINE=1` 离线跑。
仓库里存着多个量化时，`DECIDER_GGUF_FILE` 是**必需**的；本服务不会替你猜（挑错量化 = 换了个模型）。
本地目录里只有一个 `.gguf` 时可以省略。

```python
# 同一件事，直接写代码
from decider.infer import Decider
d = Decider("Mapika/decider-4b-GGUF", gguf_file="decider-4b-v2.1-Q4_K_M.gguf")
d.decide("Hello", [{"question": "Language?", "options": ["en", "zh"]}])
```

### HTTP 服务的环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DECIDER_MODEL` | `/models/decider.gguf` | `.gguf` 路径 / 目录 / HF 仓库 ID（`scripts/serve.sh` 的第一个参数会覆盖它） |
| `DECIDER_GGUF_FILE` | 未设 | 目录或仓库里的 `.gguf` 文件名 |
| `DECIDER_HOST` | `0.0.0.0` | 监听地址；设 `127.0.0.1` 则只能 `docker exec` 进去访问 |
| `DECIDER_PORT` | `8000` | 监听端口（第二个参数会覆盖它） |
| `DECIDER_N_CTX` | `40960` | llama.cpp 上下文，即**单行最长 token 数** |
| `DECIDER_N_GPU_LAYERS` | `-1` | `-1` 全部层 offload 到 GPU（CUDA 构建 + `--gpus all`），`0` 走 CPU |
| `DECIDER_N_THREADS` | 未设 | llama.cpp 的 CPU 线程数 |
| `DECIDER_MAX_STATE_TOKENS` | `32768` | `/v1/systemone` 的 state 截断，与 `decider.serve` 同值 |
| `DECIDER_MAX_FWD_TOKENS` | `65536` | 单次打分调用的 token 预算 |
| `DECIDER_MAX_QUEUE` | `256` | 已受理未答复的请求数上限，超出回 503 |
| `DECIDER_LOG_LEVEL` | `info` | uvicorn 日志级别 |

## CUDA / 驱动对齐

| | |
| --- | --- |
| 基镜像 | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` |
| 为什么不是 24.04 | CUDA 12.4 **没有** Ubuntu 24.04 的官方镜像（24.04 从 12.5.1 才开始支持），所以是 22.04 |
| 为什么 Python 靠 PPA | 22.04 自带 Python 3.10，decider-ai 要 `>=3.11`，上游 CI 用 3.12，走 deadsnakes |
| CUDA 架构 | `sm_86` = Ampere = A40，`--build-arg LLAMA_CUDA_ARCH=` 可改（`native` / `all` / `80;86` 也收） |
| 驱动约束 | CUDA 12.x 的次版本兼容允许 **驱动 ≥ 525** 跑 12.x 用户态库，550.90.12 满足；容器用的是镜像自带的 CUDA 用户态（cudart / cuBLAS），不是宿主机的 |
| 为什么 `-runtime` 交付 | nvcc / 头文件 / 静态库只在构建阶段需要，运行阶段留着要多几 GB。这也是 llama.cpp 官方 `.devops/cuda.Dockerfile` 的做法：devel 构建 → runtime 交付（两边同为 CUDA 12.4.1 / Ubuntu 22.04 / Python 3.12，venv 的绝对路径与 ABI 不变） |
| 运行阶段额外装的 `libgomp1` | llama.cpp 的 OpenMP 运行时；`-runtime` 镜像里没有 gcc，不会自带 |

## 关于 torch：这里装的是 CPU 版

基镜像仍是 CUDA 12.4.1，但 venv 里的 torch 走 `--index-url https://download.pytorch.org/whl/cpu`，不是 PyPI 默认的 CUDA 版。理由：

1. **两条服务路径上所有 CUDA 计算都在 llama.cpp 里。** 无论 `python -m decider.infer`、`Decider` 还是本仓库的
   `scripts/serve_gguf.py`，都走 `GGUFEngine`；`decider/engine_gguf.py` 只拿 torch 做 `from_numpy` /
   `no_grad` / `softmax` / `masked_fill` / `split` 这类宿主侧小张量操作。
2. 走查过代码，这条链上没有无条件 CUDA 调用：`decider/infer.py:109` 只是 `torch.cuda.is_available()` 查询
   （GGUF 分支随后把 device 覆盖成 `"gguf"`），`prompt.py` / `temperature.py` / `systemone.py` / `model.py`
   里没有 `torch.cuda`。
3. 上游自己的 CI 就是这么测的：`.github/workflows/tests.yml` 的 `torch-cpu` job 用
   `pip install torch --index-url https://download.pytorch.org/whl/cpu` 跑完整套测试。
4. 省掉 torch 自带的 `nvidia-*-cu12` 用户态库（约 2 GB），更重要的是避免它在同一个进程里和基镜像的 CUDA 12.4
   争抢同一批 SONAME（`libcudart.so.12` / `libcublas.so.12`）。
5. 不会被依赖顶掉：`flash-linear-attention` 的 bare 依赖只有 `fla-core` + `einops` + `transformers`，
   `fla-core` 的 `torch>=2.7` 只挂在 `[cuda]` / `[cpu]` extra 上，所以先装 torch 不会被重新解析掉。

**代价**：`decider.serve`（上游那个 torch 引擎的 HTTP 服务）在这个镜像里跑不起来。但这不是损失 ——
它本来就服务不了 GGUF（见上文），GGUF 的 HTTP 由 `scripts/serve_gguf.py` 提供，而它只需要 CPU torch。

确要用 CUDA torch（比如你另外要跑 torch 引擎做对照）：

```bash
docker build -t decider-gguf:cu12.4.1-cudatorch --build-arg TORCH_INDEX=https://pypi.org/simple .
```

此时「镜像里只有一套 CUDA」就不再成立：PyPI 的 torch 轮子自带一套更新的 `nvidia-*-cu12`，和基镜像的 12.4 在同一个
进程里共存（12.x 主版本内 ABI 兼容，且两者 SONAME 相同会被动态链接器去重）。注意驱动 550.90.12 把 CUDA 上限锁在
12.4，而 **cu124 索引里 cp312 的 torch 最高只到 2.6.0**，更靠后的轮子是 cu126/cu128，需要更新的驱动。

## 可调 build-arg

| ARG | 默认 | 说明 |
| --- | --- | --- |
| `DECIDER_REF` | `main` | 上游分支 / tag / commit SHA（`git init` + `fetch --depth 1`，三种都收） |
| `DECIDER_REPO` | `https://github.com/Mapika/decider` | 上游仓库 |
| `LLAMA_CUDA_ARCH` | `86` | llama.cpp 的 CUDA 架构 |
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cpu` | torch 索引 |
| `TORCH_SPEC` | `torch` | torch 约束（默认不钉，与上游一致） |
| `CUDA_VERSION` | `12.4.1` | 基镜像 CUDA（同时作用于 builder 与 runtime） |
| `UBUNTU_VERSION` | `22.04` | 基镜像 Ubuntu |
| `PYTHON_VERSION` | `3.12` | Python |

## 已知限制

- **交付时本机没有 docker，也没有 GPU。** Dockerfile 与 workflow 只跑过静态校验；HTTP 服务是用一个
  `decider.infer.Decider` 桩在本地真跑过一遍（24 项断言：路由、wire 格式、422/413/503、模型解析规则），
  但**桩证明不了 llama.cpp 真实前向的任何数字**。下面这些都要靠第一次 CI 构建 / A40 上首次运行才算验过：
  - 构建期断言 `*cuda*` 落在 `llama_cpp/lib/`：llama.cpp 缺 CUDA 后端时不会报错、只会静默走 CPU，所以专门断言。
    失败会直接停在那行并打印该目录实际内容。
  - 构建期断言 `serve_gguf` 能 import 且五条路由都在：catch `serve` extra 没装上、或上游改了路由名。
  - `scripts/serve.sh` 只在 A40 上真跑一次才算验过。
- 构建时间：llama.cpp 的 CUDA 编译 + torch 下载，2 核 runner 上大致 30–60 分钟（`timeout-minutes: 120`，
  并先跑 `free-disk-space`，因为 -devel 层 + venv + tar 大约要 15 GB）。
- artifact 单件上限 **10 GB（硬限制）**，所以镜像是先 `docker save | pigz` 再上传，`compression-level: 0`
  避免二次压缩。
- 镜像以 root 运行；`/dev/nvidia*` 由 nvidia-container-toolkit 在容器启动前注入，不需要额外 `--privileged`。
- 没有 GPU 的 runner 上跑不了真实推理，所以 CI 只做 import / CUDA 后端 / 版本探测（探测失败不阻断上传）。
- HTTP 服务的并发是**串行**的：一个 worker、一个 llama.cpp context。想提高吞吐应该多开容器进程
  （各自一份 KV），而不是加 uvicorn worker —— 后者会让多个进程共用一份 context，正是这里刻意避免的。

## 文件

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | 镜像定义（devel 构建 → runtime 交付，`ENTRYPOINT []` + `CMD tail -f /dev/null`） |
| `scripts/serve.sh` | HTTP 服务入口：`scripts/serve.sh [model] [port]` |
| `scripts/serve_gguf.py` | GGUF HTTP 服务本体，路由与 wire 格式同 `decider.serve` |
| `.github/workflows/docker-build-artifact.yml` | 手动触发，构建 → `docker save \| gzip` → 上传 artifact |
| `.dockerignore` | 构建上下文只有 `Dockerfile` + `scripts/`（上游源码在镜像内拉取） |
