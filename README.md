# decider-torch

把 [Mapika/decider](https://github.com/Mapika/decider) 的 **torch 引擎**部署路径打成 Docker 镜像：
CUDA 12.4.1 / cu124，Python 3.12，**镜像不含权重**。

镜像里跑的是**上游自己的服务器** `decider.serve:app`，不是本仓库重写的实现：

| 路由 | |
|---|---|
| `POST /decide` | 纯形式（context + schema） |
| `POST /v1/systemone` | TypeSafe / Jev 线格式，`TYPESAFE_BASE_URL=http://localhost:8000` 可直接接 |
| `GET /health`、`/v1/models`、`/stats` | 健康、模型信息、计数与限额 |

## 镜像里有什么

| | |
|---|---|
| 有 | 上游源码 @ `DECIDER_REF`，按 `decider-ai[serve]` 安装（`fastapi` / `uvicorn[standard]` / `httpx`） |
| 有 | Python 3.12 + **torch cu124**，含它自带的整套 CUDA 用户态（cudart / cuBLAS / cuDNN / NCCL / nvrtc） |
| 有 | `triton` + `flash-linear-attention` —— Qwen3.5 混合层的 Gated DeltaNet 靠它 |
| 有 | `scripts/serve.sh`（启动入口）、`scripts/check_image.py`（镜像契约自检） |
| 没有 | 权重 / tokenizer / `decider_config.json`：由调用方挂载或从 Hub 拉 |
| 没有 | llama.cpp 与 GGUF 支持（`decider.serve` 读不了 GGUF） |
| 没有 | CUDA 基础镜像、nvcc、任何需要编译的步骤 |

## 构建

```bash
docker build -t decider-torch:cu124 .
```

| build arg | 默认 | 说明 |
|---|---|---|
| `DECIDER_REF` | `main` | 上游分支 / tag / commit SHA。镜像内用 `curl` 取 tarball 解包，不装 git |
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cu124` | 目标驱动把 CUDA 上限压到 12.4，所以是 cu124 |
| `TORCH_SPEC` | `torch` | 想复现就 `torch==2.6.0` —— cu124 索引里 cp312 的天花板 |
| `PYTHON_VERSION` | `3.12` | 上游 `requires-python>=3.11`，CI 用 3.12 |
| `DECIDER_REPO` | `https://github.com/Mapika/decider` | fork 的话改这里 |

构建就是下载 + pip 安装，**没有任何编译**。对比上一版 GGUF 方案：那一版光 `llama-cpp-python` 源码编译就 703 秒，且必须两阶段（devel 编译 → runtime 运行）。

## 运行

模型在 Hub 上：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v decider-hf:/hf-cache -v decider-triton:/triton-cache \
  decider-torch:cu124 scripts/serve.sh Mapika/decider-2b 8000
```

模型已经在宿主机目录里（不下载、不用卷）：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /srv/decider-2b:/models/decider-2b:ro \
  decider-torch:cu124 scripts/serve.sh /models/decider-2b 8000
```

`scripts/serve.sh [model] [port]` 的第一个参数既可以是 Hub 仓库 ID，也可以是本地 checkpoint 目录；端口默认 8000。自检：

```bash
curl -s localhost:8000/health
curl -s localhost:8000/stats
curl -s localhost:8000/v1/models
```

### 两个卷为什么值

| 卷 | 环境变量 | 库自己的默认路径 | 不挂会怎样 |
|---|---|---|---|
| `/hf-cache` | `HF_HOME` | `~/.cache/huggingface`（root → `/root/.cache/huggingface`） | 用仓库 ID 时，每次新建容器都重新下 4 GB（2B bf16）/ 8.4 GB（4B）。用本地目录则完全用不到 |
| `/triton-cache` | `TRITON_CACHE_DIR` | `$TRITON_HOME/.triton/cache`（root → `/root/.triton/cache`） | flash-linear-attention 的 Triton kernel 是针对具体 GPU JIT 编译的，cubin 与机器绑定，**没法预先打进镜像**；不挂就每次重编 |

两个路径名都是**这个镜像起的**，不是库的约定——`Dockerfile:74` 的 `ENV` 把它们指过去，卷挂的正是这两个路径。之所以必须显式指：库的默认位置在容器的可写层里，容器一删缓存就没了。

⚠️ 一个坑：`TRITON_CACHE_DIR` **设成空字符串不等于「不缓存」**，且行为随版本变：
- triton 3.2（torch 2.6.0+cu124 钉的版本）里是 `os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()` → **静默回落到 `/root/.triton/cache`**，缓存照样写，只是写到了卷外；
- triton ≥3.4 的 `knobs.cache.dir` 拿到空串后 falsy → 每次编译 `RuntimeError: Could not create or locate cache dir`。

两种情况都别踩，不要写 `-e TRITON_CACHE_DIR=`。

## 启动不是瞬间的

两段一次性开销，都发生在端口开始监听之前：

1. **Triton 首次编译** —— Qwen3.5 混合层走 `flash-linear-attention` 的 Triton kernel，第一次前向要为这张卡编译一遍。挂了 `/triton-cache` 只付一次。
2. **CUDA graph 全网格捕获** —— `decider.serve` 在启动时把 `(batch bucket, length bucket)` 整张网格捕获完再封盘，之后任何请求都不再触发捕获或 `torch.compile`（`docs/SERVING.md` 第 2 节，89 张图）。`DECIDER_WARMUP=0` 可跳过，代价是每个请求都走 eager。

所以 healthcheck / 编排的 grace period 要给够。

## 旋钮

完整变量表在上游 `decider/serve.py` 的模块 docstring 与 `docs/SERVING.md`。这个部署里最该知道的：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DECIDER_MODEL` | 镜像里已预设 `Mapika/decider-2b` | 仓库 ID 或本地目录；`serve.sh` 的第一个参数会覆盖 |
| `DECIDER_DEVICE` | `auto` | `cuda` / `cuda:0` / `cpu`。**非 CUDA 时没有图、不做 warmup，全部 eager** |
| `DECIDER_WARMUP` | `1` | `0` 跳过图捕获 |
| `DECIDER_MAX_BATCH` | `32` | 一次前向的行数上限 |
| `DECIDER_BATCH_WAIT_MS` / `DECIDER_BATCH_ADAPTIVE_WAIT_MS` | `0` / `2` | 攒批窗口（后者只在上一批含多个请求时才生效） |
| `DECIDER_SHARED` / `DECIDER_SHARED_MIN_TOKENS` | `1` / `768` | 多问题 + 长 state 的共享前缀路径 |
| `DECIDER_MAX_QUEUE_ROWS` | `4096` | 排队行数上限，超出 503 —— **单位是行，不是请求** |
| `DECIDER_FP8` | `0` | **A40 上保持 0**：FP8 张量核心从 Ada（sm_89）/ Hopper 起，Ampere 没有 |
| `DECIDER_COMPILE` | `0` | **本镜像不支持设成 1**：torch.compile 的 inductor 后端需要主机编译器 + ninja，镜像里没装。要用就自己加 `build-essential`/`ninja` 重建 |

## A40 / 驱动约束

- 驱动 **550.90.12** → CUDA 上限 **12.4** → 只能用 cu124 索引，cp312 最高 torch **2.6.0**。
- `--gpus all` 是必须的：镜像里**没有** `libcuda.so.1`，它由宿主驱动经 NVIDIA container runtime 在容器启动时注入；这也是构建期不能真跑前向的原因。
- A40 无 NVLink，多卡张量并行只能走 PCIe。
- 显存：2B bf16 约 4 GB、4B bf16 约 8.4 GB；48 GB 的 A40 放 4B 余量充足。

## 上游自己的 serve.sh 也在镜像里

路径 `/opt/decider/scripts/serve.sh`。它默认 `DECIDER_HOST=127.0.0.1` 和 `UVICORN=.venv312/bin/uvicorn` —— 前者在容器里穿不过 published port（表现像端口映射挂了），后者只在上游自己的 `uv venv` 开发环境里存在。本镜像把这两个都预设了（`DECIDER_HOST=0.0.0.0`、`UVICORN=uvicorn`），所以它**一个字不改就能直接用**：

```bash
docker run --rm --gpus all -p 8000:8000 -v decider-hf:/hf-cache \
  decider-torch:cu124 /opt/decider/scripts/serve.sh Mapika/decider-2b 8000
```

本仓库的 `scripts/serve.sh` 与它的差别只有三处：自己写死 `0.0.0.0` 默认值（不依赖 ENV）、加一层 `decider_config.json` 预检、挡掉 `.gguf` 参数。

预检解决的是一个真实的静默失败：`decider.serve` 读 `decider_config.json`（layout、temperature、`temperature_by_type`、`schema_first`）时用的是 `try/except` 全吞，仓库里没有这个文件就按 **temperature 1.0 + plain layout** 跑，而且一个字都不说。对 decider 模型这是错的；对「stock checkpoint + `DECIDER_LAYOUT=chat`」则是正常的。所以它只大声告警、不阻断启动。想关掉：`DECIDER_CONFIG_CHECK=0`。

## CI：构建并上传为 artifact

`.github/workflows/docker-build-artifact.yml`，只 `workflow_dispatch`，输入 `upstream_ref` / `torch_index` / `torch_spec`。产物是 `decider-torch-<flavour>-<ref>-amd64.tar.gz` + `.sha256` + `.manifest.txt`，**不推任何 registry**。

```bash
sha256sum -c decider-torch-*.tar.gz.sha256
gzip -dc decider-torch-*.tar.gz | docker load
```

镜像契约由 `scripts/check_image.py` 定义，**构建期和 CI smoke test 跑的是同一个文件**，两边不会漂。9 项检查（全部不需要 GPU）：

1. torch 是 cu124 构建 —— 装成 CPU 轮子会让整个引擎静默跑在 CPU 上
2. torch 的 CUDA 动态依赖在镜像内全部可解析，只允许宿主驱动那几个缺失（按精确名白名单，`libcuda*` 前缀会把 `libcudart.so.12` 一起放行）
3. `decider.serve` 可导入且五条路由齐全
4. `triton` + `fla` 在位 —— 缺了会在 A40 第一次前向才炸
5. `transformers>=5`
6. `scripts/serve.sh` 存在、可执行、bash 语法干净、确实启动 `decider.serve:app`
7. `/hf-cache` 与 `/triton-cache` 可写
8. 两个缓存**确实落在上面那两个路径** —— 问 `FileCacheManager` 自己刚建出来的目录在哪、问 `huggingface_hub.constants.HF_HUB_CACHE` 在哪，而不是回读环境变量（回读是循环论证：库不认这个变量也照样通过）。⚠️ 不能拿 triton 的 `default_cache_dir()` 当答案——3.2 里它**只是 fallback、恒返回 `~/.triton/cache`**，跟环境变量无关，第一版检查就栽在这（构建期假红）
9. 没有把权重打进镜像

单跑：

```bash
docker run --rm decider-torch:cu124 python /app/scripts/check_image.py
```

## 安全

**没有鉴权。** `DECIDER_HOST` 默认 `0.0.0.0` 是为了让端口映射真的可用；只发布到可信网络，或者设 `DECIDER_HOST=127.0.0.1` 并改用 `docker exec` 访问。
