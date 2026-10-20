# decider-torch

把 [Mapika/decider](https://github.com/Mapika/decider) 的 **torch 引擎**部署路径打成 Docker 镜像。
decider 本身是 **PyPI 上的 `decider-ai` 发布轮子**（版本由 `DECIDER_VERSION` 钉死），基础镜像是
**`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`**——torch 2.7.1 / CUDA 12.6 / cuDNN 9 / triton 3.3.1 与
Python 都来自它，本镜像**自己不再装 torch**。**镜像不含权重**。

镜像里跑的是**上游自己的服务器** `decider.serve:app`，不是本仓库重写的实现：

| 路由 | |
|---|---|
| `POST /decide` | 纯形式（context + schema） |
| `POST /v1/systemone` | TypeSafe / Jev 线格式，`TYPESAFE_BASE_URL=http://localhost:8000` 可直接接 |
| `GET /health`、`/v1/models`、`/stats` | 健康、模型信息、计数与限额 |

## 镜像里有什么

| | |
|---|---|
| 有 | **`decider-ai` 1.9.0**（PyPI 发布轮子，`pip install "decider-ai[serve]==1.9.0"`；`serve` extra 带 `fastapi` / `uvicorn[standard]` / `httpx`）。**不是从源码装**，见下节 |
| 有 | **torch 2.7.1 + CUDA 12.6 + cuDNN 9**：来自基础镜像 `pytorch/pytorch`，**不是 pip 装的**，含整套 CUDA 用户态（cudart / cuBLAS / cuDNN / NCCL / nvrtc） |
| 有 | `triton 3.3.1`（基础镜像） + `flash-linear-attention`（pip）—— Qwen3.5 混合层的 Gated DeltaNet 靠它 |
| 有 | `causal_conv1d 1.7.0`（预编译轮子，按 torch/cpython/ABI 匹配）—— 少了它 transformers 会退回参考实现 |
| 有 | Python ≥3.11，来自基础镜像的 conda base 环境（确切版本由镜像决定，构建日志的 `python=` 行会打出来）。上游 `requires-python>=3.11`，`check_image.py` 断言而不是假设 |
| 有 | `scripts/serve.sh`（启动入口）、`scripts/check_image.py`（镜像契约自检） |
| 没有 | 权重 / tokenizer / `decider_config.json`：由调用方挂载或从 Hub 拉 |
| 没有 | llama.cpp 与 GGUF 支持（`decider.serve` 读不了 GGUF） |
| 没有 | nvcc / 编译器工具链（用 `-runtime` 而不是 `-devel`）、任何需要编译的步骤 |

## 构建

```bash
docker build -t decider-torch:cu126 .
```

| build arg | 默认 | 说明 |
|---|---|---|
| `PYTORCH_IMAGE` | `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` | 基础镜像，**tag 里同时钉住 torch / CUDA 次版本 / cuDNN**。换它就等于换这三样：`2.8.0-cuda12.8-cudnn9-runtime` 会给 triton 3.4.0 + ptxas 12.8 |
| `DECIDER_VERSION` | `1.9.0` | decider-ai 的**确切发布版本**（PyPI）。写成 `==`，不接受范围——镜像要能只凭这个值复现，`check_image.py` 拿装上的 dist 版本与它逐字比对 |
| `CCV_VERSION` / `CCV_TAG` | `1.7.0` / `v1.7.0` | causal_conv1d 的版本与 GitHub release tag |

构建 = 拉基础镜像 + **一次 `apt-get install`**（三个 InfiniBand 运行时库，见下）+ **两次 pip 安装**（`decider-ai` 轮子 + `causal_conv1d` 轮子），没有编译器，整个 Dockerfile 只有 4 个 `RUN`。对比上一版 GGUF 方案：那一版光 `llama-cpp-python` 源码编译就 703 秒，且必须两阶段（devel 编译 → runtime 运行）。

### 为什么 runtime 基础镜像还要装三个库

`pytorch/pytorch:...-runtime` 装好的 `libnccl.so.2` 把 `libibverbs.so.1`、`librdmacm.so.1`、`libmlx5.so.1` 声明成了 **DT_NEEDED**——NVIDIA 编译 NCCL 时把 InfiniBand 传输**链进去**了，不是运行时 `dlopen`。这三个库是 `libtorch_cuda.so` 的传递依赖，缺了它**加载不了**。

麻烦在于 torch 把这个加载推迟到第一次 CUDA 调用，所以症状不是 `ImportError`，而是 `torch.cuda.is_available()` 返回 `False`、引擎安静地跑在 CPU 上——正是这个镜像整套设计要避免的静默降级。单张 A40 永远走不到 IB 传输路径，但动态链接器不知道这件事。

```dockerfile
apt-get install -y --no-install-recommends libibverbs1 librdmacm1 ibverbs-providers
```

约 3 MB 运行时库，无编译器无头文件。`check_image.py` 的 ldd 扫描会带上这三个的名字，并且失败信息里直接写该装哪个包。

## 为什么装 PyPI 上的 `decider-ai`，而不是从源码装

上游的发布轮子是 `py3-none-any`（纯 Python），`decider/serve.py`、`decider/engine_v2.py`、`decider/batching.py` 全在里面——仓库比轮子多出来的只有 `scripts/` 和 `docs/`，而启动脚本本仓库自己就有一份。所以「`curl` 取 tarball 再 `pip install /opt/decider`」这条路买到的是零，代价却有几样：一个可能 404 的 URL、一个会移动的 `main`、一份需要与轮子对齐的源码副本，以及镜像是哪一版 decider **说不出来**。

改成 `pip install "decider-ai[serve]==${DECIDER_VERSION}"` 之后：

- 镜像的 decider 版本由**一个字符串**决定，可复现，也写进 `org.opencontainers.image.version` 与 CI 产物名；
- `check_image.py` 拿 `importlib.metadata.version("decider-ai")` 与 `DECIDER_VERSION` **逐字比对**——不是「pip 说装好了」，而是「装上的就是声明的那一版」；
- 镜像里不再需要 `curl` 和 `git`（两者都删了），基础层少一步网络。

代价说清楚：`main` 上的新东西要等上游发版才有。要跑未发布的分支，把 `DECIDER_VERSION` 换成一个自己 build 的 wheel 或改回源码安装——但那样产物名就不能只靠版本号标识了。

## 为什么基础镜像是 pytorch/pytorch

上一版是 `python:3.12-slim` + `pip install --index-url .../whl/cu126 torch`，让 torch 轮子自带 CUDA 用户态。这一版反过来：**torch、CUDA 12.6、cuDNN 9 三者由 `pytorch/pytorch` 提供**，镜像自己一行 `pip install torch` 都不写。

理由是它们本来就是一个整体。cuDNN 的算子选择、cuBLAS 的 kernel、torch 编译时用的 CUDA 版本是绑在一起测的；pip 能做的只是按依赖求解器挑一组 `nvidia-*-cu12` 轮子摞在 torch 下面，那个组合没人验证过。基础镜像里这套是发布方编译并测过的，这也是「不要自己从 ubuntu 装驱动/CUDA/torch」这条通行建议的字面意思。

三个落地细节：

- **不装 torch，并当场检查它没被换掉**。`decider-ai` 声明的是裸 `torch`，pip 会认为已装即满足；Dockerfile 在 `pip install "decider-ai[serve]==$DECIDER_VERSION"` 前后各读一次 `torch.__version__`，不一致就直接失败——否则「torch 来自基础镜像」只是期望，而不是事实。
- **`-runtime` 而不是 `-devel`**：这里不编译任何 CUDA 扩展。`causal_conv1d` 是预编译轮子，fla 的 kernel 由 triton 在运行时 JIT，而 triton 自带 ptxas。devel 只是白白多背一套 nvcc。
- **基础镜像先报一遍身份**：`torch <版本> cuda <版本> python <版本>`，且不是 CUDA 构建就在那一步停住，而不是等两 GB 的 pip 装完再由 `check_image.py` 报告。

代价说清楚：镜像比 slim + 轮子那版**大几 GB**（基础镜像含完整 CUDA 运行时和 conda 环境），Python 也不再由本仓库挑——它来自镜像的 conda base，上游只要求 `requires-python>=3.11`，够用；但 Python 版本直接决定 `causal_conv1d` 的轮子名（`cp311`/`cp312`），所以 Dockerfile 现场推导，`check_image.py` 再断言它 ≥3.11。

顺带，这套结构正好对应「AI 镜像三条常识」的另外两条：依赖用包管理器声明式装（这里是 `pip install "decider-ai[serve]==$DECIDER_VERSION"`，版本由一个 pin 定死），权重一律走卷、不进镜像（`/hf-cache`，以及 `/triton-cache`）。

## 版本为什么这么选（驱动 550.90.12 / A40）

宿主机驱动 **550.90.12** 是 CUDA **12.4** 那个 toolkit 的驱动，但它**不是**镜像只能用 cu124 的理由：

> CUDA 12.x 是 minor-version compatible —— 用任意 12.x toolkit 编译的应用，都能跑在 **≥525** 的驱动上（上限到 580）。
> 所以 550 上跑 cu126 是**官方允许**的；CUDA 13 才需要 ≥580。

真正卡住的是 **triton**，**另外 `causal_conv1d` 的预编译轮子卡住上限**。下表如实测的 wheel 元数据（`torch` 的 `requires_dist` 里钉的 `triton==` 与 `nvidia-*-cu12`）：

| torch | triton | cudart / cuDNN | 非 Blackwell 的 ptxas | `causal_conv1d 1.7.0` 轮子 | 与 fla 门槛 |
|---|---|---|---|---|---|
| 2.6.0（cu124 天花板） | 3.2.0 | 12.4 / — | 12.4.99 | `cu12torch2.6` | ❌ 低于 **3.3** |
| **2.7.1（本镜像，`...-cuda12.6-cudnn9-runtime`）** | **3.3.1** | **12.6.77 / 9.5.1** | **12.4.99** | **`cu12torch2.7` ✅** | ✅ 刚好过线 |
| 2.8.0（`...-cuda12.8-cudnn9-runtime`） | 3.4.0 | 12.8.90 / 9.10.2 | 12.8.93 | `cu12torch2.8` ✅ | ✅ 有余量，但 sm_86 的 codegen 变了 |
| 2.9.x / 2.10.0（cu128） | 3.5.x / 3.6.0 | 12.8.90 / 9.10.2 | 12.8.93 | ❓ 未见对应轮子 | ✅ 但落地先卡在卷积轮子上 |
| 2.14.x | 3.8 | **cu13** | — | ❌ | ❌ 需要驱动 ≥580，550 上跑不了 |

选 2.7.1 的关键理由在 ptxas 那一列：**triton 3.3 给 sm_86 用的 ptxas 是 12.4.99，和 3.2.0 完全相同**（3.3 的 `cmake/nvidia-toolchain-version.json` 里 `ptxas: 12.4.99`，12.8 的那个只给 Blackwell）。也就是说编译出来的 kernel 一位没变，**只是满足了 fla 的版本门槛**——数值上不引入新变量。2.8.0 能用（也能跑在 550 上，CUDA 12.x 是 minor-version compatible），但没有收益，代价是把 sm_86 的 codegen 挪到 ptxas 12.8。

换成 `pip install decider-ai` 之后，这个判断**没有被推翻，只是变得更重要**：`decider-ai` 声明的 torch 是裸的，而 fla 的两条门槛（`torch>=2.7`、`triton>=3.3`）写在 **`fla-core[cuda]` 这个 extra 里**——我们装的是裸 `fla`，所以 **pip 根本不会校验它们**。torch 的唯一来源就是基础镜像，所以这两条由 `check_image.py` 断言（第 6、7 项），而不是指望依赖求解器。也正因为 2.7.1 刚好压在两条地板上，将来 fla 抬高任意一条，构建会**当场失败**并提示下一个可用 tag，而不是上机后静默降级。

为什么这条门槛必须过？因为 fla 在版本不足时**只 warning、不报错**，而紧随其后的平台探测会把失败也吞掉：

```python
# fla/utils/_device.py
required_triton_version = package_version.parse("3.3.0")   # 低于它只 logger.warning
@cache
def get_available_device():
    try:    return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        _cpu_device_warning(); return 'cpu'                # 且被 @cache 永久记住
```

一旦落到 `'cpu'`，transformers 就走 `modeling_qwen3_5.py` 的参考实现
`(query @ key.transpose(-1,-2)) * pairwise_decay` —— **O(N²) 物化，既慢又爆显存**。4B + A40 就是这么 OOM 的。

`causal_conv1d` 是同一类静默降级：没装 → transformers 每个混合层都用参考卷积。轮子必须与 torch 次版本、cpython tag、C++ ABI 三者对齐，所以 Dockerfile **按实际装上的 torch 反推文件名**，再兜底试另一个 ABI 拼写。

## 运行

模型在 Hub 上：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v decider-hf:/hf-cache -v decider-triton:/triton-cache \
  decider-torch:cu126 scripts/serve.sh Mapika/decider-2b 8000
```

模型已经在宿主机目录里（不下载、不用卷）：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /srv/decider-2b:/models/decider-2b:ro \
  decider-torch:cu126 scripts/serve.sh /models/decider-2b 8000
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

两个路径名都是**这个镜像起的**，不是库的约定——`Dockerfile` 的 `ENV` 块把它们指过去，卷挂的正是这两个路径。之所以必须显式指：库的默认位置在容器的可写层里，容器一删缓存就没了。

⚠️ 一个坑：`TRITON_CACHE_DIR` **设成空字符串不等于「不缓存」**，且行为随版本变：
- triton ≤3.3（含本镜像的 3.3.1；`knobs` 模块从 3.4 才有）是 `os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()` → **静默回落到 `/root/.triton/cache`**，缓存照样写，只是写到了卷外；
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
| `DECIDER_WARMUP` | `1` | `0` 跳过图捕获（**没有** `DECIDER_GRAPHS` 这个变量） |
| `DECIDER_B_BUCKETS` / `DECIDER_T_BUCKETS` | `1,2,4,8,16,32` / 64…8192 共 17 个 | 逗号分隔。图数 = 满足 `B==1 or B*T<=budget` 的笛卡尔积，默认 89 张 |
| `DECIDER_GRAPH_TOKEN_BUDGET` | `32768` | 单张图的行词元上限，和上一行一起决定网格大小 |
| `DECIDER_MAX_BATCH` | `32` | 一次前向的行数上限 |
| `DECIDER_BATCH_WAIT_MS` / `DECIDER_BATCH_ADAPTIVE_WAIT_MS` | `0` / `2` | 攒批窗口（后者只在上一批含多个请求时才生效） |
| `DECIDER_SHARED` / `DECIDER_SHARED_MIN_TOKENS` | `1` / `768` | 多问题 + 长 state 的共享前缀路径 |
| `DECIDER_MAX_QUEUE_ROWS` | `4096` | 排队行数上限，超出 503 —— **单位是行，不是请求** |
| `DECIDER_FP8` | `0` | **A40 上保持 0**：FP8 张量核心从 Ada（sm_89）/ Hopper 起，Ampere 没有 |
| `DECIDER_COMPILE` | `0` | **本镜像不支持设成 1**：torch.compile 的 inductor 后端需要主机编译器 + ninja，镜像里没装。要用就自己加 `build-essential`/`ninja` 重建 |

## A40 / 驱动约束

- 驱动 **550.90.12** = CUDA **12.4** 的 toolkit 驱动，但 CUDA 12.x minor-version compatible（≥525），所以镜像用 **cu126 是合法组合**；真正决定版本的是 triton 门槛，见上一节。
- `--gpus all` 是必须的：镜像里**没有** `libcuda.so.1`，它由宿主驱动经 NVIDIA container runtime 在容器启动时注入；这也是构建期不能真跑前向的原因。
- A40 无 NVLink，多卡张量并行只能走 PCIe。**`DECIDER_FP8` 必须保持 `0`**：FP8 张量核心从 Ada（sm_89）起，Ampere 没有。
- 显存：2B bf16 约 4 GB、4B bf16 约 8.4 GB。

### 默认 89 张 CUDA graph 在 44 GiB 上装不下（4B 实测）

`engine_v2.py` 的网格是 `[(B,T) for T in T_BUCKETS for B in B_BUCKETS if B==1 or B*T<=token_budget]`：
17 个 `T`（64→8192）× 6 个 `B`（1,2,4,8,16,32），budget 32768 → **正好 89 张**。每张按 `B*T` 占激活：

| 配置 | 图数 | ΣB·T | 实测/外推私有池 |
|---|---|---|---|
| 默认 | 89 | 628,800 | ≈70 GiB ❌（实测到 50 张、Σ=159,552 时已 17.88 GiB，≈0.115 MiB/token） |
| `DECIDER_B_BUCKETS=1,2,4` | 51 | 214,592 | ≈23.5 GiB ✅ |
| `DECIDER_B_BUCKETS=1,2,4,8` + `DECIDER_GRAPH_TOKEN_BUDGET=16384` | 62 | 230,464 | ≈25.8 GiB ✅ |

保住 `B=1` 的全部 `T`（长行只能靠它），砍掉大 batch 那半边。完全没有 `DECIDER_GRAPHS` 这个环境变量——关图只有 `DECIDER_WARMUP=0`（`use_graphs` 只看 device 是否为 cuda）；关掉后 `logits_all` 走 eager，`max_rows` 回落到 `token_budget // T`，慢但不崩。

## 入口脚本是本仓库的

发布轮子里没有 `scripts/`，所以镜像的入口是 `scripts/serve.sh`（装在 `/app/scripts/`，`WORKDIR` 是 `/app`）。它干的事和上游仓库里那个脚本一样（`exec uvicorn decider.serve:app`），差别有三处：

- **默认 `DECIDER_HOST=0.0.0.0`**：上游默认 `127.0.0.1` 在容器里穿不过 published port，表现像端口映射挂了；`UVICORN` 也预设成 `uvicorn`，因为上游默认的 `.venv312/bin/uvicorn` 只在它自己的 `uv venv` 开发环境里存在。
- 加一层 `decider_config.json` 预检。
- 挡掉 `.gguf` 参数（torch 引擎读不了，见下）。

预检解决的是一个真实的静默失败：`decider.serve` 读 `decider_config.json`（layout、temperature、`temperature_by_type`、`schema_first`）时用的是 `try/except` 全吞，仓库里没有这个文件就按 **temperature 1.0 + plain layout** 跑，而且一个字都不说。对 decider 模型这是错的；对「stock checkpoint + `DECIDER_LAYOUT=chat`」则是正常的。所以它只大声告警、不阻断启动。想关掉：`DECIDER_CONFIG_CHECK=0`。

## CI：构建并上传为 artifact

`.github/workflows/docker-build-artifact.yml`，只 `workflow_dispatch`，输入 `decider_version`（PyPI 上的确切发布版本）与 `pytorch_image`（基础镜像，含 tag）。产物是 `decider-torch-<base tag>-<版本>-amd64.tar.gz` + `.sha256` + `.manifest.txt`，**不推任何 registry**。默认组合下就是 `decider-torch-2.7.1-cuda12.6-cudnn9-runtime-1.9.0-amd64`。

```bash
sha256sum -c decider-torch-*.tar.gz.sha256
gzip -dc decider-torch-*.tar.gz | docker load
```

镜像契约由 `scripts/check_image.py` 定义，**构建期和 CI smoke test 跑的是同一个文件**，两边不会漂。14 项检查（全部不需要 GPU）：

1. **Python ≥3.11** —— 版本现在由基础镜像决定（不是本仓库挑的），而 `decider-ai` 的 `requires-python` 和 `causal_conv1d` 的轮子名（`cp311`/`cp312`）都取决于它
2. torch 是 **CUDA 12.x** 构建，且轮子 tag 与 `torch.version.cuda` 一致 —— 装成 CPU 轮子会让整个引擎静默跑在 CPU 上；装成 cu13x 则超出驱动能力
3. torch 的 CUDA 动态依赖在镜像内全部可解析，只允许宿主驱动那几个缺失（按精确名白名单，`libcuda*` 前缀会把 `libcudart.so.12` 一起放行）。`ldd` 的搜索路径是**在镜像自己的 `LD_LIBRARY_PATH` 前面追加** torch 的库目录，不是替换——基础镜像自带系统 CUDA 12.6，替换会把它变成「检查自己造出来的缺库」。这一项是构建期真正拦下过东西的那一项：NCCL 的三个 InfiniBand 库缺失在这里失败，而不是在 A40 上表现为 `torch.cuda.is_available() == False`。顺带它也是 `libgomp1` 的兜底——那个没装的话这里会报 `libgomp.so.1: not found`
4. `decider.serve` 可导入且五条路由齐全
5. **装上的 `decider-ai` 就是 `DECIDER_VERSION` 声明的那一版** —— 逐字比对 dist 元数据，防的是「pip 装了个别的版本」和「镜像说不出自己跑的是哪版」
6. `triton` + `fla` 在位
7. **`triton` 版本 ≥ fla 自己声明的门槛** —— 门槛从 `fla/utils/_device.py` 的**源码文本**里正则读（`required_triton_version = package_version.parse("3.3.0")`），读不到才回落到文档里的 `3.3.0`。这里刻意**不 import**：那个模块在 import 期就调 `triton.runtime.driver.active.get_current_target()`，构建机上没有 GPU，探测本身就会失败——连 `find_spec("fla.utils._device")` 都不行，因为查找子模块会先导入父包。第一版用 `inspect.getsource` 就栽在这，构建日志里只留下 `fla floor unreadable`（检查降级了，但降级得很安静）。这一项本身是拿真实 GPU 失败换来的：`find_spec("fla")` 全绿、镜像上机第一次前向就退回 O(N²) 参考实现的正是它
8. **`torch` 版本 ≥ fla 自己声明的门槛** —— 从 `fla-core` 的 dist 元数据（`torch>=2.7`）里读。这条门槛写在 `fla-core[cuda]` 这个 extra 里，而我们装的是裸 `fla`，**pip 永远不会校验它**；torch 又只来自基础镜像，所以只能在这里断言
9. **`causal_conv1d` 真的能 import**，不是只装上了（同一个坑：装了 ≠ 能用）
10. `transformers>=5`，并把 `transformers` / `flash-linear-attention` 的实际版本打出来（这两个是 `decider-ai` 下面**没有 pin** 的依赖，重建一次就可能变）
11. `scripts/serve.sh` 存在、可执行、bash 语法干净、确实启动 `decider.serve:app`
12. `/hf-cache` 与 `/triton-cache` 可写
13. 两个缓存**确实落在上面那两个路径** —— 问 `FileCacheManager` 自己刚建出来的目录在哪、问 `huggingface_hub.constants.HF_HUB_CACHE` 在哪，而不是回读环境变量（回读是循环论证：库不认这个变量也照样通过）。⚠️ 不能拿 triton 的 `default_cache_dir()` 当答案——3.2/3.3 里它**只是 fallback、恒返回 `~/.triton/cache`**，跟环境变量无关，第一版检查就栽在这（构建期假红）
14. 没有把权重打进镜像

单跑：

```bash
docker run --rm decider-torch:cu126 python /app/scripts/check_image.py
```

## 安全

**没有鉴权。** `DECIDER_HOST` 默认 `0.0.0.0` 是为了让端口映射真的可用；只发布到可信网络，或者设 `DECIDER_HOST=127.0.0.1` 并改用 `docker exec` 访问。
