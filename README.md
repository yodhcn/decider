# decider-gguf

把 [Mapika/decider](https://github.com/Mapika/decider) 的 **llama.cpp / GGUF** 部署路径打成 Docker 镜像，
用 GitHub Actions 在 ubuntu runner 上构建、导出 `tar.gz` 并上传为 **workflow artifact**（不推任何 registry）。

目标宿主机：A40 / 驱动 550.90.12 / CUDA 12.4。

## 镜像里有什么

- 上游源码（`DECIDER_REF`）以 `decider-ai[gguf]` 安装
- `llama-cpp-python >=0.3.35,<0.4`：**从源码编译**，`CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=86"`
- Python 3.12 / Ubuntu 22.04 / `nvidia/cuda:12.4.1-runtime`

**不含**模型文件：`.gguf` 权重、tokenizer、`decider_config.json` 都由你挂载。

### 为什么是 `decider-ai[gguf]` 而不是 `llama-server`

上游 README 里写得很直接：

> The GGUF files are not chat models: loading one in `llama-cli` or Ollama gives a text model, not decisions.

> The HTTP server does not serve GGUF files

decider 的读出头不是「让模型接着往下写」，而是把 answer slot 位置的 logits 投影到选项字母行上，这一步只有
`decider/engine_gguf.py` 通过 llama-cpp-python 的低层 C API（`llama_decode` + `llama_get_logits_ith`）做。
所以「通过 llama.cpp 部署 decider」= 装 GGUF extra，用 `decider.infer.Decider`，起 `llama-server` 只会得到一个普通文本模型。

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

`.gguf` 和它的 tokenizer、`decider_config.json` 必须放在同一个目录（上游 `Decider` 就是这么解析的）：

```bash
huggingface-cli download Mapika/decider-4b-GGUF --local-dir /srv/decider-4b-GGUF
# /srv/decider-4b-GGUF/ 里应有 decider-4b-v2.1-Q4_K_M.gguf + tokenizer 文件 + decider_config.json
```

### 2. 跑上游自带的 demo（三组 routing + 一个 10 字段 typed schema）

`python -m decider.infer <本地 .gguf 路径>` 是上游 `decider/infer.py` 的 `__main__`，拿本地 `.gguf` 直接可跑：

```bash
docker run --rm --gpus all -v /srv/decider-4b-GGUF:/models decider-gguf:cu12.4.1 \
  python -m decider.infer /models/decider-4b-v2.1-Q4_K_M.gguf
```

### 3. 跑自己的逻辑

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

### 4. 不落盘模型：仓库 ID + 缓存卷

```bash
docker run --rm --gpus all -v decider-hf:/hf-cache decider-gguf:cu12.4.1 python -c "
from decider.infer import Decider
d = Decider('Mapika/decider-4b-GGUF', gguf_file='decider-4b-v2.1-Q4_K_M.gguf')
print(d.decide('Hello', [{'question': 'Language?', 'options': ['en', 'zh']}]))"
```

首次联网写入 `/hf-cache`（= 镜像里的 `HF_HOME`），之后可加 `-e HF_HUB_OFFLINE=1` 离线跑。
`gguf_options` 可传 `n_ctx`、`n_gpu_layers`（-1 全量 offload，本镜像的 CUDA 构建默认走这条）、`n_threads`。

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

1. 这条路径上**所有 CUDA 计算都在 llama.cpp 里**。`decider/engine_gguf.py` 只拿 torch 做 `from_numpy` / `no_grad` / `softmax` / `masked_fill` / `split` 这类小 CPU 张量操作。
2. 走查过代码，这条 import 链上没有无条件 CUDA 调用：`decider/infer.py:109` 只是 `torch.cuda.is_available()` 查询（GGUF 分支随后把 device 覆盖成 `"gguf"`），`prompt.py` / `temperature.py` / `systemone.py` / `model.py` 里没有 `torch.cuda`。
3. 上游自己的 CI 就是这么测的：`.github/workflows/tests.yml` 的 `torch-cpu` job 用 `pip install torch --index-url https://download.pytorch.org/whl/cpu` 跑完整套测试。
4. 省掉 torch 自带的 `nvidia-*-cu12` 用户态库（约 2 GB），更重要的是避免它在同一个进程里和基镜像的 CUDA 12.4 争抢同一批 SONAME（`libcudart.so.12` / `libcublas.so.12`）。
5. 不会被依赖顶掉：`flash-linear-attention` 的 bare 依赖只有 `fla-core` + `einops` + `transformers`，`fla-core` 的 `torch>=2.7` 只挂在 `[cuda]` / `[cpu]` extra 上，所以先装 torch 不会被重新解析掉。

**代价**：`decider.serve`（HTTP 服务）和 torch 引擎需要 CUDA torch，这个镜像跑不了；而且即便换成 CUDA torch，HTTP 服务也不服务 GGUF（上游原话）。要 CUDA torch 版：

```bash
docker build -t decider-gguf:cu12.4.1-cudatorch --build-arg TORCH_INDEX=https://pypi.org/simple .
```

此时「镜像里只有一套 CUDA」就不再成立：PyPI 的 torch 轮子自带一套更新的 `nvidia-*-cu12`，和基镜像的 12.4 在同一个进程里共存（12.x 主版本内 ABI 兼容，且两者 SONAME 相同会被动态链接器去重）。

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

- **本机没有 docker**，交付时只跑过静态校验（Dockerfile 指令解析 + workflow 结构）。下面这条断言**依赖第一次 CI 构建**才算验过：
  ```python
  # Dockerfile 里的构建期自检
  assert glob.glob(os.path.join(llama_cpp_lib_dir, "*cuda*"))
  ```
  llama.cpp 缺 CUDA 后端时不会报错、只会静默走 CPU，所以这里专门断言 `*cuda*` 落在 `llama_cpp/lib/`。失败会直接停在这行并打印该目录的实际内容。
- 构建时间：llama.cpp 的 CUDA 编译 + torch 下载，2 核 runner 上大致 30–60 分钟（`timeout-minutes: 120`，并先跑 `free-disk-space`，因为 -devel 层 + venv + tar 大约要 15 GB）。
- artifact 单件上限 **10 GB（硬限制）**，所以镜像是先 `docker save | pigz` 再上传，`compression-level: 0` 避免二次压缩。
- 镜像以 root 运行；`/dev/nvidia*` 由 nvidia-container-toolkit 在容器启动前注入，不需要额外 `--privileged`。
- 没有 GPU 的 runner 上跑不了真实推理，所以 CI 只做 import / CUDA 后端 / 版本探测（探测失败不阻断上传）。真实 GPU 冒烟请在 A40 宿主机上跑第 2 节那条 `python -m decider.infer`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | 镜像定义（devel 构建 → runtime 交付，`ENTRYPOINT []` + `CMD tail -f /dev/null`） |
| `.github/workflows/docker-build-artifact.yml` | 手动触发，构建 → `docker save \| gzip` → 上传 artifact |
| `.dockerignore` | 构建上下文只有 `Dockerfile`（上游源码在镜像内拉取） |
