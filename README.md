# LocateAnything-vLLM

Accelerated inference for **LocateAnything-3B** by decoupling the vision encoder from the LLM and offloading autoregressive generation to a high-throughput **vLLM** server.

[English](#english) | [中文](#chinese)

---

<a id="english"></a>

## English

### Overview

This project implements an optimized inference pipeline for **LocateAnything-3B**. It requires **two running services** to work:

1. **vLLM Server** — hosts the decoupled Qwen2 text model (`locate_qwen2_model`) and performs the heavy autoregressive generation.
2. **This Client** — loads the vision encoder (`MoonViT`) + `MLP1` and the standalone text embedding layer, stitches image/text embeddings, and sends them to the vLLM server via the OpenAI-compatible `prompt_embeds` API.

```
┌─────────────────────┐      prompt_embeds (Base64)      ┌─────────────────────┐
│   This Client /     │ ───────────────────────────────▶ │   vLLM Server       │
│   FastAPI (app.py)  │      OpenAI Completions API      │   locate_qwen2_model│
│   MoonViT + MLP1    │ ◀─────────────────────────────── │   Qwen2 AR generation
└─────────────────────┐        generated text            └─────────────────────┘
```

Pipeline inside the client:

1. Load the full LocateAnything-3B model locally.
2. Strip away the `language_model`, keeping only the vision encoder (`MoonViT`), `MLP1`, and the projection head.
3. Use a standalone `nn.Embedding` layer (`qwen2_embed_tokens.safetensors`) for text token embedding lookup.
4. Stitch vision and text embeddings together and inject them directly into vLLM.

This avoids slow Hugging Face `generate()` loops and manual KV-cache management, while leveraging vLLM's PagedAttention / FlashAttention for fast autoregressive decoding.

### Repository Structure

| File | Description |
|------|-------------|
| `app.py` | FastAPI service that receives an image + categories and returns detected bounding boxes. |
| `locateanything_vllm.py` | Standalone benchmark / batch-inference script. |
| `pre.py` | Extract only the `embed_tokens` weights to `qwen2_embed_tokens.safetensors`. |
| `pre_wQwen2.py` | Extract `embed_tokens` weights **and** save the decoupled Qwen2 LM for vLLM serving. |
| `Dockerfile` | Docker image for the local ViT + FastAPI client side. The `qwen2_embed_tokens.safetensors` file is baked in at build time. |
| `run_vLLM_docker.sh` | One-liner to launch the vLLM server in a separate container. |

### Prerequisites

- Python >= 3.10
- Two CUDA-capable GPUs (or one GPU large enough to run both services sequentially, not recommended for production)
- Full `LocateAnything-3B` weights available locally or via Hugging Face

### Quick Start

You need **two terminals** running at the same time.

#### Terminal 1 — Extract models and start vLLM

```bash
# 1. Extract the text model and embedding weights
python pre_wQwen2.py
```

This produces:

- `./locate_qwen2_model/` — the decoupled Qwen2 LM for vLLM
- `./qwen2_embed_tokens.safetensors` — standalone text embedding weights

Then start the vLLM server:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./locate_qwen2_model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  --dtype half \
  --max-model-len 8192 \
  --attention-backend TRITON_ATTN \
  --enable-prompt-embeds
```

Or use the pre-built FP8 model and the provided Docker script:

```bash
bash run_vLLM_docker.sh
```

Wait until vLLM prints `Application startup complete`.

#### Terminal 2 — Run the FastAPI client

```bash
python app.py
```

Then send a request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@example.jpg" \
  -F "categories=person,bicycle"
```

Response:

```json
{
  "status": "success",
  "raw_output": "...",
  "parsed_json": {
    "boxes_count": 2,
    "boxes": [[x1, y1, x2, y2], ...]
  },
  "annotated_image": "<base64>",
  "metrics": {
    "vit_time_ms": 75.0,
    "vllm_time_ms": 250.0,
    "total_time_ms": 330.0
  }
}
```

#### Batch benchmark

```bash
python locateanything_vllm.py
```

Place images under `./pic/` and adjust `MAX_CONCURRENT_WORKERS` to test different concurrency levels.

### Docker Build

The Dockerfile builds **only the client side**. You still need a running vLLM server reachable from the container.

```bash
# 1. Build the client image (embeds qwen2_embed_tokens.safetensors and app.py)
docker build -t locateanything-vllm-client .

# 2. Run the client container
#    Replace <vllm-host> with the actual host/IP of your vLLM server.
docker run -it --gpus=all \
  -p 8000:8000 \
  -e VLLM_BASE_URL=http://<vllm-host>:8000/v1 \
  locateanything-vllm-client
```

By default, `app.py` connects to `http://localhost:8000/v1`. If vLLM is running in another container or on another host, update `base_url` in `app.py` or pass it via environment variable.

### Connection Details

- The client uses the OpenAI Python SDK to talk to vLLM.
- Default vLLM endpoint: `http://localhost:8000/v1`
- The `model` name in the client request must match the model name used by `vllm serve`:
  - If you served `./locate_qwen2_model`, use `model="locate_qwen2_model"`.
  - If you served `shigureui/LocateAnything-Qwen2-FP8`, use `model="shigureui/LocateAnything-Qwen2-FP8"`.

### Key Details

- `prompt_embeds` are encoded as Base64-serialized PyTorch tensors and sent through `extra_body`.
- `skip_special_tokens=False` and `spaces_between_special_tokens=False` are required to preserve special coordinate tokens such as `<box>` and `<ref>`.
- The local ViT stage runs under an async lock to prevent GPU memory spikes from concurrent inferences.

---

<a id="chinese"></a>

## 中文

### 项目简介

本项目为 **LocateAnything-3B** 提供了一套高性能推理方案。它依赖**两个同时运行的服务**：

1. **vLLM 服务** — 承载剥离后的 Qwen2 文本模型（`locate_qwen2_model`），负责繁重的自回归文本生成。
2. **本客户端** — 加载视觉编码器（`MoonViT`）+ `MLP1` 与独立的文本 embedding 层，完成图文 embedding 拼接后，通过 OpenAI 兼容的 `prompt_embeds` 接口发送给 vLLM。

```
┌─────────────────────┐      prompt_embeds (Base64)      ┌─────────────────────┐
│   本客户端 /         │ ───────────────────────────────▶ │   vLLM 服务         │
│   FastAPI (app.py)  │      OpenAI Completions API      │   locate_qwen2_model│
│   MoonViT + MLP1    │ ◀─────────────────────────────── │   Qwen2 自回归生成   │
└─────────────────────┐        生成的文本                └─────────────────────┘
```

客户端内部流水线：

1. 本地加载完整 LocateAnything-3B 模型。
2. 剥离掉 `language_model`，仅保留视觉编码器（`MoonViT`）、`MLP1` 与投影层。
3. 使用独立的 `nn.Embedding` 层（权重来自 `qwen2_embed_tokens.safetensors`）完成文本 token 的查表嵌入。
4. 将图文 embedding 拼接后，通过 `prompt_embeds` 接口直接注入 vLLM 进行自回归生成。

这样可以绕过 Hugging Face 原生的 `generate()` 循环与繁琐的 KV-Cache 管理，把繁重的序列生成工作交给 vLLM 的 PagedAttention / FlashAttention 后端加速。

### 仓库结构

| 文件 | 说明 |
|------|------|
| `app.py` | FastAPI 服务：接收图片与类别列表，返回检测框。 |
| `locateanything_vllm.py` | 独立基准 / 批量推理脚本。 |
| `pre.py` | 仅导出 `embed_tokens` 权重为 `qwen2_embed_tokens.safetensors`。 |
| `pre_wQwen2.py` | 同时导出 `embed_tokens` 权重，并保存剥离后的 Qwen2 LM 供 vLLM 加载。 |
| `Dockerfile` | 本地 ViT + FastAPI 客户端镜像；构建时会自动下载并内置 `qwen2_embed_tokens.safetensors`。 |
| `run_vLLM_docker.sh` | 一键启动 vLLM 服务容器的命令。 |

### 前置条件

- Python >= 3.10
- 两块支持 CUDA 的 GPU（或一块足够大的 GPU 同时跑两个服务，生产环境不推荐）
- 本地或 Hugging Face 可访问的完整 `LocateAnything-3B` 权重

### 快速开始

需要**同时打开两个终端**运行。

#### 终端 1 — 剥离模型并启动 vLLM

```bash
# 1. 剥离文本模型与 Embedding 权重
python pre_wQwen2.py
```

会生成：

- `./locate_qwen2_model/` — 供 vLLM 加载的 Qwen2 文本模型
- `./qwen2_embed_tokens.safetensors` — 独立的文本 embedding 权重

然后启动 vLLM 服务：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./locate_qwen2_model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  --dtype half \
  --max-model-len 8192 \
  --attention-backend TRITON_ATTN \
  --enable-prompt-embeds
```

也可以使用社区已发布的 FP8 模型与提供的脚本：

```bash
bash run_vLLM_docker.sh
```

等待 vLLM 输出 `Application startup complete` 后再启动客户端。

#### 终端 2 — 运行 FastAPI 客户端

```bash
python app.py
```

发送请求：

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@example.jpg" \
  -F "categories=person,bicycle"
```

返回示例：

```json
{
  "status": "success",
  "raw_output": "...",
  "parsed_json": {
    "boxes_count": 2,
    "boxes": [[x1, y1, x2, y2], ...]
  },
  "annotated_image": "<base64>",
  "metrics": {
    "vit_time_ms": 75.0,
    "vllm_time_ms": 250.0,
    "total_time_ms": 330.0
  }
}
```

#### 批量压测

```bash
python locateanything_vllm.py
```

将图片放到 `./pic/` 目录，调整脚本中的 `MAX_CONCURRENT_WORKERS` 即可测试不同并发等级。

### Docker 构建

Dockerfile **只构建客户端**。你仍然需要额外启动一个可被该容器访问的 vLLM 服务。

```bash
# 1. 构建客户端镜像（会内置 qwen2_embed_tokens.safetensors 与 app.py）
docker build -t locateanything-vllm-client .

# 2. 运行客户端容器
#    把 <vllm-host> 替换为 vLLM 服务实际所在的主机/IP。
docker run -it --gpus=all \
  -p 8000:8000 \
  -e VLLM_BASE_URL=http://<vllm-host>:8000/v1 \
  locateanything-vllm-client
```

默认 `app.py` 连接 `http://localhost:8000/v1`。如果 vLLM 运行在另一个容器或另一台机器上，请修改 `app.py` 中的 `base_url`，或者通过环境变量传入。

### 连接说明

- 客户端通过 OpenAI Python SDK 与 vLLM 通信。
- vLLM 默认地址：`http://localhost:8000/v1`
- 客户端请求里的 `model` 名称必须与 `vllm serve` 时使用的模型名称一致：
  - 若服务启动的是 `./locate_qwen2_model`，客户端应使用 `model="locate_qwen2_model"`。
  - 若服务启动的是 `shigureui/LocateAnything-Qwen2-FP8`，客户端应使用 `model="shigureui/LocateAnything-Qwen2-FP8"`。

### 关键细节

- `prompt_embeds` 以 Base64 编码的 PyTorch 张量形式通过 `extra_body` 发送。
- 必须设置 `skip_special_tokens=False` 与 `spaces_between_special_tokens=False`，否则 `<box>`、`<ref>` 等坐标相关特殊 token 会被过滤，导致解析失败。
- 本地 ViT 阶段使用异步锁串行执行，防止并发推理时显存暴涨。

---

## License

This project is released under the terms of the repository's LICENSE file.
