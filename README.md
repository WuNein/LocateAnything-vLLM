# LocateAnything-vLLM: Accelerated Inference via Prompt Embeddings 🚀

[🇺🇸 English](#english) | [🇨🇳 中文](#chinese)

<a id="english"></a>

## English

### Introduction
This project provides an optimized inference pipeline for **LocateAnything-3B**. By decoupling the Vision Encoder (MoonViT) from the Large Language Model (Qwen2), we leverage **vLLM** to exponentially accelerate the Auto-Regressive (AR) text generation phase. 

Instead of dealing with complex manual KV-cache management and slow Hugging Face native `generate()` loops, this pipeline computes mixed multimodal embeddings (Text + Image) locally, and sends them via Base64 serialization to a high-throughput vLLM server using the OpenAI-compatible Completions API.

### Features
- **Decoupled Architecture**: Extracts the pure text backbone (`Qwen2ForCausalLM`) and standalone `embed_tokens` for lightweight local processing.
- **vLLM Acceleration**: Utilizes PagedAttention and FlashAttention backend in vLLM for the heavy lifting of sequence generation.
- **Prompt Embeddings API**: Seamlessly stitches Vision features and Text features, injecting them directly into the LLM's hidden space via `extra_body={"prompt_embeds": ...}`.
- **Lossless Bounding Box Decoding**: Bypasses vLLM's default tokenizer filtering (`skip_special_tokens=False`) to ensure exact coordinate tokens (e.g., `<box>`, `<ref>`) are returned perfectly.

### Quick Start

#### 1. Extract and Save the Text Model
Run the extraction script to isolate the LLM backbone and the embedding layer:
```python
# Saves the Qwen2 LM to ./locate_qwen2_model safely
model.language_model.save_pretrained("./locate_qwen2_model", safe_serialization=True)
tokenizer.save_pretrained("./locate_qwen2_model")

# Saves standalone text embedding weights
save_file({"weight": model.language_model.model.embed_tokens.weight.detach().cpu()}, "qwen2_embed_tokens.safetensors")
```

#### 2. Start the vLLM Server
Launch the vLLM server using the extracted standalone text model. **Make sure to include `--enable-prompt-embeds`.**
```bash
CUDA_VISIBLE_DEVICES=3 vllm serve locate_qwen2_model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  --dtype half \
  --max-model-len 8192 \
  --attention-backend TRITON_ATTN \
  --enable-prompt-embeds
```

#### 3. Run Inference Client
Execute the client script/Jupyter Notebook. The client will:
1. Process the image using `MoonViT` + `MLP1`.
2. Convert text `input_ids` to embeddings using the standalone `nn.Embedding`.
3. Stitch image embeddings into the `<image>` token slots.
4. Encode the final tensor to Base64 and send it to vLLM.

<details>
<summary><b>Click to expand: Post-processing & Visualization Code</b></summary>

```python
# Extract boxes from API Text Response
boxes = parse_boxes(completion.choices[0].text, image_width, image_height)

# Draw on PIL Image
visualized_img = draw_boxes(image, boxes, width=3)
visualized_img.show()
```
</details>

---
<br>

<a id="chinese"></a>

## 中文

### 项目简介
本项目为 **LocateAnything-3B** 提供了一套经过深度优化的推理流水线。通过将视觉编码器（MoonViT）与大语言模型（Qwen2）解耦，我们巧妙地利用了 **vLLM** 来指数级加速自回归（AR）生成阶段。

本方案抛弃了原本复杂的手动 KV-Cache 维护与低效的 Hugging Face 原生 `generate()` 函数。它在本地计算并融合多模态特征（文本 + 图片），随后将完整的 Embedding 张量经 Base64 序列化后，通过 OpenAI 兼容的 Completions API 发送给高吞吐量的 vLLM 服务端进行推理。

### 核心特性
- **架构解耦**：成功剥离纯文本底座（`Qwen2ForCausalLM`）与独立的文本映射层（`embed_tokens`），实现轻量化的前端本地处理。
- **vLLM 极致加速**：将序列生成的繁重计算交由 vLLM（基于 PagedAttention 与 FlashAttention）接管，大幅提升推理速度（TPS）。
- **Prompt Embeddings API**：无缝拼接视觉特征与文本特征，并通过原生 API 的 `extra_body={"prompt_embeds": ...}` 直接注入到大模型隐层空间。
- **无损坐标解析**：通过动态打通底层采样参数（`skip_special_tokens=False`），绕过 vLLM 默认的特殊字符过滤，确保精准找回 `<box>`、`<ref>` 等原版坐标 Token。

### 快速开始

#### 1. 剥离并保存文本模型与独立 Embedding
运行提取脚本，将多模态模型中的 LLM 底座隔离保存，并导出独立的文本特征层权重：
```python
# 将 Qwen2 语言模型安全保存至 ./locate_qwen2_model 目录
model.language_model.save_pretrained("./locate_qwen2_model", safe_serialization=True)
tokenizer.save_pretrained("./locate_qwen2_model")

# 单独保存文本 Embedding 权重
save_file({"weight": model.language_model.model.embed_tokens.weight.detach().cpu()}, "qwen2_embed_tokens.safetensors")
```

#### 2. 启动 vLLM 服务端
使用刚才剥离的纯文本目录启动 vLLM 服务。**注意：必须携带 `--enable-prompt-embeds` 参数以接收张量输入。**
```bash
CUDA_VISIBLE_DEVICES=3 vllm serve locate_qwen2_model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  --dtype half \
  --max-model-len 8192 \
  --attention-backend TRITON_ATTN \
  --enable-prompt-embeds
```

#### 3. 运行客户端推理
执行客户端脚本或 Jupyter Notebook，其内部工作流如下：
1. 本地调用 `MoonViT` + `MLP1` 处理视觉输入并投影。
2. 使用独立的 `nn.Embedding` 将文本 `input_ids` 查表转换为特征。
3. 将视觉特征填充至序列中的 `<image>` 占位槽。
4. 将合并后的最终特征进行 Base64 编码，并向 vLLM 发起网络请求。

<details>
<summary><b>点击展开：输出解析与可视化处理代码</b></summary>

```python
# 从 API 返回的纯文本中正则解析检测框
boxes = parse_boxes(completion.choices[0].text, image_width, image_height)

# 在 PIL 图像上绘制边框与标签
visualized_img = draw_boxes(image, boxes, width=3)
visualized_img.show()
```
</details>
