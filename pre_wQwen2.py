import re
import torch
from PIL import Image, ImageDraw, ImageFont  # 导入绘制库
from transformers import AutoModel, AutoTokenizer, AutoProcessor

model_path = "LocateAnything-3B"
device  = "cuda"
dtype=torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device).eval()

import torch
from safetensors.torch import save_file

# 1. 获取对应的 config 变量（如果你的 model.config 包含这些的话）
# 否则也可以直接硬编码，从你的打印信息看：vocab_size=152681, hidden_size=2048
vocab_size = model.language_model.config.vocab_size
hidden_size = model.language_model.config.hidden_size

# 2. 提取 embed_tokens 层的权重，并转移到 CPU 上
embed_weight = model.language_model.model.embed_tokens.weight.detach().cpu()

# 3. 将其保存为 safetensors 格式
save_dict = {"weight": embed_weight}
save_file(save_dict, "qwen2_embed_tokens.safetensors")

print(f"Embedding 权限已保存，形状为: {embed_weight.shape}") 
# 预期输出: 形状为: torch.Size([152681, 2048])

import torch
import os

# 假设 `model` 是你的完整多模态模型 (例如 Qwen2-VL)
# tokenizer 是你对应的分词器

output_path = "./locate_qwen2_model"
os.makedirs(output_path, exist_ok=True)

# 1. 直接调用 language_model 的 save_pretrained
# 参数 safe_serialization=True 会强制将权重保存为 safetensors 格式
model.language_model.save_pretrained(
    output_path,
    safe_serialization=True 
)

# 2. 【极其重要】记得将分词器也保存到同一个目录下
# 因为 vLLM 加载时需要 tokenizer_config.json 和 vocab 数据
tokenizer.save_pretrained(output_path)

print(f"语言模型已成功剥离并保存至: {output_path}")
