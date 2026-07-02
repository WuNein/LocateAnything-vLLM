import os
import io
import re
import gc
import time
import asyncio
import numpy as np
import torch
import torch.nn as nn
import pybase64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import json

app = FastAPI(title="Advanced Locate Server")

# ==========================================
# 核心模型加载类（单例模式）
# ==========================================
class ModelServer:
    def __init__(self, model_path="LocateAnything-3B", device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.max_size = 768
        
        print("====== 1. 初始化 Tokenizer & Processor ======")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        print("====== 2. 加载全量模型并拆卸 Language Model ======")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(self.device).eval()
        
        self.image_token_index = model.image_token_index
        self.extract_feature = model.extract_feature
        self.mlp1 = model.mlp1
        
        if hasattr(model, "language_model") and model.language_model is not None:
            model.language_model.to("cpu")
            del model.language_model
        del model
        
        gc.collect()
        torch.cuda.empty_cache()
        
        print("====== 3. 加载独立的纯净 Embedding 层 ======")
        vocab_size = 152681
        hidden_size = 2048
        self.standalone_embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=None)
        self.standalone_embed_tokens.load_state_dict(load_file("qwen2_embed_tokens.safetensors"))
        self.standalone_embed_tokens = self.standalone_embed_tokens.to(self.device)
        
        print("====== 4. 初始化 vLLM 客户端 ======")
        # base_url 可通过环境变量 VLLM_BASE_URL 配置，默认连本机 8000 端口
        self.vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.vllm_model = os.environ.get("VLLM_MODEL", "shigureui/LocateAnything-Qwen2-FP8")
        self.vllm_client = OpenAI(api_key="EMPTY", base_url=self.vllm_base_url)
        print(f"vLLM 服务端: {self.vllm_base_url}, 模型名: {self.vllm_model}")
        
        # 异步锁：确保本地 GPU 提取特征时单线程排队，防止显存爆炸
        self.gpu_lock = asyncio.Lock()
        print("模型与服务组件初始化成功！")

    def _tensor2base64(self, x: torch.Tensor) -> str:
        with io.BytesIO() as buf:
            torch.save(x, buf)
            buf.seek(0)
            binary_data = buf.read()
        return pybase64.b64encode(binary_data).decode("utf-8")

    def _parse_boxes(self, text: str, width: int, height: int):
        """
        解析 vLLM 返回的检测框文本。
        假设格式通常为: label [ymin, xmin, ymax, xmax] 或类似的归一化坐标(0-1000)
        这里写一个通用的正则提取器，请根据具体模型的输出来微调此函数。
        """
        # 匹配如 [123, 456, 789, 912] 这种形式
        pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches = re.findall(pattern, text)
        
        boxes = []
        for match in matches:
            # 多数 Locate 模型输出的是 0-1000 的归一化坐标
            ymin, xmin, ymax, xmax = [float(x) / 1000.0 for x in match]
            # 还原到真实像素尺寸
            box_pixel = [
                int(xmin * width),  # left
                int(ymin * height), # top
                int(xmax * width),  # right
                int(ymax * height)  # bottom
            ]
            boxes.append(box_pixel)
        return boxes

    def _draw_boxes_to_base64(self, image: Image.Image, boxes: list) -> str:
        """在图上画框，并转为 base64 字符串供前端/回传使用"""
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        for box in boxes:
            # box: [left, top, right, bottom]
            draw.rectangle(box, outline="red", width=3)
        
        buffered = io.BytesIO()
        draw_img.save(buffered, format="JPEG")
        return pybase64.b64encode(buffered.getvalue()).decode("utf-8")


# 初始化全局单例模型服务器
# 可以在服务启动时加载
server = None

@app.on_event("startup")
async def startup_event():
    global server
    server = ModelServer()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    categories: str = Form(..., description="分类列表，用逗号分隔，例如: person,bicycle")
):
    if server is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    start_total = time.perf_counter()
    category_list = [c.strip() for c in categories.split(",") if c.strip()]
    
    try:
        # 1. 异步读取上传文件并加载为 PIL Image
        contents = await file.read()
        orig_image = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_width, orig_height = orig_image.size
        
        # 缩放至模型要求的 max_size
        image = orig_image.copy()
        image.thumbnail((server.max_size, server.max_size), Image.Resampling.LANCZOS)
        
        # 2. 准备 inputs
        cats = "</c>".join(category_list)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        
        text = server.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = server.processor.process_vision_info(messages)
        
        # 3. GPU 推理阶段（加锁，单张处理，杜绝 Batch 冲突和显卡过载）
        async with server.gpu_lock:
            start_vit = time.perf_counter()
            
            inputs = server.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(server.device)
            pixel_values = inputs["pixel_values"].to(server.dtype)
            input_ids = inputs["input_ids"]
            image_grid_hws = inputs.get("image_grid_hws", None)
            
            if isinstance(image_grid_hws, np.ndarray):
                image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)

            with torch.no_grad():
                vit_embeds = server.extract_feature(pixel_values, image_grid_hws)
                if isinstance(vit_embeds, list) or image_grid_hws is not None:
                    vit_embeds = torch.cat(vit_embeds, dim=0)
                vit_embeds = server.mlp1(vit_embeds)
                
                # 文本 embedding 缝合
                input_embeds = server.standalone_embed_tokens(input_ids).to(server.dtype)
                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)
                input_ids_flat = input_ids.reshape(B * N)
                selected = (input_ids_flat == server.image_token_index)
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
                input_embeds = input_embeds.reshape(B, N, C)
                
                prompt_embeds = input_embeds.squeeze(0).clone().detach()
                encoded_embeds = server._tensor2base64(prompt_embeds)
            
            torch.cuda.synchronize()
            vit_time_ms = (time.perf_counter() - start_vit) * 1000

        # 4. 远端 vLLM 请求阶段 (释放锁，支持并发 I/O)
        start_vllm = time.perf_counter()
        
        # 采用 loop.run_in_executor 防止 openai 同步库阻塞事件循环
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None, 
            lambda: server.vllm_client.completions.create(
                model=self.vllm_model,
                prompt=None,
                max_tokens=1024,
                temperature=0.0,
                extra_body={
                    "prompt_embeds": encoded_embeds,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False
                },
            )
        )
        vllm_time_ms = (time.perf_counter() - start_vllm) * 1000
        
        # 获取纯文本响应
        vllm_output_text = completion.choices[0].text
        
        # 5. 后处理：解析 Bounding Boxes 并画图
        detected_boxes = server._parse_boxes(vllm_output_text, orig_width, orig_height)
        annotated_image_b64 = server._draw_boxes_to_base64(orig_image, detected_boxes)
        
        total_time_ms = (time.perf_counter() - start_total) * 1000
        
        # 6. 回传结果
        return JSONResponse(status_code=200, content={
            "status": "success",
            "raw_output": vllm_output_text,              # 原始信息 (vLLM 返回的文本)
            "parsed_json": {                             # json解析出来的框数据
                "boxes_count": len(detected_boxes),
                "boxes": detected_boxes,                 # [[xmin, ymin, xmax, ymax], ...]
            },
            "annotated_image": annotated_image_b64,      # 回传打标后的图片(Base64格式)
            "metrics": {
                "vit_time_ms": round(vit_time_ms, 2),
                "vllm_time_ms": round(vllm_time_ms, 2),
                "total_time_ms": round(total_time_ms, 2)
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "failed",
            "error": str(e)
        })