import os
import re
import gc
import io
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import pybase64
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from safetensors.torch import load_file
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, AutoProcessor

class AdvancedLocateBenchmark:
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
        self.vllm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        print("初始化成功！")

    def _tensor2base64(self, x: torch.Tensor) -> str:
        with io.BytesIO() as buf:
            torch.save(x, buf)
            buf.seek(0)
            binary_data = buf.read()
        return pybase64.b64encode(binary_data).decode("utf-8")

    def predict_single_pipeline(self, image_path: str, categories: list[str]) -> dict:
        """运行单张图片的完整流水线，并精准记录各阶段耗时"""
        metrics = {"vit_time_ms": 0.0, "vllm_time_ms": 0.0, "total_time_ms": 0.0, "status": "success"}
        start_total = time.perf_counter()
        
        try:
            # 1. 图像载入与预处理
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
            
            cats = "</c>".join(categories)
            prompt = f"Locate all the instances that matches the following description: {cats}."
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            
            text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)

            pixel_values = inputs["pixel_values"].to(self.dtype)
            input_ids = inputs["input_ids"]
            image_grid_hws = inputs.get("image_grid_hws", None)
            if isinstance(image_grid_hws, np.ndarray):
                image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)

            # 2. 本地 ViT 提取阶段 (计时开始)
            start_vit = time.perf_counter()
            with torch.no_grad():
                vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
                if isinstance(vit_embeds, list) or image_grid_hws is not None:
                    vit_embeds = torch.cat(vit_embeds, dim=0)
                vit_embeds = self.mlp1(vit_embeds)
                
                # 文本 embedding 缝合
                input_embeds = self.standalone_embed_tokens(input_ids).to(self.dtype)
                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)
                input_ids_flat = input_ids.reshape(B * N)
                selected = (input_ids_flat == self.image_token_index)
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
                input_embeds = input_embeds.reshape(B, N, C)
                
                prompt_embeds = input_embeds.squeeze(0).clone().detach()
                encoded_embeds = self._tensor2base64(prompt_embeds)
            
            # 强行同步 GPU 确保计时准确
            torch.cuda.synchronize()
            metrics["vit_time_ms"] = (time.perf_counter() - start_vit) * 1000

            # 3. 远端 vLLM 请求阶段 (计时开始)
            start_vllm = time.perf_counter()
            completion = self.vllm_client.completions.create(
                model="shigureui/LocateAnything-Qwen2-FP8",
                prompt=None,
                max_tokens=1024,
                temperature=0.0,
                extra_body={
                    "prompt_embeds": encoded_embeds,
                    # "return_token_ids": True,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False
                },
            )
            metrics["vllm_time_ms"] = (time.perf_counter() - start_vllm) * 1000
            
        except Exception as e:
            metrics["status"] = f"failed: {str(e)}"
            
        metrics["total_time_ms"] = (time.perf_counter() - start_total) * 1000
        return metrics

# ==========================================
# 并发性能测试入口
# ==========================================
if __name__ == "__main__":
    pic_dir = "./pic"
    test_categories = ["person", "bicycle"]
    
    # 🔍 【核心参数控制】: max_workers 就是你的并发数 (相当于并发 Batch Size)
    # 如果想测试单线程吞吐，设为 1；如果想压榨 vLLM 极限，可以设为 4, 8, 16...
    MAX_CONCURRENT_WORKERS = 4

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_list = []
    for ext in extensions:
        image_list.extend(glob.glob(os.path.join(pic_dir, ext)))
        
    if not image_list:
        print(f"未在 {pic_dir} 找到图片。")
        exit()

    print(f"--- 开始性能评测 ---")
    print(f"图片总数: {len(image_list)} | 设定并发数(线程数): {MAX_CONCURRENT_WORKERS}")
    
    benchmarker = AdvancedLocateBenchmark()
    
    total_vit_time = 0.0
    total_vllm_time = 0.0
    success_count = 0
    
    start_benchmark_wall_time = time.perf_counter()
    
    # 利用线程池并发处理图片群
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        # 提交所有任务
        future_to_img = {executor.submit(benchmarker.predict_single_pipeline, img, test_categories): img for img in image_list}
        
        for future in as_completed(future_to_img):
            img_name = os.path.basename(future_to_img[future])
            res = future.result()
            
            if res["status"] == "success":
                success_count += 1
                total_vit_time += res["vit_time_ms"]
                total_vllm_time += res["vllm_time_ms"]
                print(f"图片: {img_name} -> [本地ViT+缝合]: {res['vit_time_ms']:.1f}ms | [远端vLLM生成]: {res['vllm_time_ms']:.1f}ms")
            else:
                print(f"图片: {img_name} -> 处理失败: {res['status']}")

    end_benchmark_wall_time = time.perf_counter()
    total_wall_time_s = end_benchmark_wall_time - start_benchmark_wall_time
    
    # --- 最终 Benchmark 报告 ---
    print("\n" + "="*40)
    print("           BENCHMARK REPORT           ")
    print("="*40)
    if success_count > 0:
        print(f"成功处理图片数: {success_count}/{len(image_list)}")
        print(f"单图平均 本地ViT 耗时: {total_vit_time / success_count:.2f} ms")
        print(f"单图平均 远端vLLM 耗时: {total_vllm_time / success_count:.2f} ms")
        print(f"【对比结论】: vLLM 是 ViT 的 {total_vllm_time / total_vit_time:.1f} 倍耗时")
    print("-" * 40)
    print(f"总并发实际墙钟耗时 (Wall Time): {total_wall_time_s:.2f} 秒")
    print(f"系统吞吐量 (Throughput): {success_count / total_wall_time_s:.2f} images/sec")
    print("="*40)