# 1. 基础镜像
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

# 2. 设置工作目录
WORKDIR /app

# 3. 设置环境变量
# 禁用 pip 缓存以减小镜像体积，设置 Python 环境变量
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1\
    PIP_BREAK_SYSTEM_PACKAGES=1

# 4. 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install \
    opencv-python-headless==4.11.0.86 \
    transformers==4.57.1 \
    "numpy<=1.26.0" \
    Pillow==11.1.0 \
    peft \
    torchvision \
    decord==0.6.0 \
    lmdb==1.7.5 \
    fastapi \
    uvicorn \
    python-multipart 

# 5. 下载并内置独立的 Embedding 权重文件
# 该文件用于本地文本 token -> embedding 查表，需与 app.py 工作目录保持一致
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && rm -rf /var/lib/apt/lists/* && \
    wget -q -O /app/qwen2_embed_tokens.safetensors \
    https://huggingface.co/shigureui/LocateAnythingEmb/resolve/main/qwen2_embed_tokens.safetensors

# 6. 复制项目代码
COPY ./app.py /app

# 7. 暴露 FastAPI 默认端口
EXPOSE 8000

# 8. 启动命令（app.py 中定义了 app = FastAPI()）
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]