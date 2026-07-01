# 1. 基础镜像
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

# 2. 设置工作目录
WORKDIR /app

# 3. 设置环境变量
# 禁用 pip 缓存以减小镜像体积，设置 Python 环境变量
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 5. 安装 Python 依赖
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


# 7. 暴露 FastAPI 默认端口
EXPOSE 8000

# 8. 启动命令（假设你的 FastAPI 启动文件叫 main.py，里面定义了 app = FastAPI()）
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]