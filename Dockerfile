FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/bytedance/LatentSync.git
WORKDIR /app/LatentSync

# LatentSync's own dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Worker dependencies
COPY requirements.txt /app/worker-requirements.txt
RUN pip install --no-cache-dir -r /app/worker-requirements.txt

# Download model weights into the image (faster cold starts)
RUN huggingface-cli download ByteDance/LatentSync-1.6 --local-dir checkpoints

COPY handler.py /app/LatentSync/handler.py

CMD ["python", "-u", "handler.py"]
