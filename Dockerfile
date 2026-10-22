FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/bytedance/LatentSync.git
WORKDIR /app/LatentSync

RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.txt /app/worker-requirements.txt
RUN pip install --no-cache-dir -r /app/worker-requirements.txt

RUN huggingface-cli download ByteDance/LatentSync-1.6 --local-dir checkpoints

RUN mkdir -p /root/.cache/torch/hub/checkpoints \
    && for f in 2DFAN4-cd938726ad.zip s3fd-619a316812.pth vgg16-397923af.pth; do \
         if [ -f "checkpoints/auxiliary/$f" ]; then \
           ln -sf "/app/LatentSync/checkpoints/auxiliary/$f" "/root/.cache/torch/hub/checkpoints/$f"; \
         fi; \
       done

COPY handler.py /app/LatentSync/handler.py

CMD ["python", "-u", "handler.py"]
