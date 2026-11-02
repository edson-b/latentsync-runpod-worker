FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_DISABLE_XET=1
ENV PYTHONUNBUFFERED=1
# Keep every cache INSIDE the image so nothing is downloaded at runtime
ENV HF_HOME=/app/hf_cache
ENV TORCH_HOME=/root/.cache/torch

# build-essential is needed because insightface compiles a small extension
RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git
WORKDIR /app/LatentSync

RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.txt /app/worker-requirements.txt
RUN pip install --no-cache-dir -r /app/worker-requirements.txt

# Model checkpoints (UNet, whisper, auxiliary face models)
RUN huggingface-cli download ByteDance/LatentSync-1.6 --local-dir checkpoints

# Face-detector weights: link them where torch expects them. Build FAILS if missing.
RUN mkdir -p /root/.cache/torch/hub/checkpoints \
    && for f in 2DFAN4-cd938726ad.zip s3fd-619a316812.pth vgg16-397923af.pth; do \
         test -f "checkpoints/auxiliary/$f" || { echo "MISSING $f"; ls -R checkpoints/auxiliary; exit 1; }; \
         ln -sf "/app/LatentSync/checkpoints/auxiliary/$f" "/root/.cache/torch/hub/checkpoints/$f"; \
       done

# VAE used by the pipeline (otherwise fetched from the internet on every fresh worker)
RUN python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"

# Fail the build if anything essential is missing
RUN test -f checkpoints/latentsync_unet.pt && test -f checkpoints/whisper/small.pt

# Everything is baked in now: block runtime downloads
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY handler.py /app/LatentSync/handler.py

CMD ["python", "-u", "handler.py"]
