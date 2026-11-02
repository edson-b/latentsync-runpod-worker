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
# Pinned to the newest LatentSync commit at time of writing (20 Jun 2025)
RUN git clone https://github.com/bytedance/LatentSync.git && \
    cd LatentSync && git checkout a229c39
WORKDIR /app/LatentSync

RUN pip install --no-cache-dir -r requirements.txt

# Worker extras (this is the VIDEO repo's requirements.txt: runpod, requests, huggingface_hub)
COPY requirements.txt /app/worker-requirements.txt
RUN pip install --no-cache-dir -r /app/worker-requirements.txt

# Model checkpoints (UNet, whisper tiny, auxiliary face models)
RUN huggingface-cli download ByteDance/LatentSync-1.6 --local-dir checkpoints

# Link the VGG weights where torch looks for them. Other auxiliary files are optional
# (current LatentSync uses insightface, not the old 2DFAN4 / s3fd files).
RUN mkdir -p /root/.cache/torch/hub/checkpoints \
    && for f in vgg16-397923af.pth; do \
         if [ -f "checkpoints/auxiliary/$f" ]; then \
           ln -sf "/app/LatentSync/checkpoints/auxiliary/$f" "/root/.cache/torch/hub/checkpoints/$f"; \
         else echo "NOTE: $f not found"; fi; \
       done; \
    echo "--- auxiliary contents ---"; ls -R checkpoints/auxiliary | head -50

# VAE used by the pipeline (otherwise fetched from the internet on every fresh worker)
RUN python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"

# Fail the build if anything essential is missing (the config uses whisper TINY, not small)
RUN test -f checkpoints/latentsync_unet.pt && test -f checkpoints/whisper/tiny.pt

# Everything is baked in now: block runtime downloads
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY handler.py /app/LatentSync/handler.py

CMD ["python", "-u", "handler.py"]
