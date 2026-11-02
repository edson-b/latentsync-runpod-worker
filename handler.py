"""
LatentSync worker for RunPod serverless  (endpoint 2zunynale35egv).

Models are loaded ONCE when the worker starts (not once per job), so each job
only pays for the actual lip-sync work.

Contract with the website (voice-clone/process-video.php + cron-check-jobs.php):
  input : video_url, video_format, audio_url, audio_format, max_video_seconds,
          video_upload_url, inference_steps, guidance_scale   (+ task = "lipsync")
  output: {"video_uploaded": true, "gpu_seconds": N, "measured_seconds": N}
          or {"error": "short reason"}   (the site refunds the customer automatically)
"""
import base64
import os
import shutil
import subprocess
import tempfile
import time

import requests
import runpod
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed
from DeepCache import DeepCacheSDHelper

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)                                # LatentSync uses paths relative to its repo root

UA = {"User-Agent": "marketplace-worker/1.0"}
VIDEO_EXTS = {"mp4", "mov", "webm"}
AUDIO_EXTS = {"wav", "mp3", "m4a", "aac", "ogg", "flac"}
MAX_INLINE_BYTES = 6 * 1024 * 1024               # only used by the legacy base64 fallback

UNET_CONFIG = "configs/unet/stage2_512.yaml"
UNET_CKPT = "checkpoints/latentsync_unet.pt"
USE_DEEPCACHE = os.environ.get("LS_DEEPCACHE", "1") == "1"   # set LS_DEEPCACHE=0 if quality drops
SEED = 1247


# --------------------------------------------------------------------------
# Model loading: runs once per worker, at startup
# --------------------------------------------------------------------------
def load_pipeline():
    t0 = time.time()
    config = OmegaConf.load(UNET_CONFIG)

    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16 else torch.float32

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), UNET_CKPT, device="cpu"
    )
    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler
    ).to("cuda")

    if USE_DEEPCACHE:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    print(f"Models loaded in {time.time() - t0:.1f}s", flush=True)
    return pipeline, config, dtype


PIPELINE, CONFIG, DTYPE = load_pipeline()


def run_lipsync(video, audio, out, work, steps, guidance):
    set_seed(SEED)
    PIPELINE(
        video_path=video,
        audio_path=audio,
        video_out_path=out,
        num_frames=CONFIG.data.num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        weight_dtype=DTYPE,
        width=CONFIG.data.resolution,
        height=CONFIG.data.resolution,
        mask_image_path=CONFIG.data.mask_image_path,
        temp_dir=os.path.join(work, "tmp"),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def download(url, path):
    with requests.get(url, headers=UA, stream=True, timeout=(15, 300)) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    if os.path.getsize(path) < 1000:
        raise RuntimeError("downloaded file is empty")


def duration(path):
    """Length in seconds via ffprobe (ships with ffmpeg). Returns 0.0 if unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def upload(url, path):
    with open(path, "rb") as f:
        r = requests.post(url, data=f, headers={**UA, "Content-Type": "application/octet-stream"},
                          timeout=(15, 600))
    r.raise_for_status()


# --------------------------------------------------------------------------
# Job handler
# --------------------------------------------------------------------------
def handler(job):
    started = time.time()
    inp = job.get("input") or {}
    if inp.get("task") == "warmup":          # models are already loaded at startup, so this just confirms readiness
        return {"status": "warm"}
    for key in ("video_url", "audio_url"):
        if not inp.get(key):
            return {"error": f"Missing required input: {key}"}

    vext = str(inp.get("video_format") or "mp4").lower().lstrip(".")
    aext = str(inp.get("audio_format") or "wav").lower().lstrip(".")
    if vext not in VIDEO_EXTS:
        return {"error": f"Unsupported video format: {vext}"}
    if aext not in AUDIO_EXTS:
        return {"error": f"Unsupported audio format: {aext}"}

    work = tempfile.mkdtemp(prefix="ls_")
    video = os.path.join(work, f"in.{vext}")
    audio = os.path.join(work, f"in.{aext}")
    out = os.path.join(work, "out.mp4")

    try:
        download(inp["video_url"], video)
        download(inp["audio_url"], audio)

        measured = duration(video)
        max_sec = float(inp.get("max_video_seconds") or 0)
        if max_sec and measured > max_sec + 1:
            return {"error": f"Video too long ({measured:.0f}s, limit {max_sec:.0f}s)"}

        steps = int(inp.get("inference_steps") or 20)
        guidance = float(inp.get("guidance_scale") or 1.5)

        run_lipsync(video, audio, out, work, steps, guidance)

        if not os.path.exists(out) or os.path.getsize(out) < 1000:
            return {"error": "LatentSync produced no video"}

        result = {"status": "done", "gpu_seconds": round(time.time() - started, 1),
                  "measured_seconds": round(measured, 2)}

        target = inp.get("video_upload_url") or inp.get("upload_url")   # upload_url = old name
        if target:
            upload(target, out)
            result["video_uploaded"] = True
            return result

        size = os.path.getsize(out)
        if size > MAX_INLINE_BYTES:
            return {"error": "No video_upload_url and result too large to return inline"}
        with open(out, "rb") as f:
            result["video_b64"] = base64.b64encode(f.read()).decode()
        return result

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory (video too large)"}
    except Exception as e:
        print("Job failed:", repr(e), flush=True)
        return {"error": str(e)[:300]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
