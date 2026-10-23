"""
LatentSync worker for RunPod serverless  (endpoint 2zunynale35egv).

Contract with the website (voice-clone/process-video.php + cron-check-jobs.php):
  input : video_url, video_format, audio_url, audio_format, max_video_seconds,
          video_upload_url, inference_steps, guidance_scale   (+ task = "lipsync")
  output: {"video_uploaded": true, "gpu_seconds": N, "measured_seconds": N}
          or {"error": "short reason"}   (the site refunds the customer automatically)

The finished video is POSTed to video_upload_url (a one-time signed link on your site).
The site's cron then finds the file at the name it chose when the job was queued.
"""
import base64
import os
import shutil
import subprocess
import tempfile
import time

import requests
import runpod

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "marketplace-worker/1.0"}   # some hosts/WAFs reject the default python-requests agent
VIDEO_EXTS = {"mp4", "mov", "webm"}
AUDIO_EXTS = {"wav", "mp3", "m4a", "aac", "ogg", "flac"}
MAX_INLINE_BYTES = 6 * 1024 * 1024               # only used by the legacy base64 fallback
INFERENCE_TIMEOUT = 840                          # seconds; endpoint execution timeout is 900


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


def handler(job):
    started = time.time()
    inp = job.get("input") or {}
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

        proc = subprocess.run(
            [
                "python", "-m", "scripts.inference",
                "--unet_config_path", "configs/unet/stage2_512.yaml",
                "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
                "--inference_steps", str(int(inp.get("inference_steps") or 20)),
                "--guidance_scale", str(float(inp.get("guidance_scale") or 1.5)),
                "--video_path", video,
                "--audio_path", audio,
                "--video_out_path", out,
            ],
            cwd=APP_DIR, capture_output=True, text=True, timeout=INFERENCE_TIMEOUT,
        )
        if proc.returncode != 0:
            print("LatentSync stderr tail:", (proc.stderr or "")[-1500:], flush=True)
            return {"error": "LatentSync failed: " + ((proc.stderr or "").strip().splitlines() or ["no output"])[-1][:200]}

        if not os.path.exists(out) or os.path.getsize(out) < 1000:
            return {"error": "LatentSync produced no video"}

        result = {"status": "done", "gpu_seconds": round(time.time() - started, 1),
                  "measured_seconds": round(measured, 2)}

        target = inp.get("video_upload_url") or inp.get("upload_url")   # upload_url = old name
        if target:
            upload(target, out)
            result["video_uploaded"] = True
            return result

        # No upload link supplied: small results can still be returned inline
        size = os.path.getsize(out)
        if size > MAX_INLINE_BYTES:
            return {"error": "No video_upload_url and result too large to return inline"}
        with open(out, "rb") as f:
            result["video_b64"] = base64.b64encode(f.read()).decode()
        return result

    except subprocess.TimeoutExpired:
        return {"error": "LatentSync timed out"}
    except Exception as e:
        return {"error": str(e)[:300]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
