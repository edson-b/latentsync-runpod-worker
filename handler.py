"""
Fixed handler for the LatentSync worker (endpoint kwm324hmfdj6kz).

Your website sends: video_url, audio_url, inference_steps, guidance_scale  (no upload_url).
The old handler demanded upload_url, so every job failed instantly. Now upload_url is optional,
and the finished video is returned as video_b64, which your PHP (vd_save_output) already accepts.
"""
import base64
import os
import subprocess
import uuid

import requests
import runpod

MAX_INLINE_BYTES = 6 * 1024 * 1024   # RunPod limits the result size


def download(url, path):
    with requests.get(url, stream=True, timeout=(15, 300)) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)


def handler(job):
    inp = job.get("input") or {}
    for key in ("video_url", "audio_url"):
        if not inp.get(key):
            return {"error": f"Missing required input: {key}"}

    uid = uuid.uuid4().hex
    video = f"/tmp/{uid}.mp4"
    audio = f"/tmp/{uid}.wav"
    out = f"/tmp/{uid}_out.mp4"

    try:
        download(inp["video_url"], video)
        download(inp["audio_url"], audio)

        subprocess.run(
            [
                "python", "-m", "scripts.inference",
                "--unet_config_path", "configs/unet/stage2_512.yaml",
                "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
                "--inference_steps", str(inp.get("inference_steps", 20)),
                "--guidance_scale", str(inp.get("guidance_scale", 1.5)),
                "--video_path", video,
                "--audio_path", audio,
                "--video_out_path", out,
            ],
            check=True,
        )

        if not os.path.exists(out) or os.path.getsize(out) < 1000:
            return {"error": "LatentSync produced no video"}

        size = os.path.getsize(out)
        if inp.get("upload_url"):                      # optional, only if a caller provides one
            with open(out, "rb") as f:
                resp = requests.put(inp["upload_url"], data=f,
                                    headers={"Content-Type": "video/mp4"}, timeout=600)
            resp.raise_for_status()
            return {"status": "done", "video_uploaded": True}

        if size > MAX_INLINE_BYTES:
            return {"error": f"Result video too large to return inline ({size // 1024 // 1024} MB)"}
        with open(out, "rb") as f:
            return {"status": "done", "video_b64": base64.b64encode(f.read()).decode()}

    except subprocess.CalledProcessError as e:
        return {"error": f"LatentSync failed: {e}"}
    except Exception as e:
        return {"error": str(e)[:300]}
    finally:
        for p in (video, audio, out):
            if os.path.exists(p):
                os.remove(p)


runpod.serverless.start({"handler": handler})
