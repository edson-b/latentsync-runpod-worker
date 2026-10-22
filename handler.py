import os
import subprocess
import uuid

import requests
import runpod


def download(url, path):
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)


def handler(job):
    inp = job["input"]

    for key in ("video_url", "audio_url", "upload_url"):
        if key not in inp:
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

        # Upload the result to a presigned PUT URL provided by your website
        with open(out, "rb") as f:
            resp = requests.put(
                inp["upload_url"],
                data=f,
                headers={"Content-Type": "video/mp4"},
                timeout=600,
            )
        resp.raise_for_status()

        return {"status": "done", "output_url": inp.get("output_url")}

    except subprocess.CalledProcessError as e:
        return {"error": f"LatentSync failed: {e}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        for p in (video, audio, out):
            if os.path.exists(p):
                os.remove(p)


runpod.serverless.start({"handler": handler})
