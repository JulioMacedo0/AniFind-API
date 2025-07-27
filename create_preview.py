import subprocess
from pathlib import Path
import os

PREVIEW_DIR = Path("previews")
PREVIEW_DURATION = 5  # segundos

def sanitize_filename(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")

def create_preview(video_path: str, second: float) -> Path:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(video_path)
    safe_name = sanitize_filename(input_path.stem)
    start = max(0, second - PREVIEW_DURATION / 2)
    output_filename = f"{safe_name}_{int(second)}.mp4"
    output_path = PREVIEW_DIR / output_filename

    if output_path.exists():
        print(f"[✅] Preview already exists: {output_path}")
        return output_path

    print(f"[🎬] Creating preview: {output_path}")
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(PREVIEW_DURATION),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-y",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)
    return output_path
