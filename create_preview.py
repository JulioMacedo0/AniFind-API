import subprocess
from pathlib import Path
import os
import time

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
    print(f"[⏱️] Starting FFmpeg generation...")
    
    # Measure preview generation time
    generation_start_time = time.time()
    
    cmd = [
        "ffmpeg",
        "-loglevel", "error",  # Only show errors, suppress info/debug
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(PREVIEW_DURATION),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-y",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time
        
        # Get output file size for reporting
        if output_path.exists():
            output_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"[✅] Preview generation completed: {output_path}")
            print(f"[⏱️] Generation time: {generation_duration:.2f}s | Output size: {output_size_mb:.1f}MB")
        else:
            print(f"[❌] Preview file not created: {output_path}")
            
    except subprocess.CalledProcessError as e:
        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time
        print(f"[❌] FFmpeg failed after {generation_duration:.2f}s: {e}")
        raise
    
    return output_path
