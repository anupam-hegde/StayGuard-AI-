import os
import subprocess
from PIL import Image

webp_path = r"C:\Users\anupa\.gemini\antigravity-ide\brain\77aa5ea0-581c-49ec-91d0-37e5ab18aa6c\stayguard_ui_walkthrough_1793115149486.webp"
temp_frames_dir = "brag-output/work/walkthrough_frames"
os.makedirs(temp_frames_dir, exist_ok=True)

print(f"Opening animated WebP: {webp_path}")
im = Image.open(webp_path)
n_frames = im.n_frames
print(f"Total frames to extract: {n_frames}")

for i in range(n_frames):
    im.seek(i)
    # Convert RGBA to RGB with dark background fill
    frame = Image.new("RGB", (1920, 1080), (8, 12, 22))
    # Center the (1920, 982) frame vertically
    frame.paste(im.convert("RGB"), (0, 49))
    frame.save(os.path.join(temp_frames_dir, f"frame_{i:04d}.png"))
    if (i + 1) % 50 == 0:
        print(f"Extracted {i + 1}/{n_frames} frames")

print("All frames extracted. Encoding to MP4 via ffmpeg...")
mp4_out = "docs/assets/ui_walkthrough_demo.mp4"
cmd_mp4 = [
    "ffmpeg", "-y",
    "-r", "20",
    "-i", os.path.join(temp_frames_dir, "frame_%04d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "19",
    "-preset", "fast",
    "-movflags", "+faststart",
    mp4_out
]
subprocess.run(cmd_mp4, check=True)
print(f"MP4 encoded successfully: {mp4_out}")

print("Generating optimized animated WebP stream...")
webp_out = "docs/assets/ui_walkthrough_demo.webp"
cmd_webp = [
    "ffmpeg", "-y",
    "-i", mp4_out,
    "-vf", "fps=10,scale=960:-1:flags=lanczos",
    "-loop", "0",
    webp_out
]
subprocess.run(cmd_webp, check=True)
print(f"Animated WebP created: {webp_out}")
