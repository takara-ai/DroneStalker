import os
import subprocess
import sys
import glob

def make_video(image_dir, output="output.mp4", fps=30):
    os.chdir(image_dir)
    # Detect extension
    exts = ["*.png", "*.jpg", "*.jpeg"]
    patterns = [glob.glob(e) for e in exts]
    images = [img for group in patterns for img in group]

    if not images:
        print("No images found.")
        return

    # ffmpeg image2 loader = correct way
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", "*.png",
        "-pix_fmt", "yuv420p",
        output
    ]

    subprocess.run(cmd)
    print("Done:", output)

if __name__ == "__main__":
    make_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output.mp4")
