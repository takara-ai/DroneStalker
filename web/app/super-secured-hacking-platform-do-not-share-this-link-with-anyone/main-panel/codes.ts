export const codes = {
  fan: {
    code: `import argparse  # noqa: INP001
import time

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource
from rpm_estimator import RpmEstimator


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get indexes corresponding to events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),  # gray
    on_color: tuple[int, int, int] = (255, 255, 255),  # white
    off_color: tuple[int, int, int] = (0, 0, 0),  # black
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (0, 0, 0),  # black by default
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row_str = (
            f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # second row
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window", type=float, default=10, help="Windows duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    args = parser.parse_args()

    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    # Enforce playback speed via dropping:
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)

    # RPM estimator tuned around ~1000 RPM (~16.7 Hz)
    rpm_est = RpmEstimator(min_hz=5.0, max_hz=80.0, history_s=6.0)
    for batch_range in pacer.pace(src.ranges()):
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        frame = get_frame(window)
        draw_hud(frame, pacer, batch_range)

        # Update and overlay RPM estimate
        t_end_s = batch_range.end_ts_us / 1e6
        win_s = src.window_length_us / 1e6 if hasattr(src, "window_length_us") else (args.window / 1000.0)
        est = rpm_est.update(window, t_end_s, win_s)
        if est.rpm is not None:
            RpmEstimator.overlay_text(frame, f"RPM ≈ {est.rpm:6.1f}")
        else:
            RpmEstimator.overlay_text(frame, "RPM ≈ …")

        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
`,
    source: "https://github.com/takara-ai/DroneStalker",
  },
  motionDetection: {
    code: `#!/usr/bin/env python3
"""
Script to create videos from FRED dataset frames using ffmpeg.

Creates two videos:
1. colored.mp4 - from RGB frames
2. motion.mp4 - from Event frames
"""

import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Tuple, Optional

# Configuration
SEQUENCE_FOLDER = "60"  # Change this to select different sequence
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "web" / "public" / "data" / SEQUENCE_FOLDER

# Frame directories
RGB_DIR = DATA_DIR / SEQUENCE_FOLDER / "RGB"
EVENT_DIR = DATA_DIR / SEQUENCE_FOLDER / "Event" / "Frames"

# Output video files
COLORED_VIDEO = OUTPUT_DIR / "colored.mp4"
MOTION_VIDEO = OUTPUT_DIR / "motion.mp4"

# Video settings
FPS = 30  # Frames per second
VIDEO_CODEC = "libx264"  # H.264 codec for MP4
QUALITY = "good"  # Quality preset: good, best, realtime
PRESET = "medium"  # Encoding preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow

# Compression settings
TARGET_SIZE_MB = 8  # Target file size in MB (None to use CRF quality instead)
TARGET_SIZE_TOLERANCE = 0.1  # Allow 10% variance in target size
USE_2PASS = False  # Use 2-pass encoding for better quality at target bitrate (slower but better)
MAX_RESOLUTION = None  # Max resolution (e.g., "1280:720" or None for original)
CRF = 23  # Quality (for H.264: 0-51, lower = better quality, higher = smaller files, 23 is default)
# Only used if TARGET_SIZE_MB is None


def check_ffmpeg():
    """Check if ffmpeg is installed and available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        return False


def check_directories():
    """Check if required directories exist."""
    if not RGB_DIR.exists():
        print(f"Error: RGB directory not found: {RGB_DIR}")
        return False

    if not EVENT_DIR.exists():
        print(f"Error: Event frames directory not found: {EVENT_DIR}")
        return False

    return True


def count_frames(directory, extension):
    """Count frames in a directory."""
    frames = list(directory.glob(f"*.{extension}"))
    return len(frames)


def parse_rgb_timestamp(filename: str) -> Optional[float]:
    """
    Parse timestamp from RGB filename.
    Format: Video_230_17_16_02.213938.jpg -> 17_16_02.213938
    Returns: seconds since start (or absolute time in seconds)
    """
    # Extract timestamp pattern: HH_MM_SS.microseconds
    match = re.search(r"(\d+)_(\d+)_(\d+)\.(\d+)", filename)
    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    microseconds = int(match.group(4))

    # Convert to total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + microseconds / 1_000_000

    return total_seconds


def parse_event_timestamp(filename: str) -> Optional[float]:
    """
    Parse timestamp from Event filename.
    Formats:
    - Video_0_frame_100032333.png -> 100032333 (microseconds)
    - Video_0_100032333.png -> 100032333 (microseconds)
    Returns: seconds (microseconds / 1_000_000)
    """
    # Extract the numeric timestamp
    # Pattern matches both: Video_<number>_frame_<timestamp>.png and Video_<number>_<timestamp>.png
    match = re.search(r"Video_\d+_(?:frame_)?(\d+)\.png", filename)
    if not match:
        return None

    microseconds = int(match.group(1))
    return microseconds / 1_000_000


def extract_timestamps(frames: List[Path], is_rgb: bool) -> List[Tuple[Path, float]]:
    """
    Extract timestamps from frame filenames.

    Returns:
        List of (frame_path, timestamp_in_seconds) tuples, sorted by timestamp
        Timestamps are normalized to start from 0 (relative to first frame)
    """
    frame_timestamps = []

    for frame in frames:
        if is_rgb:
            timestamp = parse_rgb_timestamp(frame.name)
        else:
            timestamp = parse_event_timestamp(frame.name)

        if timestamp is not None:
            frame_timestamps.append((frame, timestamp))
        else:
            print(f"Warning: Could not parse timestamp from {frame.name}")

    # Sort by timestamp
    frame_timestamps.sort(key=lambda x: x[1])

    # Normalize timestamps to start from 0 (relative to first frame)
    if frame_timestamps:
        first_timestamp = frame_timestamps[0][1]
        normalized = [(frame, ts - first_timestamp) for frame, ts in frame_timestamps]
        return normalized

    return frame_timestamps


def calculate_frame_rate(frame_timestamps: List[Tuple[Path, float]]) -> float:
    """
    Calculate average frame rate from timestamps.

    Returns:
        Average frames per second
    """
    if len(frame_timestamps) < 2:
        return FPS  # Default if we can't calculate

    # Calculate time differences
    time_diffs = []
    for i in range(1, len(frame_timestamps)):
        diff = frame_timestamps[i][1] - frame_timestamps[i - 1][1]
        if diff > 0:  # Only positive differences
            time_diffs.append(diff)

    if not time_diffs:
        return FPS

    # Calculate average time difference
    avg_diff = sum(time_diffs) / len(time_diffs)

    # Frame rate is inverse of average time difference
    calculated_fps = 1.0 / avg_diff

    return calculated_fps


def calculate_target_bitrate(duration_seconds: float, target_size_mb: float) -> int:
    """
    Calculate target bitrate in kbps for a given duration and target file size.

    Args:
        duration_seconds: Video duration in seconds
        target_size_mb: Target file size in MB

    Returns:
        Target bitrate in kbps
    """
    if duration_seconds <= 0:
        return 500  # Default bitrate

    # Convert MB to bits, then divide by duration to get bits per second
    target_bits = target_size_mb * 8 * 1024 * 1024  # MB to bits
    target_bps = target_bits / duration_seconds
    target_kbps = int(target_bps / 1000)

    # Ensure minimum bitrate
    return max(target_kbps, 100)


def create_video(input_dir, output_file, extension, is_rgb=True, fps=None):
    """
    Create a video from frames using ffmpeg with timestamp-based timing.

    Args:
        input_dir: Directory containing frames
        output_file: Output video file path
        extension: File extension (e.g., "jpg", "png")
        is_rgb: True for RGB frames, False for Event frames
        fps: Optional FPS override (if None, will be calculated from timestamps)
    """
    # Get all frames
    frames = list(input_dir.glob(f"*.{extension}"))

    if not frames:
        print(f"Error: No {extension} files found in {input_dir}")
        return False

    # Extract timestamps from filenames
    print(f"Extracting timestamps from {len(frames)} frames...")
    frame_timestamps = extract_timestamps(frames, is_rgb)

    if not frame_timestamps:
        print(f"Error: Could not extract timestamps from frames")
        return False

    # Calculate frame rate from timestamps if not provided
    if fps is None:
        calculated_fps = calculate_frame_rate(frame_timestamps)
        fps = calculated_fps
        print(f"Calculated frame rate from timestamps: {fps:.3f} FPS")
    else:
        print(f"Using specified frame rate: {fps} FPS")

    # Calculate video duration
    video_duration = frame_timestamps[-1][1] - frame_timestamps[0][1]
    print(f"Video duration: {video_duration:.2f} seconds")

    # Determine encoding method based on target size
    use_target_bitrate = TARGET_SIZE_MB is not None
    if use_target_bitrate:
        target_bitrate = calculate_target_bitrate(video_duration, TARGET_SIZE_MB)
        print(f"Target file size: {TARGET_SIZE_MB} MB")
        print(f"Target bitrate: {target_bitrate} kbps")
    else:
        print(f"Using CRF quality mode (CRF={CRF})")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create file list with duration information for proper timing
    file_list = output_file.parent / f".{output_file.stem}_filelist.txt"

    try:
        # Write file list with duration based on actual timestamps
        with open(file_list, "w") as f:
            for i, (frame, timestamp) in enumerate(frame_timestamps):
                f.write(f"file '{frame.absolute()}'\n")

                # Calculate duration until next frame
                if i < len(frame_timestamps) - 1:
                    next_timestamp = frame_timestamps[i + 1][1]
                    duration = next_timestamp - timestamp
                    # Ensure minimum duration
                    duration = max(duration, 1.0 / (fps * 2))
                else:
                    # Last frame: use average frame duration
                    duration = 1.0 / fps

                f.write(f"duration {duration:.6f}\n")

            # Repeat last frame to ensure proper duration
            if frame_timestamps:
                last_frame, _ = frame_timestamps[-1]
                f.write(f"file '{last_frame.absolute()}'\n")

        # Build base ffmpeg command
        base_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list),
            "-vsync",
            "cfr",  # Constant frame rate output
            "-r",
            str(fps),  # Output frame rate
            "-c:v",
            VIDEO_CODEC,
            "-pix_fmt",
            "yuv420p",  # Ensure compatibility
        ]

        # Add resolution scaling if specified
        if MAX_RESOLUTION:
            base_cmd.extend(["-vf", f"scale={MAX_RESOLUTION}"])

        # Add encoding parameters based on target size or CRF
        if use_target_bitrate and USE_2PASS:
            # 2-pass encoding for better quality at target bitrate
            # First pass: write stats file
            stats_file = output_file.parent / f".{output_file.stem}.log"

            pass1_cmd = base_cmd + [
                "-b:v",
                f"{target_bitrate}k",
                "-maxrate",
                f"{int(target_bitrate * 1.2)}k",  # 20% buffer
                "-minrate",
                f"{int(target_bitrate * 0.8)}k",  # 20% buffer
                "-pass",
                "1",
                "-passlogfile",
                str(stats_file),
                "-f",
                "null",
                "/dev/null",  # Discard output for first pass
            ]

            pass2_cmd = base_cmd + [
                "-b:v",
                f"{target_bitrate}k",
                "-maxrate",
                f"{int(target_bitrate * 1.2)}k",
                "-minrate",
                f"{int(target_bitrate * 0.8)}k",
                "-pass",
                "2",
                "-passlogfile",
                str(stats_file),
                "-preset",
                PRESET,
                "-stats_period",
                "0.5",
                "-loglevel",
                "info",
                "-an",
                str(output_file),
            ]

            # Run 2-pass encoding
            print("\nRunning 2-pass encoding...")
            print("Pass 1/2: Analyzing video...")
            result1 = subprocess.run(pass1_cmd, capture_output=True, text=True)
            if result1.returncode != 0:
                print(f"✗ Pass 1 failed: {result1.stderr}")
                return False

            print("Pass 2/2: Encoding video...")
            print("\nEncoding progress:")
            print("-" * 60)
            cmd = pass2_cmd
        elif use_target_bitrate:
            # Single-pass with target bitrate
            cmd = base_cmd + [
                "-b:v",
                f"{target_bitrate}k",
                "-maxrate",
                f"{int(target_bitrate * 1.2)}k",
                "-minrate",
                f"{int(target_bitrate * 0.8)}k",
                "-preset",
                PRESET,
                "-stats_period",
                "0.5",
                "-loglevel",
                "info",
                "-an",
                str(output_file),
            ]
        else:
            # CRF quality mode
            cmd = base_cmd + [
                "-crf",
                str(CRF),
                "-b:v",
                "0",  # Variable bitrate
                "-preset",
                PRESET,
                "-stats_period",
                "0.5",
                "-loglevel",
                "info",
                "-an",
                str(output_file),
            ]

        if not (use_target_bitrate and USE_2PASS):
            print(f"\nCreating video: {output_file.name}")
            print(f"Input directory: {input_dir}")
            print(f"Extension: {extension}")
            print(f"Frames: {len(frame_timestamps)}")
            print(f"FPS: {fps:.3f}")
            print("\nEncoding progress:")
            print("-" * 60)

        # Run ffmpeg and show progress in real-time
        # For 2-pass, pass 1 is already done, so we only need to run pass 2
        if use_target_bitrate and USE_2PASS:
            # Pass 2 command is already set, run it with progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr for progress
                text=True,
                bufsize=1,  # Line buffered
            )
        else:
            # Single pass encoding
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr for progress
                text=True,
                bufsize=1,  # Line buffered
            )

        if process:
            # Read and display progress in real-time from stderr
            def read_stderr():
                """Read stderr and parse progress information."""
                last_frame = 0
                for line in iter(process.stderr.readline, ""):
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        # Parse and display progress information
                        # ffmpeg progress format: frame=  123 fps= 30 q=28.0 size=    1024kB time=00:00:04.12 bitrate= 512.0kbits/s speed=1.0x
                        if "frame=" in line:
                            # Extract frame number
                            frame_match = re.search(r"frame=\s*(\d+)", line)
                            if frame_match:
                                frame_num = int(frame_match.group(1))
                                last_frame = frame_num

                                # Extract other useful info
                                fps_match = re.search(r"fps=\s*([\d.]+)", line)
                                time_match = re.search(r"time=(\d+:\d+:\d+\.\d+)", line)
                                bitrate_match = re.search(
                                    r"bitrate=\s*([\d.]+)\s*(\w+)", line
                                )
                                speed_match = re.search(r"speed=\s*([\d.]+)x", line)

                                # Build progress string
                                progress_pct = (frame_num / len(frame_timestamps)) * 100
                                info_parts = []

                                if fps_match:
                                    info_parts.append(f"{fps_match.group(1)} fps")
                                if time_match:
                                    info_parts.append(f"time={time_match.group(1)}")
                                if bitrate_match:
                                    info_parts.append(
                                        f"{bitrate_match.group(1)} {bitrate_match.group(2)}"
                                    )
                                if speed_match:
                                    info_parts.append(f"{speed_match.group(1)}x")

                                info_str = " | ".join(info_parts) if info_parts else ""

                                # Display progress (overwrite same line)
                                print(
                                    f"\r📹 Frame {frame_num}/{len(frame_timestamps)} ({progress_pct:.1f}%) | {info_str}",
                                    end="",
                                    flush=True,
                                )

            # Start reading stderr in a separate thread
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Wait for process to complete
            process.wait()
            stderr_thread.join(timeout=1)  # Wait for stderr reading to finish

            # Clear the progress line and show completion
            print()  # New line after progress

            if process.returncode == 0:
                # Clean up stats file if it exists
                if use_target_bitrate and USE_2PASS:
                    stats_file = output_file.parent / f".{output_file.stem}.log"
                    if stats_file.exists():
                        try:
                            stats_file.unlink()
                        except:
                            pass

                print(f"✓ Successfully created {output_file.name}")
                return True
            else:
                print(f"✗ Error creating video (exit code: {process.returncode})")
                # Show error output
                if process.stderr:
                    try:
                        stderr_output = process.stderr.read()
                        if stderr_output:
                            print(f"Error details:\n{stderr_output}")
                    except:
                        pass
                return False
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error creating video: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False
    finally:
        # Clean up temporary file list
        if file_list.exists():
            file_list.unlink()


def move_coordinates_file():
    """
    Move coordinates.txt file from data directory to output directory.

    Returns:
        True if file was moved successfully, False otherwise
    """
    source_file = DATA_DIR / SEQUENCE_FOLDER / "coordinates.txt"
    dest_file = OUTPUT_DIR / "coordinates.txt"

    if not source_file.exists():
        print(f"Warning: coordinates.txt not found at {source_file}")
        return False

    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Move the file
        shutil.move(str(source_file), str(dest_file))
        print(f"✓ Moved coordinates.txt to output directory")
        return True
    except Exception as e:
        print(f"✗ Error moving coordinates.txt: {e}")
        return False


def main():
    """Main function to create videos."""
    print("=" * 60)
    print("FRED Dataset Video Creator (MP4)")
    print("=" * 60)
    print(f"Sequence folder: {SEQUENCE_FOLDER}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Check prerequisites
    if not check_ffmpeg():
        sys.exit(1)

    if not check_directories():
        sys.exit(1)

    # Count frames
    rgb_count = count_frames(RGB_DIR, "jpg")
    event_count = count_frames(EVENT_DIR, "png")

    print(f"Found {rgb_count} RGB frames")
    print(f"Found {event_count} Event frames")
    print()

    if rgb_count == 0:
        print("Error: No RGB frames found")
        sys.exit(1)

    if event_count == 0:
        print("Error: No Event frames found")
        sys.exit(1)

    # Create RGB video (colored.mp4)
    print("\n" + "=" * 60)
    print("Creating RGB video (colored.mp4)")
    print("=" * 60)
    success_rgb = create_video(RGB_DIR, COLORED_VIDEO, "jpg", is_rgb=True, fps=None)

    # Create Event video (motion.mp4)
    print("\n" + "=" * 60)
    print("Creating Event video (motion.mp4)")
    print("=" * 60)
    success_event = create_video(EVENT_DIR, MOTION_VIDEO, "png", is_rgb=False, fps=None)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if success_rgb:
        size = COLORED_VIDEO.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ colored.mp4 created ({size:.2f} MB)")
    else:
        print("✗ Failed to create colored.mp4")

    if success_event:
        size = MOTION_VIDEO.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ motion.mp4 created ({size:.2f} MB)")
    else:
        print("✗ Failed to create motion.mp4")

    # Move coordinates.txt file
    if success_rgb and success_event:
        print()
        move_coordinates_file()
        print("\n✓ All videos created successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some videos failed to create")
        sys.exit(1)


if __name__ == "__main__":
    main()
`,
    source: "https://github.com/takara-ai/DroneStalker",
  },
  tracking: {
    code: `# -------------------------------------------------------
# File: yolo/annotate.py
# -------------------------------------------------------
import os
import csv
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
IMAGE_DIR = "/home/mikus/Downloads/195/Event/Frames"
OUTPUT_DIR = "output_images"
CSV_PATH = "detections.csv"
MODEL_PATH = "best.pt"

# Performance tuning parameters
IMGSZ = 640          # Reduce to 480 or 320 for faster inference
CONF_THRESHOLD = 0.25  # Higher = fewer detections = faster
IOU_THRESHOLD = 0.45
BATCH_SIZE = 4       # Process multiple images at once (adjust based on RAM)

# -------------------------------------------------------
# Load model with optimizations
model = YOLO(MODEL_PATH)

# Export to ONNX for faster CPU inference (do this once)
try:
    onnx_path = MODEL_PATH.replace('.pt', '.onnx')
    if not os.path.exists(onnx_path):
        print("Exporting model to ONNX for optimized CPU inference...")
        # Use dynamic=True to support different batch sizes
        model.export(format='onnx', simplify=True, dynamic=True, imgsz=IMGSZ)
        print(f"ONNX model saved to: {onnx_path}")
    
    # Load the ONNX model
    model = YOLO(onnx_path, task='detect')
    print("Using ONNX model for inference")
except Exception as e:
    print(f"ONNX export failed, using PyTorch model: {e}")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect list of images
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
]

# Prepare CSV file
with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "x1", "y1", "x2", "y2", "confidence", "class_id", "class_name"])
    
    # Process images in batches
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):
        batch_files = image_files[i:i + BATCH_SIZE]
        batch_paths = [os.path.join(IMAGE_DIR, f) for f in batch_files]
        
        # Run inference on batch
        results = model(
            batch_paths,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            half=False,  # FP16 not helpful on CPU
            device='cpu',
            verbose=False,
            augment=False,  # Disable test-time augmentation
            agnostic_nms=False
        )
        
        # Process each result
        for fname, r in zip(batch_files, results):
            img = cv2.imread(os.path.join(IMAGE_DIR, fname))
            
            # Loop through detections
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Save row to CSV
                writer.writerow([fname, x1, y1, x2, y2, conf, cls, class_name])
                
                # Draw bounding box
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
            
            # Save annotated image
            out_path = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(out_path, img)

print("Done! CSV saved to:", CSV_PATH)
print("Annotated images saved in:", OUTPUT_DIR)


# -------------------------------------------------------
# File: yolo/convert_to_yolo.py
# -------------------------------------------------------
import os
import shutil
from data import parse, path
from tqdm import tqdm

runs = [15, 10, 1, 0, 63, 60, 195, 194, 187, 179, 178, 167]

pattern_path = "/home/mikus/Downloads/{}/"
image_dir = "Event/Frames/"

# Collect events
events = []
for seq in runs:
    actions = parse(pattern_path.format(seq))
    for action in actions:
        image_path = os.path.join(
            pattern_path.format(seq),
            image_dir,
            path(seq, action["name"])
        )
        events.append((seq, action, image_path))

print("Total events:", len(events))
print("Example:", events[0])

# Prepare YOLO directories
os.makedirs("data_yolo/images", exist_ok=True)
os.makedirs("data_yolo/labels", exist_ok=True)

def convert_to_yolo(x1, y1, x2, y2, w, h):
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = (x1 + (x2 - x1) / 2) / w
    cy = (y1 + (y2 - y1) / 2) / h
    return cx, cy, bw, bh

for seq, action, img_path in tqdm(events, desc="Moving", unit="event"):
    if not os.path.exists(img_path):
        print("Missing image:", img_path)
        continue

    # Determine output filenames
    out_img_name = f"{seq}_{action['name']}.png"
    out_label_name = f"{seq}_{action['name']}.txt"

    # Move/copy image file (FAST)
    shutil.copy2(img_path, f"data_yolo/images/{out_img_name}")

    # Get image size WITHOUT PIL (FAST)
    from PIL import Image
    with Image.open(img_path) as img:
        w, h = img.size

    # Write YOLO label
    cx, cy, bw, bh = convert_to_yolo(
        action["x1"], action["y1"],
        action["x2"], action["y2"],
        w, h
    )

    with open(f"data_yolo/labels/{out_label_name}", "w") as f:
        f.write(f"0 {cx} {cy} {bw} {bh}\n")

print("Done.")


# -------------------------------------------------------
# File: yolo/data.py
# -------------------------------------------------------
import os

def parse(base_path: str):
    entries = []
    coords_file = os.path.join(base_path, "coordinates.txt")

    with open(coords_file, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            time_str, rest = line.split(":", 1)
            name = time_str.replace('.', '')

            if name[0] == "0":
                name = name[1:]

            parts = [p.strip() for p in rest.split(",")]

            x1 = float(parts[0])
            y1 = float(parts[1])
            x2 = float(parts[2])
            y2 = float(parts[3])
            label = int(parts[4])
            drone_name = parts[5]

            entries.append({
                "name": name,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "label": label,
                "drone_name": drone_name
            })

    return entries


def path(seq, frame):
    return f"Video_{seq}_frame_{frame}.png"


# -------------------------------------------------------
# File: yolo/make_vid.py
# -------------------------------------------------------
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


# -------------------------------------------------------
# File: yolo/render_events.py
# -------------------------------------------------------
import argparse
import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    idx = time_order[win_start:win_stop]
    words = event_words[idx].astype(np.uint32, copy=False)

    x = (words & 0x3FFF).astype(np.int32, copy=False)
    y = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pol = ((words >> 28) & 0xF) > 0

    return x, y, pol


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (52, 37, 30),     # #1e2534
    on_color: tuple[int, int, int] = (255, 255, 255),    # #ffffff
    off_color: tuple[int, int, int] = (172, 109, 57),    # #396dac
) -> np.ndarray:

    x, y, pol = window

    frame = np.empty((height, width, 3), np.uint8)
    frame[:] = base_color

    frame[y[pol], x[pol]] = on_color
    frame[y[~pol], x[~pol]] = off_color

    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--window", type=float, default=10,
                        help="Window duration in ms")
    parser.add_argument("--speed", type=float, default=1,
                        help="Playback speed")
    parser.add_argument("--force-speed", action="store_true",
                        help="Force playback speed by dropping windows")
    parser.add_argument("--out", default="output.mp4",
                        help="Output video filename")
    args = parser.parse_args()

    width, height = 1280, 720

    src = DatFileSource(
        args.dat, width=width, height=height,
        window_length_us=args.window * 1000
    )

    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, 60, (width, height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed (FFMPEG missing?)")

    for batch in pacer.pace(src.ranges()):

        win = get_window(
            src.event_words,
            src.order,
            batch.start,
            batch.stop,
        )

        frame = get_frame(win, width=width, height=height)

        out.write(frame)

    out.release()


if __name__ == "__main__":
    main()


# -------------------------------------------------------
# File: yolo/yolo_test.py
# -------------------------------------------------------
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best(2).pt")

# Path to your image
img_path = "data_yolo/images/63_111532218.png"

# Run inference
results = model(img_path)

# Load image with OpenCV
img = cv2.imread(img_path)

# Iterate through detections
for r in results:
    for box in r.boxes:
        # Coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Confidence and class
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

# Display result
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()`,
    source: "https://github.com/takara-ai/DroneStalker",
  },
  positionPrediction: {
    code: `from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DroneStalkerModel(nn.Module):
    """
    Unified CNN + Transformer model for drone trajectory prediction.

    Architecture:
    1. CNN Event Feature Encoder: 3-layer CNN processes 64x64 event images -> 128-dim features
    2. Feature Fusion: Kinematic (4-dim) + Event (128-dim) + Positional Encoding (132-dim) -> 264-dim
    3. Transformer Encoder: 4 layers, 4 heads, processes historical sequence
    4. Transformer Decoder: 4 layers, 4 heads, predicts future trajectory
    5. Prediction Head: Maps decoder output to bounding box coordinates

    Input:
        - event_images: [batch, Np, 1, 64, 64] - Sequence of event frames
        - kinematic_features: [batch, Np, 4] - Position and velocity features (z-score normalized)

    Output:
        - predictions: [batch, Nf, 4] - Future bounding boxes (x1, y1, x2, y2) for 12 frames
    """
    def __init__(self, Np=12, Nf=12, input_dim=132, num_layers=4, num_heads=4, dim_feedforward=512, dropout=0.1):
        super(DroneStalkerModel, self).__init__()

        self.Np = Np  # Observation window (12 frames = 0.4s at 30fps)
        self.Nf = Nf  # Prediction horizon (12 frames = 0.4s at 30fps)
        self.input_dim = input_dim  # 132 = 4 kinematic + 128 event features

        # === CNN Event Feature Encoder (Step 2) ===
        # Layer 1: (1, 64, 64) -> (32, 32, 32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: (32, 32, 32) -> (64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: (64, 16, 16) -> (128, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layer: Flatten to 128-dim event feature vector
        self.fc_event = nn.Linear(128 * 8 * 8, 128)

        # === Transformer Encoder (Step 4) ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Transformer Decoder (Step 4) ===
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # === Learnable Query Tokens ===
        # Past reconstruction queries (Np frames)
        self.past_query = nn.Parameter(torch.randn(Np, input_dim))
        # Future prediction queries (Nf frames)
        self.future_query = nn.Parameter(torch.randn(Nf, input_dim))

        # === Prediction Head ===
        # Maps 132-dim decoder output to 4-dim bounding box (x1, y1, x2, y2)
        self.prediction_head = nn.Linear(input_dim, 4)

        # === Pre-compute Positional Encoding ===
        # Buffer (not a parameter, but part of model state)
        self.register_buffer('positional_encoding',
                           create_sinusoidal_positional_encoding(Np, input_dim))

    def forward(self, event_images, kinematic_features):
        """
        Forward pass: CNN feature extraction + Feature fusion + Transformer prediction + Reconstruction.

        Args:
            event_images (torch.Tensor): [batch, Np, 1, 64, 64] - Event frame sequence
            kinematic_features (torch.Tensor): [batch, Np, 4] - Normalized kinematic features

        Returns:
            tuple: (reconstructed_past, predicted_future)
                - reconstructed_past: [batch, Np, 4] - Reconstructed bounding boxes for observation window
                - predicted_future: [batch, Nf, 4] - Predicted bounding boxes for future frames
        """
        batch_size, Np, _, _, _ = event_images.shape

        # === Step 1: CNN Event Feature Extraction ===
        # Reshape: [batch, Np, 1, 64, 64] -> [batch*Np, 1, 64, 64]
        images_flat = event_images.view(batch_size * Np, 1, 64, 64)

        # Conv layer 1
        x = self.pool1(F.relu(self.conv1(images_flat)))

        # Conv layer 2
        x = self.pool2(F.relu(self.conv2(x)))

        # Conv layer 3
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten and project to 128-dim event features
        x = x.view(-1, 128 * 8 * 8)
        event_features = self.fc_event(x)  # [batch*Np, 128]

        # Reshape back: [batch*Np, 128] -> [batch, Np, 128]
        event_features = event_features.view(batch_size, Np, 128)

        # === Step 2: Feature Fusion ===
        # Concatenate kinematic + event features: [batch, Np, 4] + [batch, Np, 128] -> [batch, Np, 132]
        fused_features = torch.cat([kinematic_features, event_features], dim=-1)

        # === Step 3: Add Positional Encoding ===
        # positional_encoding: [Np, 132] -> broadcast to [batch, Np, 132]
        pos_enc = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        input_sequence_encoded = fused_features + pos_enc  # [batch, Np, 132]

        # === Step 4: Transformer Encoding ===
        # Process historical sequence through encoder
        encoder_output = self.transformer_encoder(input_sequence_encoded)  # [batch, Np, 132]

        # === Step 5a: Reconstruction (Decode Past) ===
        # Broadcast past query tokens across batch dimension
        past_queries = self.past_query.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, Np, 132]

        # Decoder reconstructs observation window using encoder memory
        decoder_output_past = self.transformer_decoder(past_queries, encoder_output)  # [batch, Np, 132]

        # Project to 4-dim bounding box coordinates
        reconstructed_past = self.prediction_head(decoder_output_past)  # [batch, Np, 4]

        # === Step 5b: Prediction (Decode Future) ===
        # Broadcast future query tokens across batch dimension
        future_queries = self.future_query.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, Nf, 132]

        # Decoder predicts future sequence using encoder memory
        decoder_output_future = self.transformer_decoder(future_queries, encoder_output)  # [batch, Nf, 132]

        # Project to 4-dim bounding box coordinates
        predicted_future = self.prediction_head(decoder_output_future)  # [batch, Nf, 4]

        return reconstructed_past, predicted_future


class FREDLoss(nn.Module):
    """
    Custom loss function for FRED trajectory forecasting.

    L = L_Nf + λ * L_Np

    Where:
    - L_Nf: Forecasting loss (L2 distance for future predictions)
    - L_Np: Reconstruction loss (L2 distance for past reconstruction)
    - λ: Scaling coefficient (0.5 in FRED benchmark)
    """
    def __init__(self, lambda_recon=0.5):
        super(FREDLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.mse_loss = nn.MSELoss()

    def forward(self, reconstructed_past, predicted_future, ground_truth_past, ground_truth_future):
        """
        Compute the total FRED loss.

        Args:
            reconstructed_past: [batch, Np, 4] - Model's reconstruction of observation window
            predicted_future: [batch, Nf, 4] - Model's prediction of future frames
            ground_truth_past: [batch, Np, 4] - Ground truth bounding boxes for observation window
            ground_truth_future: [batch, Nf, 4] - Ground truth bounding boxes for future frames

        Returns:
            tuple: (total_loss, forecasting_loss, reconstruction_loss)
        """
        # L_Nf: Forecasting loss (L2 distance / RMSE)
        # MSE = mean squared error, sqrt(MSE) = RMSE = L2 distance
        forecasting_loss = torch.sqrt(self.mse_loss(predicted_future, ground_truth_future))

        # L_Np: Reconstruction loss (L2 distance / RMSE)
        reconstruction_loss = torch.sqrt(self.mse_loss(reconstructed_past, ground_truth_past))

        # Total loss: L = L_Nf + λ * L_Np
        total_loss = forecasting_loss + self.lambda_recon * reconstruction_loss

        return total_loss, forecasting_loss, reconstruction_loss


def get_data() -> list[dict]:
    # Load dataset
    raw_data = []
    for x in range(11):
        with open(f"../../data/{x}/coordinates.txt", "r") as coord_file:
            lines = coord_file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                time_part, coords_part = line.split(":", 1)
            except ValueError:
                continue
            time_str = time_part.strip()
            coord_values = [c.strip() for c in coords_part.split(",")]
            if len(coord_values) != 4:
                continue
            try:
                time_val = float(time_str)
                x1, y1, x2, y2 = [float(c) for c in coord_values]
            except ValueError:
                continue
            frame_time_str = str(time_val).replace(".", "")
            img_path = f"../../data/{x}/Event/Frames/Video_{x}_frame_{frame_time_str}.png"
            sample = {
                "time": float(time_val),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "img_path": img_path,
            }
            raw_data.append(sample)
        print(f"Loaded {len(raw_data)} samples from {x}")
    return raw_data

# Get the centre coordinates of the bounding box for one sample
def get_center_coordinates(sample: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = sample["x1"], sample["y1"], sample["x2"], sample["y2"]
    xc = (x1 + x2) / 2 # Center x coordinate of bounding box
    yc = (y1 + y2) / 2 # Center y coordinate of bounding box
    return (xc, yc)
    
# Get the velocity for two samples
def get_kinematic_features(sample1: dict, sample2: dict, meanv: tuple[float, float], stdv: tuple[float, float], mean_coordinates: tuple[float, float], std_coordinates: tuple[float, float]) -> tuple[float, float, float, float]:
    x1, y1 = get_center_coordinates(sample1)
    x2, y2 = get_center_coordinates(sample2)
    dx = x2 - x1
    dy = y2 - y1
    dt = max((sample2["time"] - sample1["time"]), 0.0001)
    vx = dx / dt # Velocity in pixels per second (px/s)
    vy = dy / dt # Velocity in pixels per second (px/s)

    # Perform Z-score standardisation
    x_zscore = (x2 - mean_coordinates[0]) / std_coordinates[0]
    y_zscore = (y2 - mean_coordinates[1]) / std_coordinates[1]
    vx_zscore = (vx - meanv[0]) / stdv[0]
    vy_zscore = (vy - meanv[1]) / stdv[1]
    return (x_zscore, y_zscore, vx_zscore, vy_zscore) # Return the x2, y2, vx and vy (z-score standardised)

def get_mean_velocity(data: list[dict]) -> tuple[float, float]:
    """Calculate mean velocity (raw, not z-scored) across the dataset."""
    total_vx = 0
    total_vy = 0
    for i in range(len(data) - 1):
        x1, y1 = get_center_coordinates(data[i])
        x2, y2 = get_center_coordinates(data[i + 1])
        dt = max((data[i + 1]["time"] - data[i]["time"]), 0.0001)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        total_vx += vx
        total_vy += vy
    mean_vx = total_vx / (len(data) - 1)
    mean_vy = total_vy / (len(data) - 1)
    return (mean_vx, mean_vy)

def get_std_velocity(data: list[dict], mean: tuple[float, float]) -> tuple[float, float]:
    """Calculate standard deviation of velocity (raw, not z-scored) across the dataset."""
    mean_vx, mean_vy = mean
    sum_sq_vx = 0
    sum_sq_vy = 0
    for i in range(len(data) - 1):
        x1, y1 = get_center_coordinates(data[i])
        x2, y2 = get_center_coordinates(data[i + 1])
        dt = max((data[i + 1]["time"] - data[i]["time"]), 0.0001)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        sum_sq_vx += (vx - mean_vx) ** 2
        sum_sq_vy += (vy - mean_vy) ** 2
    std_vx = math.sqrt(sum_sq_vx / (len(data) - 1))
    std_vy = math.sqrt(sum_sq_vy / (len(data) - 1))
    return (std_vx, std_vy)

def get_mean_coordinates(data: list[dict]) -> tuple[float, float]:
    total_x = 0
    total_y = 0
    for i in range(len(data)):
        x, y = get_center_coordinates(data[i])
        total_x += x
        total_y += y
    mean_x = total_x / len(data)
    mean_y = total_y / len(data)
    return (mean_x, mean_y)

def get_std_coordinates(data: list[dict], mean: tuple[float, float]) -> tuple[float, float]:
    """Calculate standard deviation of center coordinates across the dataset."""
    sum_sq_x = 0
    sum_sq_y = 0
    for i in range(len(data)):
        x, y = get_center_coordinates(data[i])
        sum_sq_x += (x - mean[0]) ** 2
        sum_sq_y += (y - mean[1]) ** 2
    std_x = math.sqrt(sum_sq_x / len(data))
    std_y = math.sqrt(sum_sq_y / len(data))
    return (std_x, std_y)

def get_event_feature(sample: dict) -> torch.Tensor:
    """
    Loads an image from sample["img_path"], crops to bounding box, 
    and rescales to 64x64 pixels, converting to grayscale (single channel).

    Returns:
        torch.Tensor: A 64x64 tensor (grayscale, values in [0,1]).
    """
    img_path = sample["img_path"]
    x1, y1, x2, y2 = sample["x1"], sample["y1"], sample["x2"], sample["y2"]
    try:
        img = Image.open(img_path)
        cropped_img = img.crop((x1, y1, x2, y2))
        resized_img = cropped_img.resize((64, 64), Image.LANCZOS)
        resized_img = resized_img.convert("L")
        arr = np.array(resized_img, dtype=np.uint8)  # shape: (64, 64)
        tensor = torch.from_numpy(arr).float().div(255.0)  # normalize to [0,1]
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None
    return tensor

def create_sinusoidal_positional_encoding(sequence_length: int, dimension: int) -> torch.Tensor:
    """
    Generates the sinusoidal positional encoding matrix P.
    
    Args:
        sequence_length: Np (e.g., 12) - The length of the observation window.
        dimension: D (e.g., 132) - The feature dimension of the fused input vector (I_t).

    Returns:
        P_matrix (torch.Tensor): The positional encoding matrix of shape (sequence_length, dimension).
    """
    # Initialize P matrix: Shape (Np, D) -> (12, 132)
    P_matrix = torch.zeros(sequence_length, dimension)
    
    # Create the position index tensor (t index: 0, 1, 2, ...)
    # Shape: (Np, 1)
    position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)

    # Create the dimension index tensor (i index: 0, 2, 4, ...)
    # Only even indices are calculated, as they determine the frequency for the pair (i, i+1)
    # i: indices for which we apply sin (0, 2, 4, ..., D-2)
    indices_i = torch.arange(0, dimension, 2, dtype=torch.float)

    # Calculate the denominator (angular frequency term): 1 / (10000^(2i/D))
    # Using log-exp stability trick for powers: exp(2i/D * -log(10000))
    # Shape: (D/2) -> (66)
    div_term = torch.exp(indices_i * (-math.log(10000.0) / dimension))

    # Apply Sine and Cosine functions to fill the matrix P
    # Even indices (0, 2, 4, ...) get sine: P(t, 2i) = sin(t / denominator)
    P_matrix[:, 0::2] = torch.sin(position * div_term)
    
    # Odd indices (1, 3, 5, ...) get cosine: P(t, 2i+1) = cos(t / denominator)
    P_matrix[:, 1::2] = torch.cos(position * div_term)
    
    return P_matrix


class DroneTrajectoryDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for FRED drone trajectory prediction with reconstruction.

    Creates sliding windows of Np+Nf consecutive frames.
    Returns:
    - Observation window (Np frames): event images + kinematic features + ground truth past boxes
    - Prediction target (Nf frames): ground truth future boxes
    """
    def __init__(self, data, Np=12, Nf=12):
        self.data = data
        self.Np = Np
        self.Nf = Nf
        self.sequence_length = Np + Nf

        # Compute normalization statistics
        print("Computing normalization statistics...")
        self.mean_velocity = get_mean_velocity(data)
        self.std_velocity = get_std_velocity(data, self.mean_velocity)
        self.mean_coordinates = get_mean_coordinates(data)
        self.std_coordinates = get_std_coordinates(data, self.mean_coordinates)

        # Find valid sequence indices
        self.valid_indices = []
        for i in range(len(data) - self.sequence_length + 1):
            is_valid = True
            for j in range(i, i + self.sequence_length - 1):
                time_diff = data[j + 1]["time"] - data[j]["time"]
                if time_diff > 0.1:  # 100ms gap indicates missing frames
                    is_valid = False
                    break
            if is_valid:
                self.valid_indices.append(i)

        print(f"Created {len(self.valid_indices)} valid sequences from {len(data)} frames")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Extract observation window (Np frames) and future frames (Nf frames)
        obs_frames = self.data[start_idx : start_idx + self.Np]
        future_frames = self.data[start_idx + self.Np : start_idx + self.sequence_length]

        # === Event Images ===
        event_images = []
        for sample in obs_frames:
            event_tensor = get_event_feature(sample)
            if event_tensor is None:
                event_tensor = torch.zeros(64, 64)
            event_images.append(event_tensor.unsqueeze(0))
        event_images = torch.stack(event_images)  # [Np, 1, 64, 64]

        # === Kinematic Features ===
        kinematic_features = []
        for i in range(len(obs_frames) - 1):
            features = get_kinematic_features(
                obs_frames[i], obs_frames[i + 1],
                self.mean_velocity, self.std_velocity,
                self.mean_coordinates, self.std_coordinates
            )
            kinematic_features.append(features)
        # Duplicate last velocity for final frame
        kinematic_features.append(kinematic_features[-1])
        kinematic_features = torch.tensor(kinematic_features, dtype=torch.float32)  # [Np, 4]

        # === Ground Truth Past (for reconstruction loss) ===
        ground_truth_past = []
        for sample in obs_frames:
            bbox = [sample["x1"], sample["y1"], sample["x2"], sample["y2"]]
            ground_truth_past.append(bbox)
        ground_truth_past = torch.tensor(ground_truth_past, dtype=torch.float32)  # [Np, 4]

        # === Ground Truth Future (for forecasting loss) ===
        ground_truth_future = []
        for sample in future_frames:
            bbox = [sample["x1"], sample["y1"], sample["x2"], sample["y2"]]
            ground_truth_future.append(bbox)
        ground_truth_future = torch.tensor(ground_truth_future, dtype=torch.float32)  # [Nf, 4]

        return event_images, kinematic_features, ground_truth_past, ground_truth_future


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda'):
    """
    Training loop for DroneStalkerModel with FRED loss (forecasting + reconstruction).
    """
    model = model.to(device)
    criterion = FREDLoss(lambda_recon=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        train_forecast_loss = 0.0
        train_recon_loss = 0.0

        for batch_idx, (event_images, kinematic_features, gt_past, gt_future) in enumerate(train_loader):
            event_images = event_images.to(device)
            kinematic_features = kinematic_features.to(device)
            gt_past = gt_past.to(device)
            gt_future = gt_future.to(device)

            # Forward pass
            reconstructed_past, predicted_future = model(event_images, kinematic_features)

            # Compute loss
            loss, forecast_loss, recon_loss = criterion(reconstructed_past, predicted_future, gt_past, gt_future)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_forecast_loss += forecast_loss.item()
            train_recon_loss += recon_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Forecast: {forecast_loss.item():.4f}, Recon: {recon_loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_forecast = train_forecast_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_forecast_loss = 0.0
        val_recon_loss = 0.0

        with torch.no_grad():
            for event_images, kinematic_features, gt_past, gt_future in val_loader:
                event_images = event_images.to(device)
                kinematic_features = kinematic_features.to(device)
                gt_past = gt_past.to(device)
                gt_future = gt_future.to(device)

                reconstructed_past, predicted_future = model(event_images, kinematic_features)
                loss, forecast_loss, recon_loss = criterion(reconstructed_past, predicted_future, gt_past, gt_future)

                val_loss += loss.item()
                val_forecast_loss += forecast_loss.item()
                val_recon_loss += recon_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_forecast = val_forecast_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train - Total: {avg_train_loss:.4f}, Forecast: {avg_train_forecast:.4f}, Recon: {avg_train_recon:.4f}")
        print(f"  Val   - Total: {avg_val_loss:.4f}, Forecast: {avg_val_forecast:.4f}, Recon: {avg_val_recon:.4f}")
        print("-" * 70)

        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
            print(f"Saved best model (val_loss: {avg_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth')


def load_model(checkpoint_path, Np=12, Nf=12, device='cuda'):
    """
    Load a trained DroneStalkerModel from checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        Np: Observation window size
        Nf: Prediction horizon
        device: 'cuda' or 'cpu'

    Returns:
        Loaded model in eval mode
    """
    model = DroneStalkerModel(Np=Np, Nf=Nf, input_dim=132, num_layers=4,
                             num_heads=4, dim_feedforward=512, dropout=0.1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    return model


def run_inference(model, observation_frames, mean_velocity, std_velocity, mean_coordinates, std_coordinates, device='cuda'):
    """
    Run inference on a sequence of observation frames.

    Args:
        model: Trained DroneStalkerModel
        observation_frames: List of Np consecutive frame dictionaries from FRED dataset
        mean_velocity, std_velocity, mean_coordinates, std_coordinates: Normalization stats from training
        device: 'cuda' or 'cpu'

    Returns:
        predicted_boxes: numpy array of shape [Nf, 4] containing predicted bounding boxes [x1, y1, x2, y2]
    """
    model.eval()
    Np = len(observation_frames)

    # === Process Event Images ===
    event_images = []
    for sample in observation_frames:
        event_tensor = get_event_feature(sample)
        if event_tensor is None:
            event_tensor = torch.zeros(64, 64)
        event_images.append(event_tensor.unsqueeze(0))
    event_images = torch.stack(event_images).unsqueeze(0)  # [1, Np, 1, 64, 64]

    # === Process Kinematic Features ===
    kinematic_features = []
    for i in range(len(observation_frames) - 1):
        features = get_kinematic_features(
            observation_frames[i], observation_frames[i + 1],
            mean_velocity, std_velocity, mean_coordinates, std_coordinates
        )
        kinematic_features.append(features)
    kinematic_features.append(kinematic_features[-1])  # Duplicate last velocity
    kinematic_features = torch.tensor([kinematic_features], dtype=torch.float32)  # [1, Np, 4]

    # Move to device
    event_images = event_images.to(device)
    kinematic_features = kinematic_features.to(device)

    # Run inference
    with torch.no_grad():
        reconstructed_past, predicted_future = model(event_images, kinematic_features)

    # Convert to numpy
    predicted_boxes = predicted_future.cpu().numpy()[0]  # [Nf, 4]

    return predicted_boxes


def visualize_predictions(observation_frames, predicted_boxes, save_path=None):
    """Visualize predicted trajectory."""
    import matplotlib.pyplot as plt

    # Observed trajectory
    obs_centers = [((f['x1'] + f['x2']) / 2, (f['y1'] + f['y2']) / 2) for f in observation_frames]
    obs_x, obs_y = zip(*obs_centers)

    # Predicted trajectory
    pred_centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in predicted_boxes]
    pred_x, pred_y = zip(*pred_centers)

    plt.figure(figsize=(10, 8))
    plt.plot(obs_x, obs_y, 'bo-', label='Observed', linewidth=2, markersize=8)
    plt.plot(pred_x, pred_y, 'ro-', label='Predicted', linewidth=2, markersize=8)
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Drone Trajectory Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def inference_example():
    """Example of running inference with a trained model."""
    checkpoint_path = 'best_model.pth'
    Np, Nf = 12, 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("DroneStalker Inference Example")
    print("=" * 70)

    # Load model
    model = load_model(checkpoint_path, Np=Np, Nf=Nf, device=device)

    # Load data and compute normalization stats
    print("\nLoading dataset...")
    data = get_data()
    mean_velocity = get_mean_velocity(data)
    std_velocity = get_std_velocity(data, mean_velocity)
    mean_coordinates = get_mean_coordinates(data)
    std_coordinates = get_std_coordinates(data, mean_coordinates)

    # Select random sequence
    import random
    start_idx = random.randint(0, len(data) - Np - Nf)
    observation_frames = data[start_idx : start_idx + Np]
    ground_truth_future = data[start_idx + Np : start_idx + Np + Nf]

    print(f"\nRunning inference on frames {start_idx} to {start_idx + Np - 1}")

    # Run inference
    predicted_boxes = run_inference(model, observation_frames, mean_velocity, std_velocity,
                                   mean_coordinates, std_coordinates, device)

    print("\nPredictions:")
    print("Frame | Predicted [x1, y1, x2, y2] | Ground Truth [x1, y1, x2, y2]")
    print("-" * 70)
    for i in range(Nf):
        pred = predicted_boxes[i]
        gt = ground_truth_future[i]
        print(f"{i+1:5d} | [{pred[0]:6.1f}, {pred[1]:6.1f}, {pred[2]:6.1f}, {pred[3]:6.1f}] | "
              f"[{gt['x1']:6.1f}, {gt['y1']:6.1f}, {gt['x2']:6.1f}, {gt['y2']:6.1f}]")

    # Visualize
    visualize_predictions(observation_frames, predicted_boxes, save_path='prediction_viz.png')

    print("\nInference completed!")


def main():
    
    """Main training script."""
    # Configuration
    Np = 12  # Observation window
    Nf = 12  # Prediction horizon
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("DroneStalker Training - FRED Dataset")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Observation window (Np): {Np} frames")
    print(f"Prediction horizon (Nf): {Nf} frames")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print("=" * 70)

    # Load data
    print("\nLoading FRED dataset...")
    data = get_data()

    # Train/Val split (80/20)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    print(f"Train: {len(train_data)} frames, Val: {len(val_data)} frames")

    # Create datasets
    train_dataset = DroneTrajectoryDataset(train_data, Np=Np, Nf=Nf)
    val_dataset = DroneTrajectoryDataset(val_data, Np=Np, Nf=Nf)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=(device == 'cuda')
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(device == 'cuda')
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing DroneStalkerModel...")
    model = DroneStalkerModel(Np=Np, Nf=Nf, input_dim=132, num_layers=4,
                             num_heads=4, dim_feedforward=512, dropout=0.1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

# Help

'''
    # Kinematic features usage

    data = get_data()
    meanv = get_mean_velocity(data)
    stdv = get_std_velocity(data, meanv)
    mean_coordinates = get_mean_coordinates(data)
    std_coordinates = get_std_coordinates(data, mean_coordinates)
    get_kinematic_features(data[0], data[1], meanv, stdv, mean_coordinates, std_coordinates)
'''

'''
Generating and Extracting P_t (Positional Encoding)

    # Constants based on the model:
    D = 132 # Feature dimension (4 kinematic + 128 CNN)
    Np = 12 # Sequence length (0.4s / 33ms)
    # Generate the full positional encoding matrix P for the entire sequence
    P_matrix = create_sinusoidal_positional_encoding(Np, D)
    # Define the time step index you are currently processing (e.g., the last observation)
    t_step = Np - 1 # Let's say we want the PE for the last observed timestep (index 11)

    # Extract the positional encoding vector P_t for the current timestep
    # P_t_raw is (132,)
    P_t_raw = P_matrix[t_step, :] 

    # Reshape P_t to be a 1x132 vector for element-wise addition (fusion)
    P_t = P_t_raw.unsqueeze(0) 
'''

'''
# Input for the transformer encoder decorder
    K_i + EF_i + P_t # Kinematics features, event features (CNN output) and positional encoding input fusion (this is fed to transformer encoder decoder)
'''`,
    source: "https://github.com/takara-ai/DroneStalker",
  },
} satisfies Record<string, { code: string; source: string }>;
