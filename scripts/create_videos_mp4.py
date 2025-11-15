#!/usr/bin/env python3
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
