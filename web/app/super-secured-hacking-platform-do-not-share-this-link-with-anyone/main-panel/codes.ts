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
    code: `import argparse  # noqa: INP001

# Placeholder motion detection code for @store.ts
def main():
    print("Motion detection logic placeholder")
    
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
    code: `import argparse  # noqa: INP001

# Placeholder position prediction code for @store.ts
def main():
    print("Position prediction logic placeholder")

if __name__ == "__main__":
    main()
`,
    source: "https://github.com/takara-ai/DroneStalker",
  },
} satisfies Record<string, { code: string; source: string }>;
