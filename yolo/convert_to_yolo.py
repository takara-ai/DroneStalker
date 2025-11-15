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
