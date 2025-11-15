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