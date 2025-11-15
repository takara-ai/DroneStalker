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
cv2.destroyAllWindows()
