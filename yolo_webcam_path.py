from ultralytics import YOLO
import cv2
import torch

# Load YOLOv8 segmentation model
model = YOLO("yolov8s-seg.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # Run YOLOv8 segmentation on frame
    results = model.predict(source=frame, conf=0.5, stream=True)

    for r in results:
        # Draw segmentation mask
        annotated_frame = r.plot()

        # Show frame
        cv2.imshow("YOLOv8 Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
