# code 1 :-
# from ultralytics import YOLO
# import cv2
# import torch

# # Load YOLOv8 segmentation model
# model = YOLO("yolov8s-seg.pt")

# # Open webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret: 
#         break

#     # Run YOLOv8 segmentation on frame
#     results = model.predict(source=frame, conf=0.5, stream=True)

#     for r in results:
#         # Draw segmentation mask
#         annotated_frame = r.plot()

#         # Show frame
#         cv2.imshow("YOLOv8 Segmentation", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()



# code 2:-

# from ultralytics import YOLO
# import cv2
# import torch

# # Load your trained YOLOv8 segmentation model
# model = YOLO("runs/segment/train8/weights/best.pt")  # Path to trained model

# # Open webcam (or you can use a video file)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLOv8 segmentation on the frame
#     results = model.predict(source=frame, conf=0.5, stream=True)

#     for r in results:
#         # Draw segmentation mask
#         annotated_frame = r.plot()

#         # Display the output
#         cv2.imshow("YOLOv8 Segmentation", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()



# code 3 (combined both model):- 

import cv2
import numpy as np
from ultralytics import YOLO
import time
# === voice===
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 160)  # adjust speed if needed

# Try to set Hindi voice (optional)
for voice in engine.getProperty('voices'):
    if "hindi" in voice.name.lower() or "heera" in voice.name.lower() or "kalpana" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    engine.say(text)
    engine.runAndWait()


# Load both models
model_path = YOLO("runs/segment/train8/weights/best.pt")       # Trained segmentation model for walkable path -Done
model_obj = YOLO("yolov8s-seg.pt")           # Object detection model - done

# Analyze direction from path mask
def get_direction_label(mask, frame_width, threshold=30):
    h, w = mask.shape
    bottom_mask = mask[int(h * 0.6):, :]
    moments = cv2.moments(bottom_mask, binaryImage=True)

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    center_x = frame_width // 2
    dx = cx - center_x

    if abs(dx) < threshold:
        return "Go straight"
    elif dx < -threshold * 2:
        return "Sharp left"
    elif dx < -threshold:
        return "Slight left"
    elif dx > threshold * 2:
        return "Sharp right"
    else:
        return "Slight right"

# Start webcam
cap = cv2.VideoCapture(0)
last_direction = None
last_detected_objects = set()
last_obj_update_time = 0

print("🟢 Combined Path + Object Detection Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error")
        break

    frame_height, frame_width = frame.shape[:2]

    # 1. Run path segmentation
    results_path = model_path.predict(source=frame, verbose=False)
    mask = None

    for result in results_path:
        if result.masks is not None:
            mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (frame_width, frame_height))
            break

    # 2. Run object detection
    results_obj = model_obj.predict(source=frame, verbose=False)
    detected_objects = []

    for result in results_obj:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[cls_id]

            # Filter only relevant objects
            if confidence > 0.5 and label in ["person", "car", "bicycle", "motorcycle"]:
                detected_objects.append(label)

    # Display current direction
    direction = get_direction_label(mask, frame_width) if mask is not None else None
    if direction and direction != last_direction:
        print("🧭 Direction:", direction)
        last_direction = direction

    # Display detected objects (every 2s to avoid clutter)

    # current_time = time.time()
    # if detected_objects and current_time - last_obj_update_time > 2:
    #     unique_objects = set(detected_objects)
    #     if unique_objects != last_detected_objects:
    #         print("⚠️ Detected objects ahead:", ", ".join(unique_objects))
    #         last_detected_objects = unique_objects
    #         last_obj_update_time = current_time
        # Object detection
    results_obj = model_obj.predict(source=frame, verbose=False)
    detected_objects_with_direction = []

    for result in results_obj:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[cls_id]

        # Only alert for these objects
        if confidence > 0.5 and label in ["person", "car", "bicycle", "motorcycle"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2  # center x of box

            # Determine direction
            if cx < frame_width // 3:
                position = "left"
            elif cx > 2 * frame_width // 3:
                position = "right"
            else:
                position = "ahead"

            detected_objects_with_direction.append(f"{label} on the {position}")
    current_time = time.time()
    if detected_objects_with_direction and current_time - last_obj_update_time > 2:
        unique_msgs = list(set(detected_objects_with_direction))
        print("⚠️", " | ".join(unique_msgs))
        last_obj_update_time = current_time
    

    # Visual overlay
    frame_display = frame.copy()

    if mask is not None:
        color_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
        frame_display = cv2.addWeighted(frame_display, 0.7, color_mask, 0.3, 0)

    for result in results_obj:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            if label not in ["person", "car", "bicycle", "motorcycle"]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Path + Object Detection", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
