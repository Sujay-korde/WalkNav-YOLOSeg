import cv2
import torch
import pyttsx3
from ultralytics import YOLO

# Load models
model_seg = YOLO("runs/segment/train8/weights/best.pt")     # Your path segmentation model
model_obj = YOLO("yolov8s-seg.pt")              # Object detection model

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_direction(x_center, frame_width):
    third = frame_width // 3
    if x_center < third:
        return "left"
    elif x_center < 2 * third:
        return "center"
    else:
        return "right"

# Start webcam
cap = cv2.VideoCapture(0)

print("🟢 System running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_width = frame.shape[1]

    # Path detection
    results_seg = model_seg.predict(source=frame,  verbose=False)
    path_mask = results_seg[0].masks
    if path_mask is not None:
        mask = path_mask.data[0].cpu().numpy()
        color_mask = (mask * 255).astype("uint8")
        color_mask = cv2.merge([color_mask] * 3)
        frame = cv2.addWeighted(frame, 1, color_mask, 0.4, 0)

    # Object detection
    results_obj = model_obj.predict(source=frame, verbose=False)
    boxes = results_obj[0].boxes
    spoken = set()

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            name = model_obj.names[cls_id]
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2
            direction = get_direction(x_center, frame_width)

            label = f"{name} ({direction})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if name not in spoken:
                print(f"⚠️ Object alert: {name} ahead at your {direction}")
                speak(f"{name} ahead at your {direction}")
                spoken.add(name)

    # Show the result
    cv2.imshow("Path + Object Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
