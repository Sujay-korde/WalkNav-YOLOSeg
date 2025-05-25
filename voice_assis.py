import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time

# Load your trained YOLOv8 segmentation model
model = YOLO("runs/segment/train8/weights/best.pt")  # change this to your model path - Changed

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # speed of speech

# Function to determine direction from path mask
def get_direction_label(mask, frame_width, threshold=30):
    h, w = mask.shape
    bottom_mask = mask[int(h * 0.6):, :]  # bottom 40% of the frame
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

# Function to speak message
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(0)
last_direction = None
last_spoken_time = 0

print("🔄 Starting camera stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    frame_height, frame_width = frame.shape[:2]

    # Run YOLOv8 segmentation
    results = model.predict(source=frame, task="segment", verbose=False)
    mask = None

    # Extract segmentation mask for the walkable path (assumes only 1 class)
    for result in results:
        if result.masks is not None:
            mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (frame_width, frame_height))
            break

    # Get direction label from mask
    direction = None
    if mask is not None:
        direction = get_direction_label(mask, frame_width)

    # Speak only if new direction is detected
    current_time = time.time()
    if direction and direction != last_direction and current_time - last_spoken_time > 1.5:
        print("🔊", direction)
        speak(direction)
        last_direction = direction
        last_spoken_time = current_time

    # For visual debugging
    if mask is not None:
        color_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
        cv2.imshow("Path Guidance", overlay)
    else:
        cv2.imshow("Path Guidance", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
