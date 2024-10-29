import random
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import threading

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open the file in read mode
with open(r"C:\Users\prath\Downloads\OneDrive\Desktop\yolov8counting-trackingvehicles\coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Set video capture to laptop webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Dictionary to store objects detected in the current frame
current_frame_objects = {}

def speak(text):
    """Function to handle text-to-speech output."""
    engine.say(text)
    engine.runAndWait()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    DP = detect_params[0].numpy()  # Convert tensor array to numpy

    # Track the objects in the current frame
    frame_objects = {}

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # Returns one box
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            if conf > 0.7:  # Only announce if confidence is > 0.7
                class_name = class_list[clsID]

                # Only announce if the object was not previously detected in the last frame
                if class_name not in current_frame_objects:
                    # Create a separate thread for speech to avoid blocking
                    text = f"{class_name} detected with {round(conf * 100)} percent confidence"
                    threading.Thread(target=speak, args=(text,)).start()

                # Track object in this frame to avoid repeated announcements
                frame_objects[class_name] = True

            # Draw bounding box and label
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                3,
            )
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                f"{class_list[clsID]} {round(conf * 100)}%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Update current_frame_objects for the next iteration
    current_frame_objects = frame_objects

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
