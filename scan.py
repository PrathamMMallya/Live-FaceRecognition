import cv2 as cv
import os
import pyttsx3
import time
from training import personFolders as pf

# Initialize text-to-speech engine
engine = pyttsx3.init()

camera = 0
video = cv.VideoCapture(camera, cv.CAP_DSHOW)
haarCascade = cv.CascadeClassifier(r"C:\Users\prath\Downloads\Live-FaceRecognition\haarcascade_frontalface_default.xml")
recognizer = cv.face.LBPHFaceRecognizer.create()
recognizer.read('training.xml')

# Check if video capture is successful
if not video.isOpened():
    print("Error: Could not open video capture.")
    exit()

detected_people = set()  # To keep track of already detected names

while True:
    ret, frame = video.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    greyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarCascade.detectMultiScale(greyscale, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize the face
        id, conf = recognizer.predict(greyscale[y:y + h, x:x + w])

        # Set a threshold for confidence
        if conf < 100:  # Adjust this threshold as needed
            label = pf[id - 1] if id - 1 < len(pf) else "Pratham Mallya"
            cv.putText(frame, label, (x + 10, y - 10), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

            # Speak when "Pratham Mallya" is detected
            if label == "Pratham Mallya" and label not in detected_people:
                engine.say("Hey your brother Pratham Mallya ahead")
                engine.runAndWait()
                detected_people.add(label)  # Add to detected people

                # Wait for 2 seconds before announcing Sreeharish
                time.sleep(2)

                # Now check for Sreeharish in the same frame
                if "Sreeharish" not in detected_people:
                    engine.say("Hi your cousin Sreeharish ahead")
                    engine.runAndWait()
                    detected_people.add("Sreeharish")  # Add to detected people

        else:
            cv.putText(frame, 'Sreeharish', (x + 10, y - 10), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    cv.imshow("Face Recognition", frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv.destroyAllWindows()
