import cv2
# Import necessary libraries
import numpy as np
import os
import face_recognition

def load_known_faces():
    encodings = []
    names = []
    known_dir = "known"

    for filename in os.listdir(known_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(f"{known_dir}/{filename}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"[+] Loaded: {name}")
            else:
                print(f"[!] No face found in {filename}")

        return encodings, names