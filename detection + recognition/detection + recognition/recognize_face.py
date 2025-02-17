import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
from scipy.spatial.distance import cosine
import serial
import time
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
time.sleep(2) 
# Load Pretrained ResNet Model (Feature Extractor)
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval()

# Define Preprocessing Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Saved Face Embeddings
face_data = {}
face_dir = "faces"
for file in os.listdir(face_dir):
    if file.endswith(".npy"):
        name = file.split(".")[0]
        face_data[name] = np.load(os.path.join(face_dir, file))

# Initialize OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        arduino.write(b'0')
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_tensor = transform(face).unsqueeze(0)

        with torch.no_grad():
            embedding = resnet(face_tensor).numpy()

        # Compare with Registered Faces
        best_match = None
        best_score = 1.0  # Lower is better (cosine distance)

        for name, saved_embedding in face_data.items():
            score = cosine(embedding.flatten(), saved_embedding.flatten())
            if score < best_score:
                best_score = score
                best_match = name

        if best_match and best_score < 0.1:  # Threshold for recognition
            label = f"{best_match} ({best_score:.2f})"
            color = (0, 255, 0)
            arduino.write(b'1')

        else:
            label = "Unknown"
            color = (0, 0, 255)
            arduino.write(b'0')

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
