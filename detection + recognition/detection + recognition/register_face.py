import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os

# Load Pretrained ResNet Model (Feature Extractor)
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  
resnet.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


if not os.path.exists("faces"):
    os.makedirs("faces")


cap = cv2.VideoCapture(0)

name = input("Enter name: ")  # Get user name
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_tensor = transform(face).unsqueeze(0)

        with torch.no_grad():
            embedding = resnet(face_tensor).numpy()  # Extract features

        # Save embedding
        np.save(f"faces/{name}.npy", embedding)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Registered", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Face data for {name} saved!")
