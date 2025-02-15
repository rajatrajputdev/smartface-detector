
# Face Detector using YOLO & Arduino

A Python-based face detection system using OpenCV and the YOLOV3 Face, integrated with Arduino for signal transmission.
## Arduino Setup


#### Arduino Circuit Setup

![circuit_image](https://raw.githubusercontent.com/rajatrajputdev/smartface-detector/refs/heads/main/demonstration/circuit.png)

#### Arduino Code in C Language

```c
bool status = false;

void setup() {
    pinMode(13, OUTPUT); 
    pinMode(12, OUTPUT);
    Serial.begin(9600);
}

void loop() {
    if (Serial.available() > 0) {
        char received = Serial.read();
        if (received == '1') {
            status = true;
        } else if (received == '0') {
            status = false;
        }
    }
    
    digitalWrite(12, status ? HIGH : LOW);
    digitalWrite(13, status ? LOW : HIGH);
}
```
#### Arduino Circuit in Action when data is sent using the open stream
![circuit_demo](https://raw.githubusercontent.com/rajatrajputdev/smartface-detector/refs/heads/main/demonstration/circuit_simulation.gif)



## YOLO Face Detection  

This project uses **YOLOv3-Face** for real-time face detection, which was trained on the WIDER face dataset.   

#### 1. Download Model Weights 
[Click here to download the model weights](https://drive.google.com/file/d/1UzKtnIpGjyKlUN_oo8R5ppBTCPiya7N6/view?usp=sharing)

#### 2. Install Dependencies  
```bash
pip install opencv-python numpy
```

#### 3. Load YOLO & Webcam
 ```python
 import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3-wider_16000.weights", "yolov3-face.cfg")
cap = cv2.VideoCapture(0)
```

#### 4. Optimized Face Detection 
```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences = [], []
    for output in outputs:
        for detection in output:
            confidence = max(detection[5:])
            if confidence > 0.5:
                x, y, w, h = (detection[:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(int)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    for i in cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4).flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("YOLO Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
##
This system enables real-time face detection using YOLOv3, triggering the Arduino's LED when a face is detected. The integration ensures minimal lag and efficient performance, making it suitable for various applications.  

