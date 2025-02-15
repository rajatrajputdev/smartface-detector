import cv2
import numpy as np

# Load YOLO model for face detection
net = cv2.dnn.readNet("yolov3-wider_16000.weights", "yolov3-face.cfg")

cap = cv2.VideoCapture(0)  # Open webcam
conf_threshold = 0.5
nms_threshold = 0.4

layer_names = net.getUnconnectedOutLayersNames()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert frame into a format YOLO can process
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward(layer_names)

    boxes, confidences = [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            confidence = max(scores)
            if confidence > conf_threshold:
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply non-maximum suppression to remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, f"{confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Face Detection", frame)
    
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
