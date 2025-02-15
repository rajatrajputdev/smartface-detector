
# Face Detector using YOLO & Arduino

A Python-based face detection system using OpenCV and the YOLO algorithm, integrated with Arduino for signal transmission.
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