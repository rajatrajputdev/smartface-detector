bool status = false;  // Variable to track LED state

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
    
    // Properly update both LEDs
    digitalWrite(13, status ? HIGH : LOW);
    digitalWrite(12, status ? LOW : HIGH);
}
