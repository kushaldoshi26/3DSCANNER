// 3DINVENZA - ESP32 Turntable Firmware
// Protocol: Serial (115200 baud)
// Commands:
//   ROTATE <angle_deg>
//   STEP <steps>
//   SET_SPEED <steps_per_sec>
//   STOP
//   STATUS
//   CALIBRATE

#define STEP_PIN 14
#define DIR_PIN 27
#define EN_PIN 26

// Configuration
const int STEPS_PER_REV = 200; // Standard NEMA 17
const int MICROSTEPPING = 16;
const int TOTAL_STEPS = STEPS_PER_REV * MICROSTEPPING;

// State
int currentSpeed = 1000; // steps per second
float currentAngle = 0.0;
bool isMoving = false;

void setup() {
  Serial.begin(115200);
  
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  
  digitalWrite(EN_PIN, LOW); // Enable driver (active low)
  
  Serial.println("READY");
}

void doSteps(long steps, int dir) {
  isMoving = true;
  digitalWrite(DIR_PIN, dir);
  
  // Calculate delay based on speed
  long delayUs = 1000000 / currentSpeed;
  
  for(long i = 0; i < steps; i++) {
    // Check for incoming serial commands to allow interruption (STOP)
    if(Serial.available() > 0) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if(cmd == "STOP") {
        Serial.println("OK STOPPED");
        break;
      }
    }
    
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(delayUs / 2);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(delayUs / 2);
  }
  
  isMoving = false;
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.length() == 0) return;
    
    int spaceIdx = input.indexOf(' ');
    String cmd = input;
    String arg = "";
    
    if (spaceIdx > 0) {
      cmd = input.substring(0, spaceIdx);
      arg = input.substring(spaceIdx + 1);
    }
    
    cmd.toUpperCase();
    
    if (cmd == "STATUS") {
      Serial.print("OK ");
      Serial.print(isMoving ? "MOVING " : "IDLE ");
      Serial.println(currentAngle);
    } 
    else if (cmd == "SET_SPEED") {
      int s = arg.toInt();
      if (s > 0) {
        currentSpeed = s;
        Serial.println("OK");
      } else {
        Serial.println("ERROR Invalid speed");
      }
    }
    else if (cmd == "ROTATE") {
      float angle = arg.toFloat();
      long stepsToMove = abs((angle / 360.0) * TOTAL_STEPS);
      int dir = (angle >= 0) ? HIGH : LOW;
      
      Serial.println("OK MOVING");
      doSteps(stepsToMove, dir);
      
      currentAngle += angle;
      if(currentAngle >= 360.0) currentAngle -= 360.0;
      if(currentAngle < 0.0) currentAngle += 360.0;
      
      Serial.println("OK DONE");
    }
    else if (cmd == "STEP") {
      long steps = arg.toInt();
      int dir = (steps >= 0) ? HIGH : LOW;
      
      Serial.println("OK MOVING");
      doSteps(abs(steps), dir);
      Serial.println("OK DONE");
    }
    else if (cmd == "CALIBRATE") {
      currentAngle = 0.0;
      Serial.println("OK");
    }
    else if (cmd == "STOP") {
      // If we are here, we were already idle, but acknowledge anyway
      Serial.println("OK STOPPED");
    }
    else {
      Serial.println("ERROR Unknown command");
    }
  }
}
