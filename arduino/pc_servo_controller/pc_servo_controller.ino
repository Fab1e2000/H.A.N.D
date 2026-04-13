#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <string.h>
#include <stdlib.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const uint8_t CHANNEL_MIN = 0;
const uint8_t CHANNEL_MAX = 15;
const int ANGLE_MIN = 0;
const int ANGLE_MAX = 180;

// Typical PCA9685 pulse range for 50/60Hz hobby servos; tune if needed.
const int SERVOMIN = 120;
const int SERVOMAX = 520;

const unsigned long SERIAL_BAUD = 115200;
const bool ACK_EVERY_COMMAND = false;
const size_t LINE_BUF_SIZE = 64;

char lineBuf[LINE_BUF_SIZE];
size_t lineLen = 0;

bool i2cDeviceExists(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

int angleToPulse(int angle) {
  angle = constrain(angle, ANGLE_MIN, ANGLE_MAX);
  int pulse = map(angle, ANGLE_MIN, ANGLE_MAX, SERVOMIN, SERVOMAX);
  return constrain(pulse, SERVOMIN, SERVOMAX);
}

void setServoAngle(uint8_t channel, int angle) {
  int pulse = angleToPulse(angle);
  pwm.setPWM(channel, 0, pulse);
}

void printHelp() {
  Serial.println("Protocol: S,<channel>,<angle>");
  Serial.println("Example : S,0,90");
}

void parseAndApply(const char* line) {
  if (line == nullptr || line[0] == '\0') {
    return;
  }

  if (!(line[0] == 'S' && line[1] == ',')) {
    Serial.println("ERR,FORMAT");
    return;
  }

  // Local copy for tokenization.
  char work[LINE_BUF_SIZE];
  strncpy(work, line, LINE_BUF_SIZE - 1);
  work[LINE_BUF_SIZE - 1] = '\0';

  char* tok = strtok(work, ",");
  if (tok == nullptr || strcmp(tok, "S") != 0) {
    Serial.println("ERR,PARSE");
    return;
  }

  char* chTok = strtok(nullptr, ",");
  char* angleTok = strtok(nullptr, ",");
  if (chTok == nullptr || angleTok == nullptr) {
    Serial.println("ERR,PARSE");
    return;
  }

  int channel = atoi(chTok);
  int angle = atoi(angleTok);

  if (channel < CHANNEL_MIN || channel > CHANNEL_MAX) {
    Serial.println("ERR,CHANNEL");
    return;
  }

  angle = constrain(angle, ANGLE_MIN, ANGLE_MAX);
  setServoAngle((uint8_t)channel, angle);

  if (ACK_EVERY_COMMAND) {
    Serial.print("OK,");
    Serial.print(channel);
    Serial.print(',');
    Serial.println(angle);
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin();

  if (!i2cDeviceExists(0x40)) {
    Serial.println("ERR,PCA9685_NOT_FOUND,0x40");
    while (true) {
      delay(1000);
    }
  }

  pwm.begin();
  pwm.setPWMFreq(60);
  delay(10);

  printHelp();
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        parseAndApply(lineBuf);
        lineLen = 0;
      }
      continue;
    }

    if (lineLen < LINE_BUF_SIZE - 1) {
      lineBuf[lineLen++] = c;
    } else {
      // Prevent unlimited growth if sender goes wrong.
      Serial.println("ERR,LINE_TOO_LONG");
      lineLen = 0;
    }
  }
}
