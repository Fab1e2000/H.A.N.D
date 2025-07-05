#include<Servo.h>
Servo servo;
int minAngle = 0;
int maxAngle = 160;

void setup() {
  Serial.begin(9600);
  servo.attach(3);
  servo.write(0);
  Serial.println("servo initiated");
}

void loop() {
  if(Serial.available()){
    String input = Serial.readStringUntil('\n');
    int angle = input.toInt();
    if(angle>=minAngle&&angle<=maxAngle){
      servo.write(angle);
      Serial.println("instant angle is:");
      Serial.println(angle);
    }
  }
}

