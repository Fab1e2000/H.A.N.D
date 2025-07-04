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
    int angle = Serial.parseInt();
    if(angle>=minAngle&&angle<=maxAngle){
      servo.write(angle);
      Serial.println("instant angle is:");
      Serial.println(angle);
      while (Serial.available()) Serial.read();//stop collapse
    }
  }
}
