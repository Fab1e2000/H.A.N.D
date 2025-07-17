#include<Servo.h>
Servo servo;
const int minAngle = 0;
const int maxAngle = 160;

const int fsrPin = A0;
const int fsrPressure = 700;

int currentAngle = 0;
bool locked = false;

void setup() {
  Serial.begin(9600);
  servo.attach(3);
  servo.write(0);
  //Serial.println("initiated");
}

void loop(){
  if(Serial.available()){
    int pressure = analogRead(fsrPin);
    //Serial.print("pressure is ");
    //Serial.println(pressure);

    String input = Serial.readStringUntil('\n');
    int targetAngle = input.toInt();
    //Serial.print("target angle is ");
    //Serial.println(targetAngle);

    if(pressure <= fsrPressure){
    //Serial.println("safe, moving to target angle");
    servo.write(targetAngle);
    currentAngle = targetAngle;
    }
      else{
     if(targetAngle < currentAngle){
        //Serial.println("release command");
        servo.write(targetAngle);
        currentAngle = targetAngle;
      }
      else{
        //Serial.println("danger, reject command");
        currentAngle = currentAngle;
      }
    }
  }
  delay(100);
}
