//this is the control program of the servos attached to 4 fingers.
//designed by n015.

#include <Servo.h>
#include<math.h>

Servo servos[4];
//these should be current angles of the servos
int angles[4] = {0,0,0,0};
//maximum & minimum angles to ensure safety
int maximumAngle = 180;
int minimumAngle = 0;

void setup(){

//serial communication
  Serial.begin(9600);

//to attach motors to the port
  for(int i = 0; i < 4; i++){
    servos[i].attach(3+i);
  }   

//initaialize servos' angles(0 deg)
  for(int i = 0; i < 4; i++){
    servos[i].write(angles[i]);
  }

//status report
  Serial.println('initialization complete');
}

void loop(){
  if(Serial.available()){
    String input = Serial.readStringUntil('\n');
    input.trim();

//read
    int sep = input.indexOf(':');
    if(sep = 1){
      int index = input.substring(0, sep).toInt();
      int angle = input.substring(sep + 1).toInt();
//report input situation
      Serial.println('index is ');
      Serial.println(index);
      Serial.println('angle is ');
      Serial.println(angle);
//check if servos can do it & execute
      if(index >= 0 && index < 4 && angle >= minimumAngle && angle <= maximumAngle){
        smoothMoveTo(servos[index],angles[index],angle);
      }
      else{
        Serial.println('cannot do it');
      }
    }
    else{
      Serial.println('Invalid input');
    }
  }
} 
//this is the function to make the move smooth
void smoothMoveTo(Servo& servo, int& currentAngle, int targetAngle) {
  int steps = 50;
  int duration = 600;
  float t_delay = (float)duration / steps;

  float start = currentAngle;
  float delta = targetAngle - start;

  for (int i = 0; i <= steps; i++) {
    float t = (float)i / steps;
    float eased = 0.5 - 0.5 * cos(PI * t); 
    int angle = round(start + delta * eased);
    servo.write(angle);
    delay(t_delay);
  }

  currentAngle = targetAngle; 
  Serial.print("Moved to: ");
  Serial.println(targetAngle);
}