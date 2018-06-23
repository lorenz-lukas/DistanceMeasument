#include <Sparki.h>
#define NO_ACCEL // disables the Accelerometer, frees up 598 Bytes Flash Memory
#define NO_MAG // disables the Magnetometer, frees up 2500 Bytes Flash Memory
int distance = 0;
void setup() 
{
    Serial.begin(9600);
}

void loop()
{
    unsigned long sparki_time = millis();//miliseconds 
    distance = sparki.ping(); // measures the distance with Sparki's eyes
    
    sparki.clearLCD();
    sparki.println(sparki_time);
    sparki.println(distance); // tells the distance to the computer
    sparki.updateLCD();
    Serial.print(sparki_time);
    Serial.print(";");
    Serial.println(distance);
     
    if(distance != -1) // make sure its not too close or too far
    {
        if(distance < 7) // if the distance measured is less than 10 centimeters
        {
            sparki.beep(); // beep!
        }
    }
    if (Serial.available()) // Bluetooth conection
    {
        int inByte = Serial.read();
        int command = (char)inByte;
        if(command == 'f'){
            Serial.print((char)inByte); 
            sparki.moveForward(10); // 10cm               
        }
    }
} 

