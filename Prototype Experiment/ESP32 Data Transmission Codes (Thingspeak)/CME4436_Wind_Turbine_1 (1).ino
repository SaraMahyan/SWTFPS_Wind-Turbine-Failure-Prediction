#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// PINS
const int MIC_PIN = 36;          // Analog pin for microphone
const int GAS_SMOKE_PIN = 35;    // Analog input pin
const int POT_PIN = 34;          // ADC pin for potantiometer
const int LINE_TRACKER_PIN = 25; // Digital pin for line tracker
const int CURRENT_PIN = 39;      // ADC pin of current


// Thresholds
const int soundTreshold = 50;
const int smokeTreshold = 150;
const int potTreshold = 0;

// Global Variables
int lastWindDirection = -1;
float sensitivity = 0.185; // ACS712 model: 5A = 0.185 | 20A = 0.100 | 30A = 0.066 (V per A)
float offset = 2.5;         // Sensor output at 0A (2.5V for ACS712)

void setup() {
  Serial.begin(9600);
  pinMode(LINE_TRACKER_PIN, INPUT);

  delay(500);
}

void loop() {
  // Microphone Sensor
  int micValue = analogRead(MIC_PIN);

  if (micValue > soundTreshold) {
    Serial.print("Sound level: ");
    Serial.println(micValue);
  }

  // Gas and Smoke Sensor
  int gasValue = analogRead(GAS_SMOKE_PIN);

  if (gasValue > smokeTreshold) {
    Serial.print("Gas/Smoke Level: ");
    Serial.println(gasValue);
  }

  // Potantiometer for Wind Direction
  int potValue = analogRead(POT_PIN);
  float windDirection = map(potValue, 0, 4095, 0, 360); // Convert raw value to degrees (0°–360°)

  if (windDirection != lastWindDirection) {
    if (potValue > potTreshold) {
      Serial.print("Raw Value: ");
      Serial.print(potValue);
      Serial.print("  -> Wind Direction: ");
      Serial.print(windDirection);
      Serial.println("°");
    }
  }

  lastWindDirection = windDirection;

  // Line Tracker Sensor
  int lineTrackerValue = digitalRead(LINE_TRACKER_PIN);

  if (lineTrackerValue == HIGH) {  
    Serial.println("LINE DETECTED"); // Percepts colors which are different from WHITE like BLACK
  } else {
    Serial.println("NO LINE"); // Percepts WHITE
  }
}

// Converts Degree into Direction (North - South - East - West)
String getDirectionName(int deg) {
  if (deg >= 337 || deg < 23)  return "N";
  if (deg >= 23  && deg < 68)  return "NE";
  if (deg >= 68  && deg < 113) return "E";
  if (deg >= 113 && deg < 158) return "SE";
  if (deg >= 158 && deg < 203) return "S";
  if (deg >= 203 && deg < 248) return "SW";
  if (deg >= 248 && deg < 293) return "W";
  if (deg >= 293 && deg < 337) return "NW";
  return "ERR";
}
