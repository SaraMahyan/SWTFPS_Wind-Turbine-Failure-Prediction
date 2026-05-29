#include <Arduino.h>
#include "DHT.h"
#include <WiFi.h>
#include <HTTPClient.h>

// --- WiFi & ThingSpeak Ayarları ---
const char* ssid = "REDMI 15C";
const char* password = "berdan2020M";
String apiKey = "5X9191IDIU1V3CNQ"; // ESP1'in Kendi Kanalı API Key

unsigned long lastTime = 0;
unsigned long timerDelay = 15000; // ThingSpeak 15 saniyede bir veri kabul eder
WiFiClient client;

// --- Sensör Pinleri ---
const int anemometerPin = 36;
const int potPin = 34;
const int acsPin = 35;
const int voltagePin = 32;
const int gasPin = 33;   // Analog input for Gas Sensor
const int linePin = 25;  // Digital input for Line Detector

// --- Kalibrasyon Sabitleri ---
const float R1 = 1000.0;
const float R2 = 330.0;
const float esp32_MaxVoltage = 3.3;
const int esp32_ADC_Resolution = 4095;
const float acs_Sensitivity = 0.185;
const float acs_ZeroPoint = 2.5;

// --- DHT Ayarları ---
#define DHTPIN1 4
#define DHTPIN2 5
#define DHTPIN3 18
#define DHTTYPE DHT11

const float minVoltage = 0.033;
const float maxVoltage = 3.3;
const float maxWindSpeed = 32.4;
const float mps_to_kmh = 3.6;

DHT dht1(DHTPIN1, DHTTYPE);
DHT dht2(DHTPIN2, DHTTYPE);
DHT dht3(DHTPIN3, DHTTYPE);


// EKSİK OLAN DEĞİŞKEN BURADA TANIMLANDI
// Interrupt içinde değişeceği için 'volatile' kullanmak zorunludur.
volatile unsigned int anemometerPulseCount = 0;

// EKSİK OLAN KESME (INTERRUPT) FONKSİYONU
// ESP32'de interrupt fonksiyonları RAM'de çalışması için IRAM_ATTR ile işaretlenir.
void IRAM_ATTR countPulse() {
  anemometerPulseCount++;
}

void setup() {
  Serial.begin(9600); 
  
  // ESP2 ile olan RX/TX Serial2 bağlantısı İPTAL EDİLDİ
  
  Serial.println("ESP #1 Sistem Baslatiliyor...");

  dht1.begin();
  dht2.begin();
  dht3.begin();

  pinMode(linePin, INPUT);
  pinMode(gasPin, INPUT); 
  
  // Anemometre pini giriş olarak ayarlandı ve Interrupt bağlandı
  pinMode(anemometerPin, INPUT);
  // Sensörden gelen sinyal YÜKSELEN (RISING) kenarda pals sayacını 1 artırır
  attachInterrupt(digitalPinToInterrupt(anemometerPin), countPulse, RISING);

  // WiFi Bağlantısı
  WiFi.begin(ssid, password);
  Serial.println("WiFi'ye baglaniliyor...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Baglandi!");
  Serial.print("IP Adresi: ");
  Serial.println(WiFi.localIP());

  // CSV Başlığı (Sadece ESP1'e ait sensörler) - DHT3 Eklendi
  Serial.println("timestamp_ms,wind_direction_deg,wind_speed_kmh,dht1_temp_c,dht2_temp_c,dht3_temp_c,turbine_voltage_v,turbine_current_a,turbine_power_w,gas_value,line_detected");
}

void loop() {

  // 1. Rüzgar Yönü (Potansiyometre)
  int potValue = analogRead(potPin);
  int windDirection = map(potValue, 0, 4095, 0, 360);

  // 2. Rüzgar Hızı (Anemometre - Pulse Sinyali)
  noInterrupts();
  unsigned int pulses = anemometerPulseCount;
  anemometerPulseCount = 0; // Sayacı sıfırla ki bir sonraki döngüde baştan saysın
  interrupts();
  
  // Loop döngümüz delay(1000) ile yaklaşık 1 saniyede bir çalıştığı için pulses doğrudan frekansı (Hz) verir.
  float frequency = (float)pulses;
  float windSpeed_mps = frequency * 0.83; // Kitapçıktaki formül: frequency * 0.83m/s
  float windSpeed_kmh = windSpeed_mps * 3.6;

  Serial.print("Anemometre Frekans (Hz): ");
  Serial.println(frequency);

  // 3. Sıcaklık (DHT1, DHT2 ve DHT3) - DHT3 Okuması Eklendi
  float t1 = dht1.readTemperature();
  float t2 = dht2.readTemperature();
  float t3 = dht3.readTemperature();
  
  if (isnan(t1)) t1 = 0;
  if (isnan(t2)) t2 = 0;
  if (isnan(t3)) t3 = 0;

  // 4. Türbin Voltajı
  int rawVoltADC = analogRead(voltagePin);
  float pinVoltage = (rawVoltADC / (float)esp32_ADC_Resolution) * esp32_MaxVoltage;
  float turbineVoltage = pinVoltage * ((R1 + R2) / R2);
  if (turbineVoltage < 0.2) {
    turbineVoltage = 0.0;
  }

  // 5. Türbin Akımı & Gücü
  int rawCurrentADC = analogRead(acsPin);
  float acsVoltage = (rawCurrentADC / (float)esp32_ADC_Resolution) * esp32_MaxVoltage;
  float current_Amps = (acsVoltage - acs_ZeroPoint) / acs_Sensitivity;
  current_Amps = abs(current_Amps);
  if (current_Amps < 0.08) {
    current_Amps = 0.0;
  }
  float power_Watts = turbineVoltage * current_Amps;

  // 6. Gaz ve Çizgi Sensörleri
  int gasValue = analogRead(gasPin);      
  int lineStatus = digitalRead(linePin);  

  // 7. CSV Olarak Seri Port'a Yazdır - DHT3 Verisi Eklendi
  Serial.print("CSV,");
  Serial.print(millis());
  Serial.print(',');
  Serial.print(windDirection);
  Serial.print(',');
  Serial.print(windSpeed_kmh, 2);
  Serial.print(',');
  Serial.print(t1, 2);
  Serial.print(',');
  Serial.print(t2, 2);
  Serial.print(',');
  Serial.print(t3, 2);
  Serial.print(',');
  Serial.print(turbineVoltage, 2);
  Serial.print(',');
  Serial.print(current_Amps, 2);
  Serial.print(',');
  Serial.print(power_Watts, 2);
  Serial.print(',');
  Serial.print(gasValue);
  Serial.print(',');
  Serial.println(lineStatus); 

  // 8. THINGSPEAK'E VERİ GÖNDERİMİ (15 saniyede bir)
  if ((millis() - lastTime) > timerDelay) {
    if(WiFi.status() == WL_CONNECTED){
      HTTPClient http;
      
      // ThingSpeak Field Atamaları - field8 olarak t3 (DHT3) Eklendi!
      String serverPath = "http://api.thingspeak.com/update?api_key=" + apiKey +
                          "&field1=" + String(turbineVoltage, 2) +
                          "&field2=" + String(power_Watts, 2) +
                          "&field3=" + String(current_Amps, 2) +
                          "&field4=" + String(windSpeed_kmh, 2) +
                          "&field5=" + String(gasValue) +
                          "&field6=" + String(lineStatus) +
                          "&field7=" + String(windDirection) +
                          "&field8=" + String(t3, 2);
      
      http.begin(client, serverPath.c_str());
      int httpResponseCode = http.GET();
      
      if (httpResponseCode > 0) {
        Serial.print("ESP1 ThingSpeak Basarili! HTTP Kodu: ");
        Serial.println(httpResponseCode);
      } else {
        Serial.print("ESP1 ThingSpeak Hatasi! HTTP Kodu: ");
        Serial.println(httpResponseCode);
      }
      http.end();
    } else {
      Serial.println("WiFi Baglantisi Koptu!");
    }
    lastTime = millis();
  }

  delay(1000);
}