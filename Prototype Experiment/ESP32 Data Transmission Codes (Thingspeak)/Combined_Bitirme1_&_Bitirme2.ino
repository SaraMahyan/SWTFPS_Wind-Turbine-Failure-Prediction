#include <Arduino.h>
#include "DHT.h"
#include <WiFi.h>
#include <HTTPClient.h>

// =========================================================================
// KULLANICI AYARI: HANGİ ESP İÇİN KOD YÜKLÜYORSANIZ BURAYI DEĞİŞTİRİN!
// ESP1'e yüklerken burayı 1 yapın. ESP2'ye yüklerken burayı 2 yapın.
#define ESP_BOARD 1  
// =========================================================================

// --- WiFi Ayarları ---
const char* ssid = "REDMI 15C";
const char* password = "berdan2020M";

unsigned long lastTime = 0;
unsigned long timerDelay = 15000; // ThingSpeak 15 saniyede bir veri kabul eder
WiFiClient client;


#if ESP_BOARD == 1
  // *******************************************************************
  //                          ESP 1 KODLARI 
  // *******************************************************************
  String apiKey = "R5O98C69WRRAHIC2"; // ESP1'in Kendi Kanalı API Key (Kanal 1)
  
  // --- ESP1 Orijinal Sensör Pinleri ---
  const int potPin = 34;       
  const int acsPin = 35;       
  const int voltagePin = 32;   
  const int gasPin = 33;       
  const int linePin = 25;      
  const int anemometerPin = 36;
  
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
  
  DHT dht1(DHTPIN1, DHTTYPE);
  DHT dht2(DHTPIN2, DHTTYPE);
  DHT dht3(DHTPIN3, DHTTYPE);
  
  volatile unsigned int anemometerPulseCount = 0;
  void IRAM_ATTR countPulse() {
    anemometerPulseCount++;
  }

#elif ESP_BOARD == 2
  // *******************************************************************
  //                          ESP 2 KODLARI 
  // *******************************************************************
  String apiKey = "TJYJCHWZS9KHHHCU"; // ESP2'nin Kendi Kanalı API Key (Kanal 2)
  
  // --- ESP2 Orijinal Sensör Pinleri ---
  const int soundPin = 34; 
  const int vibPin = 35;   

#endif


void setup() {
  Serial.begin(9600); 
  
  Serial.print("Sistem Baslatiliyor... (Secili Kart: ESP ");
  Serial.print(ESP_BOARD);
  Serial.println(")");

  // --- Sadece Seçilen ESP'nin Pinlerini Hazırla ---
  #if ESP_BOARD == 1
    dht1.begin();
    dht2.begin();
    dht3.begin();
    pinMode(linePin, INPUT);
    pinMode(gasPin, INPUT); 
    pinMode(anemometerPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(anemometerPin), countPulse, RISING);
    Serial.println("timestamp_ms,wind_direction_deg,wind_speed_kmh,dht1_temp_c,dht2_temp_c,dht3_temp_c,turbine_voltage_v,turbine_current_a,turbine_power_w,gas_value,line_detected");
  
  #elif ESP_BOARD == 2
    pinMode(vibPin, INPUT);
    pinMode(soundPin, INPUT);
  #endif

  // WiFi Bağlantısı (İkisi için de ortak)
  WiFi.begin(ssid, password);
  Serial.println("WiFi'ye baglaniliyor...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Baglandi!");
  Serial.print("IP Adresi: ");
  Serial.println(WiFi.localIP());
}


void loop() {
  
  #if ESP_BOARD == 1
    // =================================================================
    //                     ESP 1 DÖNGÜSÜ (GÖREVLERİ)
    // =================================================================
    int potValue = analogRead(potPin);
    int windDirection = map(potValue, 0, 4095, 0, 360);
  
    noInterrupts();
    unsigned int pulses = anemometerPulseCount;
    anemometerPulseCount = 0; 
    interrupts();
    
    float frequency = (float)pulses;
    float windSpeed_mps = frequency * 0.83; 
    float windSpeed_kmh = windSpeed_mps * 3.6;
  
    Serial.print("Anemometre Frekans (Hz): ");
    Serial.println(frequency);
  
    float t1 = dht1.readTemperature();
    float t2 = dht2.readTemperature();
    float t3 = dht3.readTemperature();
    if (isnan(t1)) t1 = 0;
    if (isnan(t2)) t2 = 0;
    if (isnan(t3)) t3 = 0;
  
    int rawVoltADC = analogRead(voltagePin);
    float pinVoltage = (rawVoltADC / (float)esp32_ADC_Resolution) * esp32_MaxVoltage;
    float turbineVoltage = pinVoltage * ((R1 + R2) / R2);
    if (turbineVoltage < 0.2) turbineVoltage = 0.0;
  
    int rawCurrentADC = analogRead(acsPin);
    float acsVoltage = (rawCurrentADC / (float)esp32_ADC_Resolution) * esp32_MaxVoltage;
    float current_Amps = (acsVoltage - acs_ZeroPoint) / acs_Sensitivity;
    current_Amps = abs(current_Amps);
    if (current_Amps < 0.08) current_Amps = 0.0;
    float power_Watts = turbineVoltage * current_Amps;
  
    int gasValue = analogRead(gasPin);      
    int lineStatus = digitalRead(linePin);  
  
    // ESP1 CSV Çıktısı
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
  
    // ESP1 ThingSpeak Gönderimi (Kanal 1)
    if ((millis() - lastTime) > timerDelay) {
      if(WiFi.status() == WL_CONNECTED){
        HTTPClient http;
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


  #elif ESP_BOARD == 2
    // =================================================================
    //                     ESP 2 DÖNGÜSÜ (GÖREVLERİ)
    // =================================================================
    int soundVal = analogRead(soundPin);
    int vibVal = analogRead(vibPin);

    // ESP2 CSV Çıktısı
    Serial.print("CSV,");
    Serial.print(millis());
    Serial.print(",");
    Serial.print(soundVal);
    Serial.print(",");
    Serial.println(vibVal);
    
    // ESP2 ThingSpeak Gönderimi (Kanal 2)
    if ((millis() - lastTime) > timerDelay) {
      if(WiFi.status() == WL_CONNECTED){
        HTTPClient http;
        String serverPath = "http://api.thingspeak.com/update?api_key=" + apiKey +
                            "&field1=" + String(soundVal) +
                            "&field2=" + String(vibVal);
        
        http.begin(client, serverPath.c_str());
        int httpResponseCode = http.GET();
        if (httpResponseCode > 0) {
          Serial.print("ESP2 ThingSpeak Basarili! HTTP Kodu: ");
          Serial.println(httpResponseCode);
        } else {
          Serial.print("ESP2 ThingSpeak Hatasi! HTTP Kodu: ");
          Serial.println(httpResponseCode);
        }
        http.end();
      } else {
        Serial.println("WiFi Baglantisi Koptu!");
      }
      lastTime = millis();
    }

  #endif

  delay(1000);
}
