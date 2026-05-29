#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

// --- WiFi & ThingSpeak Ayarları ---
const char* ssid = "REDMI 15C";
const char* password = "berdan2020M";

// DİKKAT: BURAYA ESP2 İÇİN AÇTIĞINIZ YENİ THINGSPEAK KANALININ API KEY'İNİ GİRİNİZ!
String apiKey = "GOR59CQUP1SL4HSP"; 

unsigned long lastTime = 0;
unsigned long timerDelay = 15000; // 15 saniyede bir gönderim
WiFiClient client;

// --- Sensor Pins on ESP #2 ---
const int soundPin = 34; // Analog (Microphone) - 32 dolu olduğu için 34 seçildi
const int vibPin = 35;   // Analog (Vibration) - Analog okuma için ADC1 pini olan 35 seçildi

void setup() {
  Serial.begin(9600); 
  
  // ESP1 ile olan RX/TX Serial2 bağlantısı İPTAL EDİLDİ
  
  Serial.println("ESP #2 Sistem Baslatiliyor...");
  
  pinMode(vibPin, INPUT);
  pinMode(soundPin, INPUT);

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
}

void loop() {
  // 1. Sensörleri Oku
  int soundVal = analogRead(soundPin);
  int vibVal = analogRead(vibPin);

  // 2. Seri Port Ekranına Yazdır (Python'un okuması için CSV formatı eklendi)
  Serial.print("CSV,");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(soundVal);
  Serial.print(",");
  Serial.println(vibVal);

  // 3. THINGSPEAK'E VERİ GÖNDERİMİ (15 saniyede bir)
  if ((millis() - lastTime) > timerDelay) {
    if(WiFi.status() == WL_CONNECTED){
      HTTPClient http;
      
      // ThingSpeak Field Atamaları (ESP2 verileri 2. kanala gidiyor)
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

  delay(1000); 
}