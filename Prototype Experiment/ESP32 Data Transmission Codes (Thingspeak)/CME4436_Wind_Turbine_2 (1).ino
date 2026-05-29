#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
//#include "esp_eap_client.h" // YENİ KÜTÜPHANE (v3.0 için)
#include "DHT.h"

// ******** DEFINE SOFTWARE SERIAL *********** //
//#include <SoftwareSerial.h> // The SoftwareSerial library is included
#include <WiFi.h>
#include <HTTPClient.h>


#include <WiFi.h>            // WiFiClient nesnesi için gerekli
//#include <WiFiEnterprise.h>  // Sizin bulduğunuz kütüphane
#include "ThingSpeak.h"      // Veri gönderimi için

// ==========================================
// 1. EDUROAM AYARLARI
// ==========================================
/*const char* ssid = "eduroam";           
const char* username = "2019510134@ogr.deu.edu.tr"; // Tam mail adresi
const char* password = "28062001";            

// ==========================================
// 2. THINGSPEAK AYARLARI
// ==========================================
unsigned long myChannelNumber = 3191091;      // Kanal ID'niz
const char * myWriteAPIKey = "TJYJCHWZS9KHHHCU";    // Write API Key'iniz*/

//WiFiClient client; // ThingSpeak'in kullanacağı istemci/



String ssid = "Berdan's Phone";// The network name
String password = "brd2020M"; // The network password

String serverName = "GET https://api.thingspeak.com/update?api_key=TJYJCHWZS9KHHHCU";// Thingspeak command. Write your own API key in the key section.*/



//int rxPin = 10; // RX pin
//int txPin = 11; // TX pin

//String ip = "184.106.153.149";// Thingspeak IP address

// Constants (Change the following variables if needed)
const int anemometerPin = 34;  // GPIO pin connected to anemometer (analog pin)
const float minVoltage = 0.033;  // Voltage corresponding to 0 m/s
const float maxVoltage = 3.3;  // Voltage corresponding to 32.4 m/s (max speed) (when using voltage divider)
const float maxWindSpeed = 32.4; // Maximum wind speed in m/s

// Conversion factors
const float mps_to_kmh = 3.6;   // 1 m/s = 3.6 km/h
const float mps_to_mph = 2.23694; // 1 m/s = 2.23694 mph

// INA219 Object
Adafruit_INA219 ina219;

// ESP32 + SW-420 Vibration Sensor
int vibrationPin = 34;   // DOUT pin //27
int vibrationState = 0;

// --- Sensör pinleri ---
#define DHTPIN1 4
#define DHTPIN2 5
#define DHTPIN3 18

#define DHTTYPE DHT11

// --- 3 sensör nesnesi ---
DHT dht1(DHTPIN1, DHTTYPE);
DHT dht2(DHTPIN2, DHTTYPE);
DHT dht3(DHTPIN3, DHTTYPE);

WiFiClient client;
HTTPClient http;

void setup() {
  Serial.begin(9600);

  // INA219 Begins (ESP32'de G21 ve G22 pinlerini otomatik kullanır)
  if (! ina219.begin()) {
    Serial.println("HATA: INA219 çipi bulunamadı!");
    Serial.println("Lütfen bağlantı kablolarını (SDA, SCL, VCC, GND) kontrol et.");
    //while (1) { delay(10); } // Sonsuz döngüde bekle
  } else {
    Serial.println("INA219 başarıyla bağlandı!");
  }
  
  // Vibration Sensor Begins
  pinMode(vibrationPin, INPUT);

  // DHT Begins
  dht1.begin();
  dht2.begin();
  dht3.begin();

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("Connected to WiFi, IP Address: ");
  Serial.println(WiFi.localIP());

  // --- BAĞLANTI BAŞLANGIÇ ---
  /*Serial.println("Eduroam'a baglaniyor...");
  
  // Bulduğunuz kütüphanenin fonksiyonu (debug modu = true açık)
  if (WiFiEnterprise.begin(ssid, username, password, true)) {
    Serial.println("\n✅ Baglanti Basarili!");
    Serial.print("IP Adresi: ");
    Serial.println(WiFiEnterprise.localIP());
  } else {
    Serial.println("\n❌ Baglanti Basarisiz!");
    Serial.println("Lutfen kullanici adi ve sifreyi kontrol edin.");
  }

  // ThingSpeak Başlat
  ThingSpeak.begin(client);/

  /******************************/

  /*Serial.println("Sistem Baslatiliyor (ESP32 v3.0+ Modu)...");

  // --- EDUROAM BAĞLANTISI (YENİ YÖNTEM) ---
  Serial.println("Eduroam'a baglaniyor...");
  
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);

  // v3.0 için Yeni Yapılandırma
  esp_eap_client_config_t config;
  memset(&config, 0, sizeof(config)); // Config'i sıfırla
  
  config.identity = (uint8_t *)username;
  config.identity_len = strlen(username);
  config.username = (uint8_t *)username;
  config.username_len = strlen(username);
  config.password = (uint8_t *)password;
  config.password_len = strlen(password);

  // Sertifika kontrolünü devre dışı bırak (En kolay bağlantı için)
  esp_eap_client_set_disable_time_check(true); 

  // Yapılandırmayı uygula
  esp_eap_client_set_config(&config);
  esp_eap_client_enable();

  WiFi.begin(ssid);

  int counter = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    counter++;
    if(counter > 40) { // 20 saniye bağlanamazsa
        Serial.println("\nZaman asimi! Reset atiliyor...");
        ESP.restart();
    }
  }

  Serial.println("\n✅ Wi-Fi Baglandi!");
  Serial.print("IP Adresi: ");
  Serial.println(WiFi.localIP());

  ThingSpeak.begin(client);*/


}

void loop() {
  // ---------------------- Vibration Sensor (START) --------------------------------------------------
  /*vibrationState = digitalRead(vibrationPin);

  if (vibrationState == HIGH) {
    Serial.println("Vibration Detected!");
  } else {
    Serial.println("No vibration");
  }*/

  int vibration = readVibrationFiltered();

  // 2) %0 - %100 hesaplama
  float vibrationPercent = (vibration / 4095.0) * 100.0;

  // 3) Yazdırma
  Serial.print("Raw: ");
  Serial.print(vibration);
  Serial.print("   Vibration (%): ");
  Serial.println(vibrationPercent);

  // ---------------------- Vibration Sensor (END) -------------------------------------------------

  // ---------------------- Anemometer (START) -----------------------------------------------------
  // Read analog value from anemometer (ADC value between 0-4095 on ESP32 for 0-3.3V)
  int adcValue = analogRead(anemometerPin);
  
  // Convert ADC value to voltage (ESP32 ADC range is 0-3.3V)
  float voltage = (adcValue / 4095.00) * 3.3;
  
  // Ensure the voltage is within the anemometer operating range
  if (voltage < minVoltage) {
    voltage = minVoltage;
  } else if (voltage > maxVoltage) {
    voltage = maxVoltage;
  }
  
  // Map the voltage to wind speed
  float windSpeed_mps = ((voltage - minVoltage) / (maxVoltage - minVoltage)) * maxWindSpeed;

  // Convert wind speed to km/h and mph
  float windSpeed_kmh = windSpeed_mps * mps_to_kmh;
  float windSpeed_mph = windSpeed_mps * mps_to_mph;

  // Print wind speed
  if (windSpeed_mps != 0 || windSpeed_kmh != 0) {
    Serial.print("Wind Speed: ");
    Serial.print(windSpeed_mps);
    Serial.print(" m/s, ");
    Serial.print(windSpeed_kmh);
    Serial.print(" km/h, ");
    Serial.print(windSpeed_mph);
    Serial.println(" mph");
  }

  // ---------------------- Anemometer (END) -----------------------------------------------------

  // ---------------------- INA219 DC Current Sensor (START) -------------------------------------
  /*float shuntVoltage = 0;
  float busVoltage = 0;
  float current_mA = 0;
  float loadVoltage = 0;
  float power_mW = 0;

  // Sensörden verileri oku
  shuntVoltage = ina219.getShuntVoltage_mV(); // Şönt üzerindeki küçük voltaj düşümü
  busVoltage = ina219.getBusVoltage_V();      // Asıl ölçülen voltaj (Yük voltajı)
  current_mA = ina219.getCurrent_mA();        // Çekilen akım (Miliamper)
  power_mW = ina219.getPower_mW();            // Harcanan güç (Miliwatt)
  
  // Toplam voltajı hesapla (Bus + Shunt)
  loadVoltage = busVoltage + (shuntVoltage / 1000);

  if (isnan(current_mA)) {
    current_mA = 0;
  }
  
  if (current_mA != 0) {
    Serial.print("Voltaj (Bus):   "); Serial.print(busVoltage); Serial.println(" V");
    Serial.print("Akım (Current): "); Serial.print(current_mA); Serial.println(" mA");
    Serial.print("Güç (Power):    "); Serial.print(power_mW); Serial.println(" mW");
    
    Serial.println("-----------------------------------");
  }*/
  // ---------------------- INA219 DC Current Sensor (END) -------------------------------------


  // ---------------------------------- DHT Sensor (START) -------------------------------------
  // --- DHT Sensor 1 ---
  float h1 = dht1.readHumidity();
  float t1 = dht1.readTemperature();

  // --- DHT Sensor 2 ---
  float h2 = dht2.readHumidity();
  float t2 = dht2.readTemperature();

  // --- DHT Sensor 3 ---
  float h3 = dht3.readHumidity();
  float t3 = dht3.readTemperature();

  // Error Control
  if (isnan(h1) || isnan(t1) || isnan(h2) || isnan(t2) || isnan(h3) || isnan(t3)) {
    Serial.println("Sensörlerden biri okunamadı! Bağlantıları kontrol et.");
    return;
  }

  Serial.println("----- DHT Sensor 1 -----");
  Serial.print("Humidity: "); Serial.print(h1); Serial.print(" %  ");
  Serial.print("Temperature: "); Serial.print(t1); Serial.println(" °C");

  Serial.println("----- DHT Sensor 2 -----");
  Serial.print("Humidity: "); Serial.print(h2); Serial.print(" %  ");
  Serial.print("Temperature: "); Serial.print(t2); Serial.println(" °C");

  Serial.println("----- DHT Sensor 3 -----");
  Serial.print("Humidity: "); Serial.print(h3); Serial.print(" %  ");
  Serial.print("Temperature: "); Serial.print(t3); Serial.println(" °C");

  float busVoltage = 0;      // Asıl ölçülen voltaj (Yük voltajı)
  float current_mA = 0;        // Çekilen akım (Miliamper)
  float power_mW = 0;            // Harcanan güç (Miliwatt)
  
  Serial.println("----------------------------\n");
  // ----------------------------------- DHT Sensor (END) -------------------------------------

  // Önce bağlantıyı kontrol et, kopmuşsa tekrar bağlan
    if (!WiFiEnterprise.isConnected()) {
        Serial.println("❌ Baglanti koptu! Tekrar deneniyor...");
        if (WiFiEnterprise.begin(ssid, username, password, true)) {
          Serial.println("✅ Tekrar baglandi!");
        } else {
          Serial.println("❌ Tekrar baglanamadi. Bekleniyor...");
          delay(5000);
          return; // Bağlantı yoksa aşağıya (veri göndermeye) geçme
        }
    }

    // Bağlantı varsa işlemlere devam et:
    Serial.println("📶 Baglanti aktif. Veriler hazirlaniyor...");

    // -----------------------------------------------------------
    // 2. THINGSPEAK PAKETLEME
    // -----------------------------------------------------------
    /*ThingSpeak.setField(1, vibration);      
    ThingSpeak.setField(2, windSpeed_kmh);  
    ThingSpeak.setField(3, t1);             
    ThingSpeak.setField(4, t2);             
    ThingSpeak.setField(5, t3);             
    ThingSpeak.setField(6, busVoltage);     
    ThingSpeak.setField(7, current_mA);     
    ThingSpeak.setField(8, power_mW);       

    // -----------------------------------------------------------
    // 3. GÖNDERME
    // -----------------------------------------------------------
    int x = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);

    if(x == 200){
      Serial.println("✅ Veriler ThingSpeak bulutuna gonderildi.");
    }
    else{
      Serial.println("⚠️ Gonderim hatasi! HTTP Kodu: " + String(x));
    }

    // ThingSpeak Free limiti (En az 15 sn, güvenli olması için 20 sn)
    Serial.println("Bir sonraki gonderim icin 15 sn bekleniyor...");
    delay(15000);*/










      serverName += "&field1=";
      serverName += String(vibration); // The vibration variable to be sent
      serverName += "&field2=";
      serverName += String(windSpeed_kmh); // The windspeed variable to be sent
      serverName += "&field3=";
      serverName += String(t1); // The temperature1 variable to be sent
      serverName += "&field4=";
      serverName += String(t2); // The temperature2 variable to be sent
      serverName += "&field5=";
      serverName += String(t3); // The temperature3 variable to be sent
      serverName += "&field6=";
      serverName += String(busVoltage); // The busVoltage variable to be sent
      serverName += "&field7=";
      serverName += String(current_mA); // The current_mA variable to be sent
      serverName += "&field8=";
      serverName += String(power_mW); // The power_mW variable to be sent

      String serverPath = serverName;


      // Start HTTP request
      http.begin(client, serverPath.c_str());

      // Send HTTP GET request
      int httpResponseCode = http.GET();

      if (httpResponseCode > 0) {
        Serial.print("HTTP Response code: ");
        Serial.println(httpResponseCode);
        String payload = http.getString();
        Serial.println(payload);  // Print the response payload
      } else {
        Serial.print("Error code: ");
        Serial.println(httpResponseCode);  // Error code if request failed
      }

      // End the HTTP request and free resources
      http.end();











  // ******** SOFTWARE SERIAL *********** //
  /*//String data = "GET https://api.thingspeak.com/update?api_key=TJYJCHWZS9KHHHCU";// Thingspeak command. Write your own API key in the key section.
  data += "&field1=";
  data += String(vibration); // The vibration variable to be sent
  data += "&field2=";
  data += String(windSpeed_kmh); // The windspeed variable to be sent
  data += "&field3=";
  data += String(t1); // The temperature1 variable to be sent
  data += "&field4=";
  data += String(t2); // The temperature2 variable to be sent
  data += "&field5=";
  data += String(t3); // The temperature3 variable to be sent
  data += "&field6=";
  data += String(busVoltage); // The busVoltage variable to be sent
  data += "&field7=";
  data += String(current_mA); // The current_mA variable to be sent
  data += "&field6=";
  data += String(power_mW); // The power_mW variable to be sent
  
  data += "\r\n\r\n";
  esp.print("AT+CIPSEND="); // The length of the data to be sent to the ESP is provided
  esp.println(data.length() + 2);
  delay(2000);
  if (esp.find(">")) { // Commands are executed when the ESP8266 is ready.
  esp.print(data); // The data is being sent
  Serial.println(data);
  Serial.println("Data sent");
  delay(1000);
  }
  Serial.println("Connection closed");
  esp.println("AT+CIPCLOSE"); // The connection is being closed
  delay(1000); // A 1-minute wait is observed for new data transmission*/
  // ******** SOFTWARE SERIAL *********** //
}

int readVibrationFiltered() {
  long sum = 0;

  for (int i = 0; i < 20; i++) {
    sum += analogRead(vibrationPin); // En fazla %42 gördük.
    delay(2);  // ADC daha stabil okur
  }

  return sum / 20;  // Ortalama değer
}
