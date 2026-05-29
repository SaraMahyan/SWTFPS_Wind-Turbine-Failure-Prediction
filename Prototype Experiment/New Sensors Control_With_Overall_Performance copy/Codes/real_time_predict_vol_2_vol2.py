import serial
import joblib
import pandas as pd
import time
import warnings
import threading
import requests  # ThingSpeak'e veri göndermek için eklendi

warnings.filterwarnings("ignore")

# =========================================================================
# AYARLAR (Burayı kendi bilgisayarına göre değiştir)
# =========================================================================
MODEL_PATH = 'prototype_model_stacking.pkl' 

# ESP'lerin takılı olduğu portlar
ESP1_PORT = 'COM7'  # Rüzgar, Voltaj, Sıcaklık, Gaz, Çizgi
ESP2_PORT = 'COM3'  # Titreşim ve Ses
BAUD_RATE = 9600

# --- ThingSpeak Ayarları ---
TS_CHANNEL_ID = "3191268"
TS_WRITE_API_KEY = "7S56QDYSCE3K15RX"

# =========================================================================
# MODEL ÖZELLİKLERİ (Eğitim CSV'si ile tamamen aynı sıra olmalı!)
# =========================================================================
FEATURE_NAMES = [
    'Temperature_1', 'Wind_Direction', 'Temperature_2', 'Temperature_3', 
    'Wind_Speed_kmh', 'Turbine_Voltage', 'Turbine_Current', 'Turbine_Power', 
    'Gas_Value', 'Line_Sensor', 'Sound_Level', 'Vibration_Value'
]

# Canlı veriyi tutacağımız global sözlük
latest_sensor_data = {
    'Wind_Direction': 0.0, 'Wind_Speed_kmh': 0.0, 'Temperature_1': 0.0, 
    'Temperature_2': 0.0, 'Temperature_3': 0.0, 'Turbine_Voltage': 0.0, 
    'Turbine_Current': 0.0, 'Turbine_Power': 0.0, 'Gas_Value': 0.0, 
    'Line_Sensor': 0.0, 'Sound_Level': 0.0, 'Vibration_Value': 0.0
}

# Veri güncelleme kilit mekanizması (Thread güvenliği için)
data_lock = threading.Lock()

# =========================================================================
# ThingSpeak Gönderim Fonksiyonu
# =========================================================================
def send_to_thingspeak(prediction):
    # Tahmin sonucunu Field 5'e yazdırıyoruz
    url = f"https://api.thingspeak.com/update?api_key={TS_WRITE_API_KEY}&field5={prediction}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and response.text != '0':
            print("🌐 [BAŞARILI] Yapay Zeka kararı ThingSpeak'e (Field 5) iletildi!")
        else:
            print("⚠️ [UYARI] ThingSpeak'e yazılamadı (15 saniye limiti aşılmış olabilir).")
    except Exception as e:
        print(f"❌ [HATA] ThingSpeak Bağlantı Hatası: {e}")

# =========================================================================
# ESP1 OKUMA THREAD'İ
# =========================================================================
def read_esp1():
    try:
        ser1 = serial.Serial(ESP1_PORT, BAUD_RATE, timeout=2)
        print(f"✅ ESP1 Bağlandı ({ESP1_PORT})")
        while True:
            if ser1.in_waiting > 0:
                line = ser1.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("CSV"):
                    parts = line.split(',')
                    if len(parts) >= 12:
                        try:
                            with data_lock:
                                latest_sensor_data['Wind_Direction'] = float(parts[2])
                                latest_sensor_data['Wind_Speed_kmh'] = float(parts[3])
                                latest_sensor_data['Temperature_1'] = float(parts[4])
                                latest_sensor_data['Temperature_2'] = float(parts[5])
                                latest_sensor_data['Temperature_3'] = float(parts[6])
                                latest_sensor_data['Turbine_Voltage'] = float(parts[7])
                                latest_sensor_data['Turbine_Current'] = float(parts[8])
                                latest_sensor_data['Turbine_Power'] = float(parts[9])
                                latest_sensor_data['Gas_Value'] = float(parts[10])
                                latest_sensor_data['Line_Sensor'] = float(parts[11])
                        except ValueError as ve:
                            pass
    except Exception as e:
        print(f"❌ ESP1 Bağlantı Hatası: {e}")

# =========================================================================
# ESP2 OKUMA THREAD'İ
# =========================================================================
def read_esp2():
    try:
        ser2 = serial.Serial(ESP2_PORT, BAUD_RATE, timeout=2)
        print(f"✅ ESP2 Bağlandı ({ESP2_PORT})")
        while True:
            if ser2.in_waiting > 0:
                line = ser2.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("CSV"):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            with data_lock:
                                latest_sensor_data['Sound_Level'] = float(parts[2]) 
                                latest_sensor_data['Vibration_Value'] = float(parts[3]) if len(parts)>3 else 0.0
                        except ValueError:
                            pass
    except Exception as e:
        print(f"❌ ESP2 Bağlantı Hatası: {e}")

# =========================================================================
# YAPAY ZEKA MODELİNİ YÜKLEME
# =========================================================================
def load_ai_model():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model başarıyla yüklendi: {MODEL_PATH}\n")
        return model
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return None

# =========================================================================
# ANA DÖNGÜ (YAPAY ZEKA TAHMİNİ)
# =========================================================================
if __name__ == '__main__':
    print("="*65)
    print(" 🌬️ ÇİFT ESP'Lİ RÜZGAR TÜRBİNİ YAPAY ZEKA SİSTEMİ BAŞLATILIYOR")
    print("="*65)
    
    model = load_ai_model()
    if model is None:
        exit()

    thread1 = threading.Thread(target=read_esp1, daemon=True)
    thread2 = threading.Thread(target=read_esp2, daemon=True)
    thread1.start()
    thread2.start()
    
    print("⏳ Sensörlerin veri göndermesi bekleniyor (3 saniye)...\n")
    time.sleep(3) 
    
    last_thingspeak_update = 0 # ThingSpeak zamanlayıcısı için değişken
    
    try:
        while True:
            with data_lock:
                current_data = latest_sensor_data.copy()
            
            df_live = pd.DataFrame([current_data], columns=FEATURE_NAMES)
            
            # YAPAY ZEKA TAHMİNİ
            prediction = int(model.predict(df_live)[0])
            durum_ikon = "🔴 ARIZA TESPİT EDİLDİ (1)!" if prediction == 1 else "🟢 SİSTEM NORMAL (0)"
            
            # TERMİNAL EKRANI (Saniyede 1 güncellenir)
            print("-" * 65)
            print(f"Rüzgar : {current_data['Wind_Speed_kmh']:>5.1f} km/h | Yön: {current_data['Wind_Direction']:>3.0f}° | Titreşim: {current_data['Vibration_Value']:>4.0f}")
            print(f"Voltaj : {current_data['Turbine_Voltage']:>5.2f} V    | Akım: {current_data['Turbine_Current']:>4.2f} A | Güç: {current_data['Turbine_Power']:>5.2f} W")
            print(f"Sıcaklık: Ortam {current_data['Temperature_1']:.1f}°C | Motor {current_data['Temperature_2']:.1f}°C | Dişli {current_data['Temperature_3']:.1f}°C")
            print(f"Gaz/Duman: {current_data['Gas_Value']:.0f} | Ses: {current_data['Sound_Level']:.0f}")
            print(f"\n>> YAPAY ZEKA DURUMU: {durum_ikon}")
            print("-" * 65)
            
            # THINGSPEAK GÜNCELLEMESİ (Sadece 15 saniyede bir çalışır)
            current_time = time.time()
            if current_time - last_thingspeak_update >= 15:
                # Arka planda donmayı önlemek için ThingSpeak gönderimini ayrı bir thread olarak çalıştırıyoruz
                threading.Thread(target=send_to_thingspeak, args=(prediction,), daemon=True).start()
                last_thingspeak_update = current_time

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nSistem kullanıcı tarafından durduruldu.")