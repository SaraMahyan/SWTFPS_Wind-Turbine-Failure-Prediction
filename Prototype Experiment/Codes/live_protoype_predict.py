import time
import requests
import joblib
import pandas as pd

# 1. Kaydedilen Modeli Yükle
MODEL_PATH = 'prototype_model_stacking.pkl' 

def load_model(model_path=MODEL_PATH):
    try:
        model = joblib.load(model_path)
        print(f"Model başarıyla yüklendi: {model_path}")
        return model
    except Exception as e:
        print(f"Model yüklenemedi {model_path}: {e}")
        return None

# --- ThingSpeak Ayarları ---
THINGSPEAK_CHANNEL1_READ_API_KEY = "T2VDVAVIEXVQZN0V"
THINGSPEAK_CHANNEL1_ID = "3373698"

THINGSPEAK_CHANNEL2_READ_API_KEY = "0077M542UPVHOHJK"
THINGSPEAK_CHANNEL2_ID = "3373682"
THINGSPEAK_CHANNEL2_WRITE_API_KEY = "GOR59CQUP1SL4HSP" # Sonucu Kanal 2'ye yazacağız

def get_latest_data():
    # 1. İstek: Kanal 1'i Çek (WindTurbine)
    url1 = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL1_ID}/feeds.json?api_key={THINGSPEAK_CHANNEL1_READ_API_KEY}&results=1"
    response1 = requests.get(url1).json()
    feed1 = response1['feeds'][0]

    # 2. İstek: Kanal 2'yi Çek (Wind_Turbine_2)
    url2 = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL2_ID}/feeds.json?api_key={THINGSPEAK_CHANNEL2_READ_API_KEY}&results=1"
    response2 = requests.get(url2).json()
    feed2 = response2['feeds'][0]
    
    # Görsellerdeki GÜNCEL ThingSpeak alanlarına göre veri sözlüğü
    # DİKKAT: Anahtar kelimeler modelinin eğitim kolon isimleriyle EŞLEŞMELİDİR.
    data = {
        'Voltage': float(feed1['field1']),        # CH1 Field 1
        'Power': float(feed1['field2']),          # CH1 Field 2
        'Current': float(feed1['field3']),        # CH1 Field 3
        'Wind_Speed': float(feed1['field4']),     # CH1 Field 4
        'Smoke': float(feed1['field5']),          # CH1 Field 5
        'Line_Detector': float(feed1['field6']),  # CH1 Field 6
        'Wind_Direction': float(feed1['field7']), # CH1 Field 7
        't3': float(feed1['field8']),             # CH1 Field 8
        'Vibration': float(feed2['field1']),      # CH2 Field 1
        'Sound': float(feed2['field2'])           # CH2 Field 2
    }
    
    return pd.DataFrame([data])

def update_thingspeak_status(failure):
    # Tahmin sonucunu Kanal 2'nin Field3 (Failure) alanına yazıyoruz!
    url = f"https://api.thingspeak.com/update?api_key={THINGSPEAK_CHANNEL2_WRITE_API_KEY}&field3={failure}"
    response = requests.get(url)
    
    if response.status_code == 200 and response.text != '0':
        print(f"Durum başarıyla ThingSpeak'e (Kanal 2, Field 3) iletildi: {'ARIZA (1)' if failure == 1 else 'NORMAL (0)'}")
    else:
        print("ThingSpeak'e veri yazılamadı (Rate limit veya bağlantı hatası).")

if __name__ == '__main__':
    model = load_model()

    while True:
        try:
            if model is None:
                raise RuntimeError("Model bulunamadı, döngü durduruluyor.")

            # 1. Güncel Sensör Verilerini Çek 
            new_data = get_latest_data()
            print("\n--- Yeni Veri Alındı ---")
            
            # 2. Tahmin Yap (Veriler doğrudan DataFrame olarak modele giriyor)
            prediction = int(model.predict(new_data)[0]) 
            print(f"Yapay Zeka Kararı: {prediction}")

            # 3. Sonucu ThingSpeak Kanal 2 - Field 3'e Yaz
            update_thingspeak_status(prediction)

            # ThingSpeak 15 saniye kuralına uygun bekleme süresi
            time.sleep(16)

        except Exception as e:
            print("Çalışma hatası:", e)
            time.sleep(16)