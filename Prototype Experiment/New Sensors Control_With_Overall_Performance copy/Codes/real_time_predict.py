import time
import serial
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- AYARLAR ---
MODEL_PATH = 'prototype_model_stacking.pkl' 
SERIAL_PORT = 'COM3'  # Kendi bilgisayarındaki porta göre değiştir
BAUD_RATE = 9600      # Arduino'daki Serial.begin(9600) ile aynı olmalı

def load_model(model_path=MODEL_PATH):
    try:
        model = joblib.load(model_path)
        print(f"Model başarıyla yüklendi: {model_path}")
        return model
    except Exception as e:
        print(f"Model yüklenemedi {model_path}: {e}")
        return None

def map_value(value, from_min, from_max, to_min, to_max):
    # Veri sınırların dışına çıkarsa kırp (clip)
    value = max(min(value, from_max), from_min)
    # Yeni aralığa oranla
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

# --- SCADA Scaler Hazırlığı ---
feature_columns = [
    'Temperature_1', 'Wind_Direction', 'Temperature_2', 
    'Temperature_3', 'Wind_Speed_kmh', 'Turbine_Voltage', 'Turbine_Power'
]

referans_min = [9.999999999999998, 9.8, 22.0, 34.0, 1.2, 394.6333333333334, -1987.0]
referans_max = [34.0, 351.8, 40.0, 67.0, 12.6, 401.9, 262752.0]

bounds_df = pd.DataFrame([referans_min, referans_max], columns=feature_columns)
scada_scaler = MinMaxScaler()
scada_scaler.fit(bounds_df)
print("SCADA referans sınırları sisteme tanımlandı.\n")

if __name__ == '__main__':
    model = load_model()
    
    if model is None:
        print("Sistem durduruldu. Model bulunamadı.")
        exit()

    print(f"{SERIAL_PORT} portuna bağlanılmaya çalışılıyor...")
    
    try:
        # Serial portu başlat
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print(f"Bağlantı Başarılı! Cihazdan veri bekleniyor...\n")
        print("="*65)
        
        # Sürekli dinleme döngüsü
        while True:
            if ser.in_waiting > 0:
                # Serial'den gelen satırı oku ve temizle
                line = ser.readline().decode('utf-8').strip()
                
                # Sadece "CSV" ile başlayan satırları işleme al
                if line.startswith("CSV"):
                    parts = line.split(',')
                    
                    if len(parts) >= 10:
                        try:
                            # Modele girecek olan 7 veriyi sırasıyla sözlüğe çekiyoruz
                            data = {
                                'Temperature_1': float(parts[4]),
                                'Wind_Direction': float(parts[2]),
                                'Temperature_2': float(parts[5]),
                                'Temperature_3': float(parts[6]),
                                'Wind_Speed_kmh': float(parts[3]),
                                'Turbine_Voltage': float(parts[7]),
                                'Turbine_Power': float(parts[9])
                            }
                            
                            # Sensörlerin GERÇEK fiziksel okuma sınırlarını SCADA model sınırlarına map ediyoruz.

                            # DHT11 Sıcaklık Sensörleri (0 - 50 °C aralığı makul, bırakılabilir)
                            data['Temperature_1'] = map_value(data['Temperature_1'], 0.0, 50.0, 10.0, 34.0)
                            data['Temperature_2'] = map_value(data['Temperature_2'], 0.0, 50.0, 22.0, 40.0)
                            data['Temperature_3'] = map_value(data['Temperature_3'], 0.0, 50.0, 34.0, 67.0)

                            # Rüzgar Yönü
                            data['Wind_Direction'] = map_value(data['Wind_Direction'], 0.0, 360.0, 9.8, 351.8)

                            # Rüzgar Hızı (Verinizde max 32.4 değil, 86 km/h civarına çıkmış)
                            data['Wind_Speed_kmh'] = map_value(data['Wind_Speed_kmh'], 0.0, 86.0, 1.2, 12.6)

                            # Türbin Voltajı (Verinize göre prototip max ~3.2V)
                            # ÖNEMLİ: Prototip dururken 0V üretir ama SCADA dururken şebekeden dolayı ~398V görür.
                            # Bu yüzden prototipin 0V'unu, SCADA'nın arıza minimumu (394.63) yerine sağlıklı boşta kalma voltajına (398.0) eşitliyoruz!
                            data['Turbine_Voltage'] = map_value(data['Turbine_Voltage'], 0.0, 3.2, 398.0, 401.9)

                            # Türbin Gücü (Verinize göre prototip max ~2.5W)
                            # ÖNEMLİ: Prototipiniz 0W üretirken arızalı değildir, sadece duruyordur.
                            # 0W'ı, SCADA'nın arıza değeri olan -1987W'a DEĞİL, 0.0W'a eşitliyoruz!
                            data['Turbine_Power'] = map_value(data['Turbine_Power'], 0.0, 2.5, 0.0, 262752.0)
                            
                            new_data = pd.DataFrame([data])
                            
                            # 1. Gerçek Scale Edilmiş Veri (Eksilere veya 1'in üstüne çıkabilen)
                            new_data_scaled = scada_scaler.transform(new_data)
                            
                            # 2. Sadece görselleştirme için 0-1 arasına sıkıştırılmış (clipped) veri
                            new_data_clipped = np.clip(new_data_scaled, 0.0, 1.0)
                            
                            # Tahmin Yap (Model gerçek sınırları görmeli, bu yüzden new_data_scaled kullanıyoruz)
                            prediction = int(model.predict(new_data_scaled)[0]) 
                            durum = "ARIZALI (1) ❌" if prediction == 1 else "NORMAL (0) ✅"
                            
                            # Terminal Gösterge Paneli
                            print(f"Rüzgar: {data['Wind_Speed_kmh']:>5.1f} km/h | Yön: {data['Wind_Direction']:>5.0f}°")
                            print(f"Voltaj: {data['Turbine_Voltage']:>5.1f} V    | Güç: {data['Turbine_Power']:>5.1f} W")
                            print(f"Ortam Sıcaklık : {data['Temperature_1']:>5.1f} °C")
                            print(f"Nacelle Sıcaklık: {data['Temperature_2']:>5.1f} °C")
                            print(f"Gearbox Sıcaklık: {data['Temperature_3']:>5.1f} °C")
                            
                            # Yan Yana Karşılaştırma Tablosu
                            print("\n--- SCADA Veri Karşılaştırması ---")
                            print(f"{'Sensör':<18} | {'Gerçek Scale':<15} | {'0-1 Sınırlandırılmış'}")
                            print("-" * 60)
                            
                            for col, val_scaled, val_clipped in zip(feature_columns, new_data_scaled[0], new_data_clipped[0]):
                                print(f"{col:<18} | {val_scaled:>10.4f}      | {val_clipped:>15.4f}")
                                
                            print(f"\n-> YAPAY ZEKA DURUMU: {durum}")
                            print("=" * 65)
                            
                        except ValueError as ve:
                            pass

    except serial.SerialException as e:
        print(f"Serial port hatası! Kablonun takılı olduğundan ve doğru port (COM) yazdığından emin ol.\nHata Detayı: {e}")
    except KeyboardInterrupt:
        print("\nTest kullanıcı tarafından sonlandırıldı.")
        if 'ser' in locals() and ser.is_open:
            ser.close()