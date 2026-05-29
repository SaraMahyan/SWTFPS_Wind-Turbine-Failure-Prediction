import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def predict_and_save_to_csv(csv_file_path, model_path, output_csv_path):
    # 1. CSV dosyasını oku
    print("CSV dosyası okunuyor...")
    df = pd.read_csv(csv_file_path)
    
    # 2. Modelin beklediği sıraya göre özellikleri seç
    feature_columns = [
        'Temperature_1',    # Ambient Temperature (sensor_0_avg)
        'Wind_Direction',   # Wind Direction (sensor_1_avg)
        'Temperature_2',    # Nacelle Temperature (sensor_43_avg)
        'Temperature_3',    # Gearbox Temperature(sensor_11_avg)
        'Wind_Speed_kmh',   # Wind Speed (wind_speed_3_avg)
        'Turbine_Voltage',  # Voltage 
        'Turbine_Power'     # Power (sensor_50)
    ]
    
    # Veri setinden sadece gerekli kolonları ayır
    X = df[feature_columns].copy()
    
    # 3. Kesin SCADA Sınır Değerlerini Tanımla ve Ölçeklendir
    print("Kesin referans sınır değerlerine göre MinMaxScaler ayarlanıyor...")
    
    # Orijinal veri setinden çıkarılan tam küsuratlı Min ve Max değerleri
    referans_min = [
        9.999999999999998,   # Temperature_1
        9.8,                 # Wind_Direction
        22.0,                # Temperature_2
        34.0,                # Temperature_3
        1.2,                 # Wind_Speed_kmh
        394.6333333333334,   # Turbine_Voltage
        -1987.0              # Turbine_Power
    ]
    
    referans_max = [
        34.0,                # Temperature_1
        351.8,               # Wind_Direction
        40.0,                # Temperature_2
        67.0,                # Temperature_3
        12.6,                # Wind_Speed_kmh
        401.9,               # Turbine_Voltage
        262752.0             # Turbine_Power
    ] 
    
    # Sınırları içeren geçici bir DataFrame oluşturup Scaler'a veriyoruz
    bounds_df = pd.DataFrame([referans_min, referans_max], columns=feature_columns)
    
    scaler = MinMaxScaler()
    scaler.fit(bounds_df) # Scaler artık en hassas limitleri öğrendi
    
    # Prototip verisine transform() uyguluyoruz.
    X_scaled = scaler.transform(X)
    
    # 4. Modeli yükle
    print("Model yükleniyor...")
    model = joblib.load(model_path)
    
    # 5. Tahmin (Prediction) yap
    print("Tahminler yapılıyor...")
    predictions = model.predict(X_scaled)
    
    # 6. Tahminleri orijinal DataFrame'e yeni bir kolon olarak ekle
    df['failure'] = predictions
    
    # 7. Sonuçları yeni bir CSV dosyasına kaydet
    print(f"Sonuçlar {output_csv_path} dosyasına kaydediliyor...")
    df.to_csv(output_csv_path, index=False)
    print("İşlem başarıyla tamamlandı!")

# --- KULLANIM ---
if __name__ == "__main__":
    CSV_PATH = "combined_sensor_data.csv"                      
    MODEL_PATH = "prototype_model_best_v2.pkl"                 
    OUTPUT_PATH = "labelled_combined_sensor_data.csv"          
    
    predict_and_save_to_csv(CSV_PATH, MODEL_PATH, OUTPUT_PATH)