import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def predict_and_save_to_csv(csv_file_path, model_path, output_csv_path):
    # 1. CSV dosyasını oku
    print("CSV dosyası okunuyor...")
    df = pd.read_csv(csv_file_path)
    
    # 2. Modelin beklediği sıraya göre özellikleri seç
    # Eğitim sırası: Ambient Temp, Wind Dir, Nacelle Temp, Gearbox Temp, Wind Speed, Voltage, Power
    # Eşleşme: Temp 1 = Ambient, Temp 2 = Nacelle, Temp 3 = Gearbox
    feature_columns = [
        'Temperature_1',    # Ambient Temperature
        'Wind_Direction',   # Wind Direction
        'Temperature_2',    # Nacelle Temperature
        'Temperature_3',    # Gearbox Temperature
        'Wind_Speed_kmh',   # Wind Speed
        'Turbine_Voltage',  # Voltage
        'Turbine_Power'     # Power
    ]
    
    # Veri setinden sadece gerekli kolonları ayır
    X = df[feature_columns]
    
    # 3. Min-Max Scaler işlemini uygula
    print("Veriler MinMaxScaler ile ölçeklendiriliyor...")
    scaler = MinMaxScaler()
    
    # ÖNEMLİ NOT: Eğer modeli eğitirken kullandığınız scaler'ı kaydettiyseniz (örn: scaler.pkl), 
    # burada fit_transform yerine o scaler'ı yükleyip transform() yapmanız test verisi için daha doğrudur.
    X_scaled = scaler.fit_transform(X)
    
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
    CSV_PATH = "combined_sensor_data_v0.csv"                      # Okunacak orijinal CSV dosyasının yolu
    MODEL_PATH = "prototype_model.pkl"                        # Eğitilmiş modelin yolu
    OUTPUT_PATH = "labelled_combined_sensor_data_v0.csv"         # Çıktı olarak alınacak yeni CSV dosyasının adı
    
    predict_and_save_to_csv(CSV_PATH, MODEL_PATH, OUTPUT_PATH)