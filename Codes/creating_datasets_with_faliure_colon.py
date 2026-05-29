import pandas as pd
import numpy as np
import os

# ===========================
# AYARLAR
# ===========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Event bilgisi dosyası
EVENT_INFO_PATH = os.path.join( "event_info.csv")

# Kaç saat önceyi "failure = 1" kabul edeceğiz?
EARLY_WARNING_HOURS = 6 
STEP_MINUTES = 10    
WINDOW_STEPS = int(EARLY_WARNING_HOURS * 60 / STEP_MINUTES)  

# ===========================
# 1) EVENT INFO OKU
# ===========================

event_info = pd.read_csv(EVENT_INFO_PATH,sep=";")

# Bazı datasetlerde event_id kolonunun tipi string, bazılarında int olabilir;
# dosya adıyla eşleştirmek için stringe çeviriyoruz
event_info["event_id"] = event_info["event_id"].astype(str)

print("Toplam event satırı:", len(event_info))
print(event_info.head())

# ===========================
# 2) HER EVENT DOSYASI İÇİN FAILURE OLUŞTUR
# ===========================

for idx, row in event_info.iterrows():
    event_id = row["event_id"]
    event_label = row["event_label"]   # "anomaly" veya "normal"
    start_id = row["event_start_id"]
    end_id   = row["event_end_id"]

    # Giriş ve çıkış dosya yolları
    input_csv  = os.path.join(BASE_DIR, f"{event_id}.csv")
    output_csv = os.path.join(BASE_DIR, f"{event_id}_with_failure.csv")

    if not os.path.exists(input_csv):
        print(f"[UYARI] {input_csv} bulunamadı, atlanıyor...")
        continue

    print(f"\n=== Event {event_id} ({event_label}) işleniyor... ===")
    df = pd.read_csv(input_csv,sep=";")

    if "id" not in df.columns:
        raise ValueError(f"{input_csv} içinde 'id' kolonu yok, event_info ile eşleştiremeyiz.")

    # Her zaman bir 'failure' kolonu oluştur, varsayılan 0
    df["failure"] = 0

    if event_label.lower() == "anomaly":
        # Pencere başlangıcını hesapla (id bazlı)
        # Başlangıçta eğer negatif id çıkarsa, dataset'in min id'sine kırpıyoruz
        start_warning_id = start_id - WINDOW_STEPS
        min_id = df["id"].min()
        max_id = df["id"].max()

        start_warning_id = max(start_warning_id, min_id)
        failure_start_id = start_warning_id
        failure_end_id   = min(end_id, max_id)

        # id aralığına göre mask
        mask_failure = (df["id"] >= failure_start_id) & (df["id"] <= failure_end_id)
        df.loc[mask_failure, "failure"] = 1

        print(f"  -> anomaly: id [{failure_start_id}, {failure_end_id}] aralığı failure=1")
        print(f"     (erken uyarı {EARLY_WARNING_HOURS} saat önceye kadar)")

    else:
        # "normal" ise tüm satırlar 0 olarak kalır
        print("  -> normal event: tüm satırlar failure=0")

    # İsteğe bağlı: bilgi amaçlı dağılım yazdıralım
    counts = df["failure"].value_counts(dropna=False)
    print("  -> Dağılım:", counts.to_dict())

    df.to_csv(output_csv, index=False)
    print(f"  -> Kaydedildi: {output_csv}")

print("\n>>> Tüm event dosyaları için failure etiketleri oluşturuldu.")
