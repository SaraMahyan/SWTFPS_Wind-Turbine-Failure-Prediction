import os
import pandas as pd

# 1) Çalışma klasörünü ayarla (script Wind_Farm_A içindeyse gerek bile yok)
BASE_DIR = r"c:\Users\berda\OneDrive\Masaüstü\Wind Energy Codes\Wind_Farm_A"
os.chdir(BASE_DIR)

# 2) *_with_failure.csv dosyalarını bul
all_files = [
    f for f in os.listdir(BASE_DIR)
    if f.endswith("_with_failure.csv") and os.path.isfile(os.path.join(BASE_DIR, f))
]

print("Bulunan with_failure dosyaları:")
for f in all_files:
    print("  -", f)

dfs = []

for f in all_files:
    path = os.path.join(BASE_DIR, f)
    print(f"\n[OKUMA] {f}")

    # Eğer with_failure dosyalarını virgüllü kaydettiysen:
    df = pd.read_csv(path)

    # İstersen hangi event/dosya olduğunu takip etmek için bir kolon ekleyebiliriz
    df["dataset_name"] = f.replace("_with_failure.csv", "")

    dfs.append(df)

# 3) Hepsini birleştir
combined = pd.concat(dfs, ignore_index=True)

# 4) time_stamp'i datetime yap ve sırala (opsiyonel ama tavsiye)
if "time_stamp" in combined.columns:
    combined["time_stamp"] = pd.to_datetime(combined["time_stamp"])
    combined = combined.sort_values("time_stamp").reset_index(drop=True)

print("\nBirleşik shape:", combined.shape)
print("Failure dağılımı:", combined["failure"].value_counts(dropna=False).to_dict())

# 5) Kaydet
output_path = os.path.join(BASE_DIR, "WIND_FARM_A_COMBINED_WITH_FAILURE.csv")
combined.to_csv(output_path, index=False)
print("\n>>> Kaydedildi:", output_path)
