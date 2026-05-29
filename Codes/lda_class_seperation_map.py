import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# ===========================
# 1) VERİYİ YÜKLE
# ===========================
df = pd.read_csv("sampled_wind_data.csv")

# time_stamp varsa datetime'a çevir (ileride lazım olabilir)
if "time_stamp" in df.columns:
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")

# ===========================
# 2) TARGET VE FEATURE SEÇİMİ
# ===========================
target_col = "failure"
exclude_cols = ["time_stamp", "turbine_id"]       # feature olarak kullanmayacağımız kolonlar
exclude_cols = [c for c in exclude_cols if c in df.columns]

# Hedef değişken
y = df[target_col].values

# Sayısal kolonlar
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Feature kolonları (failure, time_stamp, turbine_id hariç)
feature_cols = [c for c in num_cols if c not in exclude_cols + [target_col]]
print(f"Feature sayısı: {len(feature_cols)}")

X = df[feature_cols].copy()

# Her ihtimale karşı: feature tarafında NaN veya inf kalmasın
X = X.replace([np.inf, -np.inf], np.nan)
mask_valid = ~X.isna().any(axis=1)

# Hem X hem y'yi aynı mask ile filtrele
X = X[mask_valid]
y = y[mask_valid]

print(f"Geçerli (NaN/inf olmayan) örnek sayısı: {X.shape[0]}")
print("Sınıf dağılımı:", np.unique(y, return_counts=True))

# ===========================
# 3) ÖLÇEKLEME (StandardScaler)
# ===========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# 4) LDA (1 bileşen)
# ===========================
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)   # shape: (n_samples, 1)
lda_scores = X_lda[:, 0]

print("LDA sınıf ortalamaları (projeksiyon uzayında):")
print("  class 0 mean:", lda_scores[y == 0].mean())
print("  class 1 mean:", lda_scores[y == 1].mean())

# ===========================
# 5) LDA EKSENİNDE SINIF AYRIMI GRAFİĞİ
# ===========================
plt.figure(figsize=(10, 6))

bins = 80  # histogram çözünürlüğü

# Normal (0) sınıfı
plt.hist(
    lda_scores[y == 0],
    bins=bins,
    alpha=0.5,
    density=True,
    label="Normal (0)"
)

# Failure (1) sınıfı
plt.hist(
    lda_scores[y == 1],
    bins=bins,
    alpha=0.5,
    density=True,
    label="Failure (1)"
)

plt.xlabel("LDA Component (class-separating axis)")
plt.ylabel("Density")
plt.title("Class Separation along LDA Axis (Failure vs Normal)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lda_class_separation_hist.png", dpi=300, bbox_inches="tight")
plt.show()
