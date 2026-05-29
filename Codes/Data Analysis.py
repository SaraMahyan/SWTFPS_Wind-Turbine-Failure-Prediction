import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import os
from scipy.stats import skew, kurtosis, variation, mode, median_abs_deviation, hmean, gmean, t

# Çıktı dosyasının adı
output_file_path = "WIND_FARM_A_COMBINED_WITH_FAILURE.csv"

def reading_csv(file_path):
    df = pd.read_csv(file_path)
    return df

try:
    # 1. İşlenmiş dosyayı oku
    df = reading_csv(output_file_path)

    print("\n--- CSV İlk 5 Satır ---")
    print(df.head())
    print("\n--- Kolon İsimleri ---")
    print(df.columns.tolist())

    target_col = "failure"

    # ====== BURASI EKLENDİ: failure kolonu kontrol + numerik hale getirme ======
    if df.empty or target_col not in df.columns:
        print("\n--- HATA (Adım 1) ---")
        print(f"'{output_file_path}' dosyası boş veya '{target_col}' kolonunu içermiyor.")
        print("Lütfen Adım 1'in başarıyla tamamlandığından emin olun.")
        raise SystemExit()

    # failure kolonunu numerik hale getir (0/1 gibi)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    if df[target_col].isna().all():
        print(f"\n--- HATA ---")
        print(f"'{target_col}' kolonu sayıya çevrilemedi (tamamen NaN).")
        print("Lütfen bu kolonu 0/1 şeklinde numerik hale getirip yeniden deneyin.")
        raise SystemExit()
    else:
        # NaN'leri 0 kabul edelim (isteğe göre 0/1 mapping ayarlanabilir)
        df[target_col] = df[target_col].fillna(0).astype(int)
        print(f"\n'{target_col}' kolonu başarıyla numerik formata çevrildi.")
        print(df[target_col].value_counts())
    # ===========================================================================

    print(f"\n--- Adım 2 Başladı: '{output_file_path}' okundu. ---")

    # 2. Değişkenleri tanımla
    exclude_cols = [target_col]

    # Sadece numerik 'feature' kolonlarını seç
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        print("--- HATA ---")
        print("Analiz edilecek numerik 'feature' kolonu bulunamadı.")
    else:
        print(f"Toplam {len(feature_cols)} adet numerik feature bulundu.")

        # 3. KORELASYON HESAPLAMA (failure ile)
        correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)

        # Korelasyonları mutlak değere göre sırala
        sorted_corr_indices = correlations.abs().sort_values(ascending=False).index
        sorted_correlations = correlations.loc[sorted_corr_indices]

        print("\n--- 'failure' Kolonu ile Korelasyon Sıralaması (Güçlüden Zayıfa) ---")
        print(sorted_correlations)
        print("------------------------------------------------------------------\n")

        # 4. GÖRSELLEŞTİRME (Boxplot Subplotları)

        # Korelasyonu en güçlü olan özellikleri (yukarıda sıraladık) kullan
        features_to_plot = sorted_correlations.index

        batch_size = 16  # Her bir PNG dosyasında 16 grafik (4x4 grid)
        num_plots = int(np.ceil(len(features_to_plot) / batch_size))

        plot_counter = 0  # Toplam kaç plot çizildiğini sayar

        for i in range(num_plots):
            # 4x4 grid (toplam 16 subplot) için bir figür oluştur
            fig, axes = plt.subplots(4, 4, figsize=(20, 18))

            # O anki grup (batch) için özellikleri seç
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            batch_features = features_to_plot[start_index:end_index]

            # Subplot'ları doldur
            for j, feature_name in enumerate(batch_features):
                ax = axes.flatten()[j]

                # Boxplot çizdir:
                sbn.boxplot(data=df, x=target_col, y=feature_name, ax=ax)

                # O subplot'un başlığına özelliğin adını ve korelasyon değerini yaz
                corr_value = sorted_correlations[feature_name]
                ax.set_title(f'{feature_name}\n(Corr: {corr_value:.3f})', fontsize=10)
                ax.set_xlabel('Failure Status')
                ax.set_ylabel('Feature Value')
                plot_counter += 1

            # Eğer son batch 16'dan azsa, boş kalan subplot'ları gizle
            remaining_plots = 16 - len(batch_features)
            if remaining_plots > 0:
                for k in range(remaining_plots):
                    axes.flatten()[15 - k].axis('off')

            # Ana başlık
            fig.suptitle(
                f"Feature Dağılımı vs. Failure (Grup {i + 1}/{num_plots})\n"
                f"(En Güçlü Korelasyondan Zayıfa Doğru Sıralı)",
                fontsize=18,
                y=1.03,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.98])

            file_name = f'failure_boxplot_group_{i + 1}.png'
            plt.savefig(file_name)
            plt.close(fig)

            print(f"'{file_name}' başarıyla kaydedildi.")

        print(f"\n--- Adım 2 Başarılı ---")
        print(f"Toplam {plot_counter} özellik için {num_plots} adet boxplot grafiği oluşturuldu.")

    # ==================== SCATTERPLOT BLOKU ====================
    try:
        print("\n--- Adım 2 (Scatterplot) Başladı ---")

        df_processed = reading_csv(output_file_path)

        # Aynı şekilde failure'ı numerik hale getirelim
        df_processed[target_col] = pd.to_numeric(df_processed[target_col], errors="coerce")
        df_processed[target_col] = df_processed[target_col].fillna(0).astype(int)

        exclude_cols = ['asset_id', 'id', 'train_test', 'status_type_id', target_col]

        # Numerik kolonları seç
        feature_cols = [
            col for col in df_processed.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_processed[col])
        ]

        if not feature_cols:
            print("--- HATA (Adım 2) ---")
            print("Analiz edilecek numerik 'feature' kolonu bulunamadı.")
        else:
            print(f"Toplam {len(feature_cols)} adet numerik feature bulundu.")

            # --- Korelasyon Hesapla ve Sırala ---
            correlations = df_processed[feature_cols + [target_col]].corr()[target_col].drop(target_col)
            sorted_features = correlations.abs().sort_values(ascending=False).index.tolist()

            # --- Scatterplotlar (4x4 grid, 16 grafik/grup) ---
            batch_size = 16
            num_plots = int(np.ceil(len(sorted_features) / batch_size))
            plot_counter = 0

            for i in range(num_plots):
                fig, axes = plt.subplots(4, 4, figsize=(20, 18))

                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                batch_features = sorted_features[start_index:end_index]

                for j, feature_name in enumerate(batch_features):
                    ax = axes.flatten()[j]

                    y = df_processed[target_col].astype(float)
                    x = df_processed[feature_name].astype(float)
                    y_jitter = y + np.random.uniform(-0.05, 0.05, size=len(y))

                    sbn.scatterplot(
                        x=x,
                        y=y_jitter,
                        hue=y,
                        palette={0: '#2ca02c', 1: '#d62728'},
                        ax=ax,
                        alpha=0.3,
                        legend=False,
                    )

                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(['0 (Normal)', '1 (Failure)'])
                    ax.set_title(f'{feature_name}\n(Corr: {correlations[feature_name]:.3f})', fontsize=10)
                    ax.set_xlabel('Feature Value')
                    ax.set_ylabel('Failure Status')
                    plot_counter += 1

                # Boş kalan subplot’ları gizle
                remaining = 16 - len(batch_features)
                for k in range(remaining):
                    axes.flatten()[15 - k].axis('off')

                fig.suptitle(
                    f"Feature Değeri vs. Failure (Grup {i + 1}/{num_plots})\n"
                    f"(En Güçlü Korelasyondan Zayıfa Doğru Sıralı)",
                    fontsize=18,
                    y=1.03,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.98])

                file_name = f'failure_scatterplot_group_{i + 1}.png'
                plt.savefig(file_name, dpi=200, bbox_inches='tight')
                plt.close(fig)

                print(f"'{file_name}' başarıyla kaydedildi.")

            print(f"\n--- Adım 2 (Scatter) Başarılı ---")
            print(f"Toplam {plot_counter} özellik için {num_plots} adet scatterplot grafiği oluşturuldu.")

    except Exception as e:
        print(f"Adım 2'de (Scatter Grafik Çizme) bir hata oluştu: {e}")

    # ==================== DISTRIBUTION (HISTOGRAM) BLOKU ====================
    try:
        print("\n--- Adım 3 Başladı: Distribution (Dağılım) Grafikleri ---")

        df_processed = reading_csv(output_file_path)
        df_processed[target_col] = pd.to_numeric(df_processed[target_col], errors="coerce")
        df_processed[target_col] = df_processed[target_col].fillna(0).astype(int)

        exclude_cols = ['asset_id', 'id', 'train_test', 'status_type_id', target_col]

        feature_cols = [
            col for col in df_processed.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_processed[col])
        ]

        if not feature_cols:
            print("--- HATA (Adım 3) ---")
            print("Dağılım grafiği çizilecek numerik 'feature' bulunamadı.")
        else:
            print(f"Toplam {len(feature_cols)} adet numerik feature bulundu.")

            correlations = df_processed[feature_cols + [target_col]].corr()[target_col].drop(target_col)
            sorted_features = correlations.abs().sort_values(ascending=False).index.tolist()

            batch_size = 16
            num_plots = int(np.ceil(len(sorted_features) / batch_size))
            plot_counter = 0

            for i in range(num_plots):
                fig, axes = plt.subplots(4, 4, figsize=(20, 18))

                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                batch_features = sorted_features[start_index:end_index]

                for j, feature_name in enumerate(batch_features):
                    ax = axes.flatten()[j]

                    sbn.histplot(
                        data=df_processed,
                        x=feature_name,
                        hue=target_col,
                        kde=True,
                        ax=ax,
                        element='step',
                        stat='density',
                        common_norm=False,
                        palette={0: '#2ca02c', 1: '#d62728'},
                        alpha=0.5,
                    )

                    ax.set_title(f'{feature_name}\n(Corr: {correlations[feature_name]:.3f})', fontsize=10)
                    ax.set_xlabel('Feature Value')
                    ax.set_ylabel('Density')
                    plot_counter += 1

                remaining = 16 - len(batch_features)
                for k in range(remaining):
                    axes.flatten()[15 - k].axis('off')

                fig.suptitle(
                    f"Feature Distribution by Failure Status (Grup {i + 1}/{num_plots})",
                    fontsize=18,
                    y=1.03,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.98])

                file_name = f'failure_distribution_group_{i + 1}.png'
                plt.savefig(file_name, dpi=200, bbox_inches=200)
                plt.close(fig)

                print(f"'{file_name}' başarıyla kaydedildi.")

            print(f"\n--- Adım 3 Başarılı ---")
            print(f"Toplam {plot_counter} özellik için {num_plots} adet dağılım grafiği oluşturuldu.")

    except Exception as e:
        print(f"Adım 3'te (Distribution Plot) bir hata oluştu: {e}")

    # ==================== GENEL BAR CHART (KORELASYON) ====================
    try:
        # 1. Dosyayı tekrar oku
        df_processed = reading_csv(output_file_path)
        df_processed[target_col] = pd.to_numeric(df_processed[target_col], errors="coerce")
        df_processed[target_col] = df_processed[target_col].fillna(0).astype(int)

        exclude_cols = ['asset_id', 'id', 'train_test', 'status_type_id', target_col]

        feature_cols = []
        for col in df_processed.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_processed[col]):
                feature_cols.append(col)

        if not feature_cols:
            print("--- HATA (Adım 2) ---")
            print("Korelasyonu hesaplanacak numerik 'feature' kolonu bulunamadı.")
        else:
            print(f"--- Adım 2 (BarChart) Başladı ---")
            print(f"Toplam {len(feature_cols)} adet numerik feature bulundu.")

            correlations = df_processed[feature_cols + [target_col]].corr()[target_col].drop(target_col)
            sorted_correlations = correlations.loc[correlations.abs().sort_values(ascending=False).index]

            print("\n--- 'failure' Kolonu ile Korelasyon Sıralaması (Metin) ---")
            print(sorted_correlations)

            data_to_plot = sorted_correlations.iloc[::-1]
            num_features = len(data_to_plot)
            fig_height = max(10, num_features * 0.4)

            plt.figure(figsize=(15, fig_height))

            colors = ['#d62728' if c < 0 else '#2ca02c' for c in data_to_plot.values]

            data_to_plot.plot(kind='barh', color=colors)

            plt.title(
                "'failure' Kolonu ile Diğer Tüm Özelliklerin Korelasyonu\n(Güçlüden Zayıfa Sıralı)",
                fontsize=18,
            )
            plt.xlabel("Korelasyon Katsayısı (Pearson)", fontsize=12)
            plt.ylabel("Feature Kolonları", fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            for index, value in enumerate(data_to_plot):
                x_pos = value + (0.01 * np.sign(value))
                ha = 'left' if value >= 0 else 'right'
                plt.text(x_pos, index, f' {value:.3f}', va='center', ha=ha, fontsize=9)

            max_abs_corr = max(abs(data_to_plot.max()), abs(data_to_plot.min()))
            plt.xlim([-(max_abs_corr * 1.2), max_abs_corr * 1.2])

            plt.tight_layout()

            chart_file_name = 'correlation_barchart_all_features.png'
            plt.savefig(chart_file_name)
            plt.close()

            print(f"\n--- Adım 2 (BarChart) Başarılı ---")
            print(f"Grafik başarıyla '{chart_file_name}' dosyasına kaydedildi.")

    except Exception as e:
        print(f"Adım 2'de (BarChart Çizme) bir hata oluştu: {e}")

    # ==================== HEATMAP & ALT KÜME HEATMAP ====================
    try:
        print("\n--- Adım 2 ve 3 (Heatmap) Başladı ---")
        df_processed = reading_csv(output_file_path)
        df_processed[target_col] = pd.to_numeric(df_processed[target_col], errors="coerce")
        df_processed[target_col] = df_processed[target_col].fillna(0).astype(int)

        exclude_cols = ['asset_id', 'id', 'train_test', 'status_type_id']

        numeric_cols = [
            col for col in df_processed.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_processed[col])
        ]

        # failure'ı mutlaka dahil et
        if target_col not in numeric_cols:
            numeric_cols.append(target_col)

        if not numeric_cols:
            print("HATA: Analiz edilecek numerik kolon bulunamadı.")
        else:
            print(f"Analiz için {len(numeric_cols)} adet numerik kolon bulundu.")
            df_features = df_processed[numeric_cols]

            # === KORELASYON HEATMAP ===
            print("\n--- Adım 2: Korelasyon Heatmap ---")
            corr_matrix = df_features.corr()

            # 1️⃣ KLASİK HEATMAP
            plt.figure(figsize=(16, 12), dpi=150)
            sbn.heatmap(
                corr_matrix,
                cmap='RdBu_r',
                annot=False,
                square=True,
                cbar_kws={"shrink": 0.5},
            )
            plt.title("Tüm Özelliklerin Korelasyon Matrisi", fontsize=16)
            plt.xticks(fontsize=6, rotation=90)
            plt.yticks(fontsize=6)
            plt.savefig("correlation_heatmap_all_features.png")
            plt.close()
            print("Klasik heatmap kaydedildi: correlation_heatmap_all_features.png")

            # 2️⃣ ALT KÜME HEATMAP (failure etrafı)
            if target_col in corr_matrix.columns:
                top_corr = corr_matrix[target_col].abs().sort_values(ascending=False).head(15).index
                subset_corr = corr_matrix.loc[top_corr, top_corr]

                plt.figure(figsize=(10, 8), dpi=150)
                sbn.heatmap(
                    subset_corr,
                    cmap='coolwarm',
                    annot=True,
                    fmt=".2f",
                    square=True,
                    cbar_kws={"shrink": 0.8},
                )
                plt.title("‘failure’ ile En İlişkili Özelliklerin Korelasyonu", fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                plt.savefig("correlation_heatmap_top_features.png")
                plt.close()
                print("Alt küme heatmap kaydedildi: correlation_heatmap_top_features.png")
            else:
                print("Uyarı: 'failure' korelasyon matrisi içinde bulunamadı (bu noktada olmaması beklenmez).")

    except Exception as e:
        print(f"Hata (Heatmap): {e}")

    # ==================== LINEPLOT TARZI KORELASYON ANALİZİ ====================
    try:
        print("\n--- Adım 2: Korelasyon Analizi (Lineplot Tarzı) ---")

        df_processed = reading_csv(output_file_path)
        df_processed[target_col] = pd.to_numeric(df_processed[target_col], errors="coerce")
        df_processed[target_col] = df_processed[target_col].fillna(0).astype(int)

        exclude_cols = ['asset_id', 'id', 'train_test', 'status_type_id']

        numeric_cols = [
            col for col in df_processed.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_processed[col])
        ]

        if target_col not in numeric_cols:
            numeric_cols.append(target_col)

        df_features = df_processed[numeric_cols]
        corr_matrix = df_features.corr()

        if target_col not in corr_matrix.columns:
            print(f"Uyarı: '{target_col}' korelasyon matrisinde bulunamadı.")
            raise SystemExit()

        top_corr = corr_matrix[target_col].sort_values(ascending=False)
        print(f"\n'{target_col}' ile korelasyonlar:\n{top_corr}\n")

        top_N = 12
        selected_features = top_corr.head(top_N).index.tolist()

        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()
        counter = 0

        # Burada amaç: korelasyon matrisi üzerindeki bazı ilişki kombinasyonlarını görselleştirmek
        for i in range(len(selected_features) - 1):
            for j in range(i + 1, len(selected_features)):
                if counter >= len(axes):
                    break

                x_col = selected_features[i]
                y_col = selected_features[j]
                ax = axes[counter]

                sbn.lineplot(
                    x=corr_matrix[x_col],
                    y=corr_matrix[y_col],
                    ax=ax,
                    color='teal',
                )
                ax.set_xlabel(x_col, fontsize=8)
                ax.set_ylabel(y_col, fontsize=8)
                ax.set_title(f"{x_col} vs {y_col}", fontsize=9)
                counter += 1

        plt.tight_layout()
        plt.suptitle(
            f"{target_col} ile En İlişkili Özelliklerin Korelasyon İlişkileri",
            fontsize=14,
            y=1.02,
        )
        plt.savefig("correlation_lineplots.png", dpi=150, bbox_inches="tight")

        print("Lineplot korelasyon grafikleri başarıyla oluşturuldu: correlation_lineplots.png")
        print("\n--- Tüm Adımlar Başarılı ---")

    except Exception as e:
        print(f"Hata (Lineplot): {e}")

    # ==================== DESCRIBE + İSTATİSTİK ÖZETİ ====================
    df_processed = reading_csv(output_file_path)
    df_processed[target_col] = pd.to_numeric(df_processed[target_col], errors="coerce")
    df_processed[target_col] = df_processed[target_col].fillna(0).astype(int)

    exclude_cols = ['asset_id', 'id', 'train_test', 'status_type_id']
    numeric_cols = [
        col for col in df_processed.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_processed[col])
    ]

    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    df_features = df_processed[numeric_cols]

    desc = df_features.describe().T

    # --- 4. Skewness & Kurtosis ---
    desc["skewness"] = df_features.apply(skew, axis=0)
    desc["kurtosis"] = df_features.apply(kurtosis, axis=0)

    def mape(y_true, y_pred, eps=1e-12):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

    def smape(y_true, y_pred, eps=1e-12):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = (np.abs(y_true) + np.abs(y_pred)) + eps
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & (denom > eps)
        if not np.any(mask):
            return np.nan
        return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0

    desc["mode"] = np.nan
    desc["median"] = np.nan
    desc["cv"] = np.nan
    desc["q1"] = np.nan
    desc["q3"] = np.nan
    desc["iqr"] = np.nan
    desc["mad"] = np.nan
    desc["mae"] = np.nan
    desc["sem"] = np.nan
    desc["range"] = np.nan
    desc["geometric_mean"] = np.nan
    desc["arithmetic_mean"] = np.nan
    desc["mape_to_mean_%"] = np.nan
    desc["smape_to_mean_%"] = np.nan

    confidence = 0.95
    desc["ci_lower"] = np.nan
    desc["ci_upper"] = np.nan

    for col in df_features.columns:
        data = df_features[col].dropna()
        n = len(data)
        if n > 1:
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            t_val = t.ppf((1 + confidence) / 2, df=n - 1)
            margin = t_val * (std / np.sqrt(n))
            desc.loc[col, "ci_lower"] = mean - margin
            desc.loc[col, "ci_upper"] = mean + margin
            desc.loc[col, "mode"] = mode(data, keepdims=True).mode[0]
            desc.loc[col, "median"] = np.median(data)
            desc.loc[col, "cv"] = variation(data, axis=0, nan_policy='omit')
            q1, q3 = np.percentile(data, [25, 75])
            desc.loc[col, "q1"] = q1
            desc.loc[col, "q3"] = q3
            desc.loc[col, "iqr"] = q3 - q1
            desc.loc[col, "mad"] = median_abs_deviation(data, nan_policy='omit')
            desc.loc[col, "mae"] = np.mean(np.abs(data - np.mean(data)))
            desc.loc[col, "sem"] = np.std(data, ddof=1) / np.sqrt(len(data))
            desc.loc[col, "range"] = np.max(data) - np.min(data)
            desc.loc[col, "geometric_mean"] = gmean(data)
            desc.loc[col, "arithmetic_mean"] = np.mean(data)
            desc.loc[col, "mape_to_mean_%"] = mape(data, np.full_like(data, mean))
            desc.loc[col, "smape_to_mean_%"] = smape(data, np.full_like(data, mean))

    def interpret_skew(value):
        if value > 0.5:
            return "Sağa çarpık (ortalama > medyan)"
        elif value < -0.5:
            return "Sola çarpık (ortalama < medyan)"
        else:
            return "Normale yakın (simetrik)"

    def interpret_kurt(value):
        if value > 3:
            return "Leptokurtik (uç değerlere sahip)"
        elif value < 3:
            return "Platykurtik (basık dağılım)"
        else:
            return "Mesokurtik (normale yakın)"

    def interpret_mape(value):
        if not np.isfinite(value):
            return "Yetersiz veri"
        if value < 10:
            return "Çok düşük bağıl hata"
        elif value < 20:
            return "Düşük bağıl hata"
        elif value < 50:
            return "Orta bağıl hata"
        else:
            return "Yüksek bağıl hata"

    def interpret_smape(value):
        if not np.isfinite(value):
            return "Yetersiz veri"
        if value < 10:
            return "Çok düşük simetrik bağıl hata"
        elif value < 20:
            return "Düşük simetrik bağıl hata"
        elif value < 50:
            return "Orta simetrik bağıl hata"
        else:
            return "Yüksek simetrik bağıl hata"

    def interpret_cv(value):
        if not np.isfinite(value):
            return "Yetersiz veri"
        if value < 0.1:
            return "Çok düşük varyans"
        elif value < 0.3:
            return "Düşük varyans"
        elif value < 0.5:
            return "Orta varyans"
        else:
            return "Yüksek varyans"

    def interpret_sem(sem, mean):
        if not np.isfinite(sem) or not np.isfinite(mean) or mean == 0:
            return "Yetersiz veri"
        rel_sem = sem / abs(mean)
        if rel_sem < 0.05:
            return "Çok güvenilir ortalama"
        elif rel_sem < 0.1:
            return "Güvenilir ortalama"
        elif rel_sem < 0.2:
            return "Orta güvenilir ortalama"
        else:
            return "Düşük güvenilir ortalama"

    def interpret_mae(mae, std):
        if not np.isfinite(mae) or not np.isfinite(std):
            return "Yetersiz veri"
        ratio_mae = mae / (std + 1e-12)
        if ratio_mae < 0.5:
            return "Hata büyüklüğü düşük (std'nin altında)"
        elif ratio_mae < 1.0:
            return "Hata büyüklüğü orta (std seviyesinde)"
        else:
            return "Hata büyüklüğü yüksek (std'nin üstünde)"

    desc["Skewness Yorumu"] = desc["skewness"].apply(interpret_skew)
    desc["Kurtosis Yorumu"] = desc["kurtosis"].apply(interpret_kurt)
    desc["MAPE Yorumu"] = desc["mape_to_mean_%"].apply(interpret_mape)
    desc["sMAPE Yorumu"] = desc["smape_to_mean_%"].apply(interpret_smape)
    desc["CV Yorumu"] = desc["cv"].apply(interpret_cv)
    desc["SEM Yorumu"] = desc.apply(lambda row: interpret_sem(row["sem"], row["mean"]), axis=1)
    desc["MAE Yorumu"] = desc.apply(lambda row: interpret_mae(row["mae"], row["std"]), axis=1)

    def general_comment(row):
        mean, std = row["mean"], row["std"]
        if std > abs(mean) * 0.5:
            return "Değerler geniş aralıkta dağılmış (yüksek varyans)"
        else:
            return "Değerler ortalamaya yakın (düşük varyans)"

    desc["Genel Dağılım Yorumu"] = desc.apply(general_comment, axis=1)

    # ---7. Sensör açıklamaları ve birimlerini excel'e ekle---
    fm = pd.read_csv("feature_description.csv", delimiter=";", engine="python")
    fm = fm.rename(columns={"sensor_name": "feature_name"}).loc[:, ["feature_name", "description", "unit"]]

    def canon(s):
        s = str(s).strip().lower()
        if s.endswith('_avg'):
            s = s[:-4]
        return s

    desc_ = desc.reset_index().rename(columns={"index": "feature_name"})
    desc_["_k"] = desc_["feature_name"].apply(canon)
    fm["_k"] = fm["feature_name"].apply(canon)

    desc = desc_.merge(fm[["_k", "description", "unit"]], on="_k", how="left").drop(columns="_k")
    desc = desc[
        ["feature_name", "description", "unit"]
        + [c for c in desc.columns if c not in ["feature_name", "description", "unit"]]
    ]

    output_excel = "numeric_summary_with_ci.xlsx"
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        desc.to_excel(writer, sheet_name="Numeric_Stats")

    print(f"\n📘 Analiz başarıyla '{output_excel}' dosyasına kaydedildi!\n")

    print(
        desc[
            [
                "mean",
                "std",
                "ci_lower",
                "ci_upper",
                "skewness",
                "kurtosis",
                "Skewness Yorumu",
                "Kurtosis Yorumu",
                "Genel Dağılım Yorumu",
            ]
        ].head(10)
    )

except FileNotFoundError:
    print(f"--- HATA ---")
    print(f"Hata: '{output_file_path}' dosyası bulunamadı.")
    print("Lütfen bu script'i ilgili CSV dosyasının olduğu dizinde çalıştırdığınızdan emin olun.")
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")
