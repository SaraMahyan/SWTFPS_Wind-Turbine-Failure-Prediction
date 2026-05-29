import pandas as pd
import numpy as np

def create_master_labeled_dataset(input_csv, output_csv):
    print("Veri okunuyor ve Master Etiketleme Algoritması başlatılıyor...\n")
    df = pd.read_csv(input_csv)
    
    # Başlangıçta tüm sisteme NORMAL (0) diyoruz
    df['failure'] = 0
    
    # =========================================================================
    # KATEGORİ 1: SENSÖR VE HABERLEŞME HATALARI
    # =========================================================================
    # 1. Ölü Sensör: Sıcaklık sensörlerinden herhangi biri 0 okuyorsa (Kopuk kablo)
    cond_dead_temp = (df['Temperature_1'] == 0) | (df['Temperature_2'] == 0) | (df['Temperature_3'] == 0)
    
    # 2. İmkansız Rüzgar: Anemometrede glitch (okuma hatası) varsa
    cond_wind_glitch = (df['Wind_Speed_kmh'] > 150)
    
    # =========================================================================
    # KATEGORİ 2: ELEKTRİKSEL ARIZALAR
    # =========================================================================
    # 3. Açık Devre (Kablo Kopması): Rüzgar var, voltaj üretiliyor ama hiç akım çekilemiyor
    cond_open_circuit = (df['Wind_Speed_kmh'] > 15) & (df['Turbine_Voltage'] > 1.0) & (df['Turbine_Current'] <= 0.05)
    
    # 4. Kısa Devre & Yangın Başlangıcı: Voltaj çöktü, Akım fırladı, Gaz sensörü duman algıladı
    cond_short_circuit = (df['Turbine_Current'] > 0.70) & (df['Turbine_Voltage'] < 0.5) & (df['Gas_Value'] > 400)
    
    # 5. Hayalet Güç (Sensör Paraziti): Rüzgar hiç esmiyor (veya çok az) ama yüksek voltaj var
    cond_ghost_power = (df['Wind_Speed_kmh'] < 2.0) & (df['Turbine_Voltage'] > 2.0)
    
    # =========================================================================
    # KATEGORİ 3: MEKANİK VE AERODİNAMİK ARIZALAR
    # =========================================================================
    # 6. Rulman/Dişli Sürtünmesi: Gearbox ortamdan 15°C daha sıcak ve sistem aşırı titriyor
    cond_friction = ((df['Temperature_3'] - df['Temperature_1']) > 15) & (df['Vibration_Value'] > 1500)
    
    # 7. Kanat Kırılması / Balanssızlık: Rüzgar şiddetli, titreşim çok yüksek ama üretilen güç komik derecede düşük
    cond_blade_damage = (df['Wind_Speed_kmh'] > 20) & (df['Vibration_Value'] > 2000) & (df['Turbine_Power'] < 0.2)
    
    # 8. Jeneratör Sıkışması (Mekanik Kilit): Rüzgar çok iyi esiyor ama pervaneyi döndüremiyor (Voltaj 0)
    cond_mechanical_block = (df['Wind_Speed_kmh'] > 15) & (df['Turbine_Voltage'] < 0.1)
    
    # =========================================================================
    # KATEGORİ 4: TERMAL ARIZALAR (AŞIRI ISINMA)
    # =========================================================================
    # 9. Jeneratör Zorlanması: Sistem yüksek güç üretiyor ama motor (Nacelle) tehlikeli seviyede ısınıyor
    cond_generator_stress = (df['Turbine_Power'] > 1.0) & (df['Temperature_2'] > 45)
    
    # 10. Kritik Aşırı Isınma: İçerideki parçalar 50°C'yi aşmış
    cond_overheat = (df['Temperature_2'] > 50) | (df['Temperature_3'] > 50)

    # =========================================================================
    # ARIZALARI BİRLEŞTİR VE UYGULA
    # =========================================================================
    # Bütün koşulları VEYA (|) operatörü ile birleştiriyoruz. Herhangi biri olursa Target = 1 olur.
    all_failures = (cond_dead_temp | cond_wind_glitch | cond_open_circuit | 
                    cond_short_circuit | cond_ghost_power | cond_friction | 
                    cond_blade_damage | cond_mechanical_block | 
                    cond_generator_stress | cond_overheat)
    
    df.loc[all_failures, 'failure'] = 1

    # --- RAPORLAMA EKRANI ---
    print("=" * 55)
    print(" 🛠️ MASTER ETİKETLEME RAPORU (Kapsamlı Analiz)")
    print("=" * 55)
    print(f"Toplam İncelenen Satır: {len(df)}")
    print(f"✅ NORMAL Çalışma (0): {len(df[df['failure'] == 0])} satır")
    print(f"❌ TESPİT EDİLEN ARIZA (1): {len(df[df['failure'] == 1])} satır\n")
    
    print("--- Arıza Türlerine Göre Dağılım ---")
    print("(Not: Bir satırda birden fazla arıza tetiklenmiş olabilir)")
    print(f"[Sensör] Kopuk Sıcaklık Sensörü : {cond_dead_temp.sum()}")
    print(f"[Sensör] İmkansız Rüzgar Hızı   : {cond_wind_glitch.sum()}")
    print(f"[Elektrik] Açık Devre / Kopuk   : {cond_open_circuit.sum()}")
    print(f"[Elektrik] Kısa Devre & Duman   : {cond_short_circuit.sum()}")
    print(f"[Elektrik] Hayalet Güç Paraziti : {cond_ghost_power.sum()}")
    print(f"[Mekanik] Rulman Sürtünmesi     : {cond_friction.sum()}")
    print(f"[Mekanik] Kanat/Balans Hasarı   : {cond_blade_damage.sum()}")
    print(f"[Mekanik] Jeneratör Sıkışması   : {cond_mechanical_block.sum()}")
    print(f"[Termal] Jeneratör Zorlanması   : {cond_generator_stress.sum()}")
    print(f"[Termal] Kritik Aşırı Isınma    : {cond_overheat.sum()}")
    print("=" * 55)
    
    # Dosyayı Kaydet
    df.to_csv(output_csv, index=False)
    print(f"\nMuhteşem! Yepyeni ve kapsamlı veri setin kaydedildi: {output_csv}")

# Kullanım
if __name__ == "__main__":
    # Okunacak ham veri csv'sinin adını ve çıkacak yeni csv'nin adını buraya yazıyorsun
    create_master_labeled_dataset('combined_sensor_data.csv', 'master_labeled_data.csv')