"""
╔══════════════════════════════════════════════════════════════╗
║  📦 VERİ ÖN İŞLEME ŞABLONU                                   ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: Ham veriyi modele hazır hale getirmek              ║
║  📥 GİRDİ: CSV dosyası (veriler.csv)                         ║
║  📤 ÇIKTI: X_train, X_test, y_train, y_test (Bellekte)       ║
║  ⏱️  NE ZAMAN: Her ML projesinde İLK ADIM                    ║
║  🔗 SONRASI: Herhangi bir regresyon modeli                   ║
╚══════════════════════════════════════════════════════════════╝

📝 KULLANIM ÖNCESİ AYARLAMALAR:
1. Satır 23: "veriler.csv" → Kendi dosya adını yaz
2. Satır 29-30: iloc[:, 1:4] → Kendi sütun aralıklarını ayarla
3. Satır 35-36: iloc[:, -1] ve iloc[:, 0:1] → Hedef ve kategorik sütunları ayarla

✅ ÇALIŞTIR: python 01_veri_on_isleme.py
   veya Jupyter'da: %run 01_veri_on_isleme.py
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

print("=" * 60)
print("🔧 VERİ ÖN İŞLEME BAŞLIYOR...")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1️⃣ VERİYİ YÜKLE
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("veriler.csv")
print(f"\n✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")

# ─────────────────────────────────────────────────────────────
# 2️⃣ EKSİK VERİLERİ DOLDUR (Ortalama ile)
# ─────────────────────────────────────────────────────────────
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
numeric_data = df.iloc[:, 1:4].values
numeric_data = imputer.fit_transform(numeric_data)
print("✅ Eksik veriler ortalama ile dolduruldu")

# ─────────────────────────────────────────────────────────────
# 3️⃣ KATEGORİK VERİLERİ SAYIYA ÇEVİR
# ─────────────────────────────────────────────────────────────
# Label Encoding (Cinsiyet gibi ikili değerler için)
le = LabelEncoder()
target = df.iloc[:, -1].values  # 🔧 Hedef değişken (y) - Son sütun
target = le.fit_transform(target)
print("✅ Hedef değişken encode edildi")

# One-Hot Encoding (Ülke gibi çoklu kategoriler için)
ohe = OneHotEncoder()
categorical = df.iloc[:, 0:1].values  # 🔧 Kategorik sütun
categorical = ohe.fit_transform(categorical).toarray()
print("✅ Kategorik değişkenler One-Hot encode edildi")

# ─────────────────────────────────────────────────────────────
# 4️⃣ VERİLERİ BİRLEŞTİR
# ─────────────────────────────────────────────────────────────
# Tüm özellikleri (features) tek bir DataFrame'de topla
df_categorical = pd.DataFrame(data=categorical, 
                              index=range(len(df)), 
                              columns=[f"cat_{i}" for i in range(categorical.shape[1])])

df_numeric = pd.DataFrame(data=numeric_data, 
                          index=range(len(df)), 
                          columns=[f"num_{i}" for i in range(numeric_data.shape[1])])

df_target = pd.DataFrame(data=target, 
                        index=range(len(df)), 
                        columns=["target"])

# X (Bağımsız değişkenler) ve y (Hedef) oluştur
X = pd.concat([df_categorical, df_numeric], axis=1)
y = df_target

print(f"✅ Veri birleştirildi: X shape = {X.shape}")

# ─────────────────────────────────────────────────────────────
# 5️⃣ EĞİTİM VE TEST OLARAK BÖL (%67 eğitim, %33 test)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.33, 
    random_state=0
)

print(f"✅ Veri bölündü:")
print(f"   • Eğitim seti: {X_train.shape[0]} satır")
print(f"   • Test seti: {X_test.shape[0]} satır")

# ─────────────────────────────────────────────────────────────
# 6️⃣ ÖLÇEKLENDİRME (Feature Scaling)
# ─────────────────────────────────────────────────────────────
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Eğitim: Öğren ve uygula
X_test = sc.transform(X_test)        # Test: Sadece uygula (öğrenme!)

print("✅ Veriler ölçeklendirildi (Standardization)")

# ─────────────────────────────────────────────────────────────
# ✅ HAZIR! Şimdi istediğin modeli çalıştırabilirsin
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉 VERİ ÖN İŞLEME TAMAMLANDI!")
print("=" * 60)
print("\n📦 Bellekte hazır değişkenler:")
print("   • X_train, X_test (Bağımsız değişkenler)")
print("   • y_train, y_test (Hedef değişken)")
print("\n▶️  Şimdi bir regresyon modeli çalıştırabilirsin!")
print("   Örnek: %run 02_regression/simple_linear.py")
print("=" * 60)
