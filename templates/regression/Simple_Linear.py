"""
╔══════════════════════════════════════════════════════════════╗
║  📈 BASİT DOĞRUSAL REGRESYON                                 ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: Tek sebep → Tek sonuç (Düz çizgi ilişkisi)         ║
║  📥 GİRDİ: X_train, y_train (Bellekten)                      ║
║  📤 ÇIKTI: y_pred (Tahminler), R² skoru, Grafik              ║
║  ⏱️  NE ZAMAN KULLAN:                                         ║
║     • Tek bir bağımsız değişken varsa                        ║
║     • İlişki düz çizgi gibi görünüyorsa                      ║
║  📊 ÖRNEK: Reklam Harcaması → Satış Miktarı                  ║
╚══════════════════════════════════════════════════════════════╝

⚠️  UYARI: Bu dosyayı çalıştırmadan ÖNCE mutlaka:
   %run 01_veri_on_isleme.py
   komutunu çalıştır! (X_train, y_train bellekte olmalı)

✅ ÇALIŞTIR: %run 02_regression/simple_linear.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("=" * 60)
print("📈 BASİT DOĞRUSAL REGRESYON")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1️⃣ MODELİ OLUŞTUR
# ─────────────────────────────────────────────────────────────
regressor = LinearRegression()
print("\n✅ Model oluşturuldu: Linear Regression")

# ─────────────────────────────────────────────────────────────
# 2️⃣ MODELİ EĞİT
# ─────────────────────────────────────────────────────────────
regressor.fit(X_train, y_train)
print("✅ Model eğitildi")

# ─────────────────────────────────────────────────────────────
# 3️⃣ TAHMİN YAP
# ─────────────────────────────────────────────────────────────
y_pred = regressor.predict(X_test)
print("✅ Tahminler yapıldı")

# ─────────────────────────────────────────────────────────────
# 4️⃣ SONUÇLARI GÖSTER
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📊 SONUÇLAR")
print("─" * 60)

# R² Skoru (1'e yakın = mükemmel, 0'a yakın = kötü)
r2 = r2_score(y_test, y_pred)
print(f"\n🎯 R² Başarı Skoru: {r2:.4f}")

if r2 > 0.9:
    print("   → 🏆 Mükemmel! Model çok iyi çalışıyor")
elif r2 > 0.7:
    print("   → ✅ İyi! Kabul edilebilir seviye")
elif r2 > 0.5:
    print("   → ⚠️  Orta! Başka model deneyebilirsin")
else:
    print("   → ❌ Zayıf! Başka model dene (Polynomial, SVR...)")

# İlk 5 tahmin vs gerçek
print("\n📋 İlk 5 Tahmin vs Gerçek:")
print("─" * 40)
for i in range(min(5, len(y_test))):
    print(f"{i+1}. Tahmin: {y_pred[i][0]:.2f} | Gerçek: {y_test.iloc[i, 0]:.2f}")

# ─────────────────────────────────────────────────────────────
# 5️⃣ GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("\n📊 Grafik çiziliyor...")

plt.figure(figsize=(10, 6))

# Eğitim verisini sırala (Çizgi düzgün olsun)
X_train_sorted_indices = X_train.ravel().argsort()
X_train_sorted = X_train[X_train_sorted_indices]
y_train_sorted = y_train.iloc[X_train_sorted_indices]

# Noktalar (Gerçek veri)
plt.scatter(X_train_sorted, y_train_sorted, color='red', 
            label='Eğitim Verisi', alpha=0.6, s=50)

# Tahmin çizgisi
plt.plot(X_train_sorted, regressor.predict(X_train_sorted), 
         color='blue', linewidth=2, label='Tahmin Çizgisi')

plt.title(f'Basit Doğrusal Regresyon (R² = {r2:.3f})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Bağımsız Değişken (X)', fontsize=12)
plt.ylabel('Hedef Değişken (y)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Grafik gösterildi")

# ─────────────────────────────────────────────────────────────
# 📝 SONRAKİ ADIMLAR
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("💡 SONRAKİ ADIMLAR")
print("=" * 60)
print("\nEğer R² skoru düşükse şunları dene:")
print("   1️⃣  Polynomial Regression: %run 02_regression/polynomial.py")
print("   2️⃣  SVR (Karmaşık kıvrımlar için): %run 02_regression/svr.py")
print("   3️⃣  Random Forest (En güçlü): %run 02_regression/random_forest.py")
print("=" * 60)
