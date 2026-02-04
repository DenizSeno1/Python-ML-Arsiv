"""
╔══════════════════════════════════════════════════════════════╗
║  🌀 POLİNOM REGRESYON                                        ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: Kıvrımlı ilişkileri yakalamak (x², x³, x⁴...)      ║
║  📥 GİRDİ: X_train, y_train (Bellekten)                      ║
║  📤 ÇIKTI: y_pred, R² skoru, Eğri grafik                     ║
║  ⏱️  NE ZAMAN KULLAN:                                         ║
║     • Linear model yetersiz kaldıysa                         ║
║     • İlişki kıvrımlı ama yumuşaksa                          ║
║  📊 ÖRNEK: Sıcaklık → Dondurma Satışı (Parabol gibi)         ║
╚══════════════════════════════════════════════════════════════╝

⚠️  UYARI: Bu dosyayı çalıştırmadan ÖNCE:
   %run 01_veri_on_isleme.py

✅ ÇALIŞTIR: %run 02_regression/polynomial.py

🔧 AYARLAR:
   • degree=4: Polinom derecesi (2-6 arası dene)
     - 2: Parabol (x²)
     - 3: Kübik (x³)
     - 4: Daha karmaşık kıvrımlar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

print("=" * 60)
print("🌀 POLİNOM REGRESYON")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1️⃣ POLİNOM DÖNÜŞÜMÜ (x → x, x², x³, x⁴...)
# ─────────────────────────────────────────────────────────────
DEGREE = 4  # 🔧 BURADAN DERECEYİ DEĞİŞTİREBİLİRSİN

poly_transformer = PolynomialFeatures(degree=DEGREE)

# Eğitim verisini dönüştür
X_train_poly = poly_transformer.fit_transform(X_train)

# Test verisini dönüştür (fit_transform DEĞİL, sadece transform!)
X_test_poly = poly_transformer.transform(X_test)

print(f"\n✅ Polinom dönüşümü yapıldı (Derece: {DEGREE})")
print(f"   • Önceki özellik sayısı: {X_train.shape[1]}")
print(f"   • Yeni özellik sayısı: {X_train_poly.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 2️⃣ MODELİ OLUŞTUR VE EĞİT
# ─────────────────────────────────────────────────────────────
regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)
print("✅ Model eğitildi")

# ─────────────────────────────────────────────────────────────
# 3️⃣ TAHMİN YAP
# ─────────────────────────────────────────────────────────────
y_pred = regressor.predict(X_test_poly)
print("✅ Tahminler yapıldı")

# ─────────────────────────────────────────────────────────────
# 4️⃣ BAŞARI SKORU
# ─────────────────────────────────────────────────────────────
r2 = r2_score(y_test, y_pred)
print("\n" + "─" * 60)
print("📊 SONUÇLAR")
print("─" * 60)
print(f"\n🎯 R² Başarı Skoru: {r2:.4f}")

if r2 > 0.9:
    print("   → 🏆 Mükemmel! Polinom çok iyi çalıştı")
elif r2 > 0.7:
    print("   → ✅ İyi! Kabul edilebilir seviye")
elif r2 > 0.5:
    print("   → ⚠️  Orta! Derece değerini artır veya azalt")
    print(f"   → 💡 degree={DEGREE} yerine {DEGREE+1} veya {DEGREE-1} dene")
else:
    print("   → ❌ Zayıf! Daha güçlü model dene (SVR, Random Forest)")

# İlk 5 tahmin vs gerçek
print("\n📋 İlk 5 Tahmin vs Gerçek:")
print("─" * 40)
for i in range(min(5, len(y_test))):
    print(f"{i+1}. Tahmin: {y_pred[i][0]:.2f} | Gerçek: {y_test.iloc[i, 0]:.2f}")

# ─────────────────────────────────────────────────────────────
# 5️⃣ GÖRSELLEŞTİRME (Pürüzsüz Eğri)
# ─────────────────────────────────────────────────────────────
print("\n📊 Grafik çiziliyor...")

plt.figure(figsize=(10, 6))

# Veriyi sırala (Eğri düzgün çıksın)
sort_indices = X_train.ravel().argsort()
X_train_sorted = X_train[sort_indices]
y_train_sorted = y_train.iloc[sort_indices]

# Gerçek veri noktaları
plt.scatter(X_train_sorted, y_train_sorted, color='red', 
            alpha=0.6, s=50, label='Gerçek Veri')

# Polinom tahmini (Eğri çizgi)
X_poly_sorted = poly_transformer.transform(X_train_sorted)
y_poly_pred = regressor.predict(X_poly_sorted)

plt.plot(X_train_sorted, y_poly_pred, color='blue', 
         linewidth=2, label=f'Polinom Tahmini (Derece {DEGREE})')

plt.title(f'Polinom Regresyon (R² = {r2:.3f})', 
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
print(f"\n🔧 Derece ayarını dene:")
print(f"   • Dosyanın 32. satırında DEGREE = {DEGREE}")
print(f"   • DEGREE = {DEGREE+1} veya {DEGREE-1} yapıp tekrar çalıştır")
print("\n🚀 Daha güçlü modeller:")
print("   • SVR: %run 02_regression/svr.py")
print("   • Random Forest: %run 02_regression/random_forest.py")
print("=" * 60)
