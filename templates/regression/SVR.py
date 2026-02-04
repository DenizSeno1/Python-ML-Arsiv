"""
╔══════════════════════════════════════════════════════════════╗
║  🔮 SUPPORT VECTOR REGRESSION (SVR)                          ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: Karmaşık, kıvrımlı ilişkileri modellemek           ║
║  📥 GİRDİ: X_train, y_train (Bellekten)                      ║
║  📤 ÇIKTI: y_pred, R² skoru, Eğri grafik                     ║
║  ⏱️  NE ZAMAN KULLAN:                                         ║
║     • Polinom yeterli gelmiyorsa                             ║
║     • Veri çok karmaşık ve kıvrımlıysa                       ║
║  📊 ÖRNEK: Borsa tahmini, karmaşık fizik formülleri          ║
║  ⚠️  DİKKAT: Hem X hem y ölçeklendirilmeli!                  ║
╚══════════════════════════════════════════════════════════════╝

⚠️  UYARI: Bu dosyayı çalıştırmadan ÖNCE:
   %run 01_veri_on_isleme.py
   (X_train zaten ölçekli, y_train'i burada ölçekleriz)

✅ ÇALIŞTIR: %run 02_regression/svr.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

print("=" * 60)
print("🔮 SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1️⃣ HEDEF DEĞİŞKENİ (y) ÖLÇEKLEME
# ─────────────────────────────────────────────────────────────
# SVR için y de ölçeklenmeli (X zaten veri ön işlemede ölçeklendi)
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train.values.reshape(-1, 1))

print("\n✅ y_train ölçeklendirildi")
print("   ℹ️  SVR için hem X hem y ölçekli olmalı")

# ─────────────────────────────────────────────────────────────
# 2️⃣ MODELİ OLUŞTUR VE EĞİT
# ─────────────────────────────────────────────────────────────
# kernel='rbf': Radial Basis Function - Kıvrımlı veriler için en iyi
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train_scaled.ravel())

print("✅ SVR modeli eğitildi (Kernel: RBF)")

# ─────────────────────────────────────────────────────────────
# 3️⃣ TAHMİN YAP (Ölçekli → Gerçek değere dönüştür)
# ─────────────────────────────────────────────────────────────
y_pred_scaled = svr_model.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

print("✅ Tahminler yapıldı ve gerçek ölçeğe döndürüldü")

# ─────────────────────────────────────────────────────────────
# 4️⃣ BAŞARI SKORU
# ─────────────────────────────────────────────────────────────
r2 = r2_score(y_test, y_pred)
print("\n" + "─" * 60)
print("📊 SONUÇLAR")
print("─" * 60)
print(f"\n🎯 R² Başarı Skoru: {r2:.4f}")

if r2 > 0.9:
    print("   → 🏆 Mükemmel! SVR çok iyi çalıştı")
elif r2 > 0.7:
    print("   → ✅ İyi! Kabul edilebilir seviye")
elif r2 > 0.5:
    print("   → ⚠️  Orta! Random Forest daha iyi olabilir")
else:
    print("   → ❌ Zayıf! Veri SVR için uygun değil, başka model dene")

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

# Veriyi sırala
sort_indices = X_train.ravel().argsort()
X_train_sorted = X_train[sort_indices]
y_train_sorted = y_train.iloc[sort_indices]

# Gerçek veri noktaları
plt.scatter(X_train_sorted, y_train_sorted, color='red', 
            alpha=0.6, s=50, label='Gerçek Veri')

# SVR tahmini (Ölçeklenmiş tahmini gerçeğe çevir)
y_svr_scaled = svr_model.predict(X_train_sorted)
y_svr_pred = sc_y.inverse_transform(y_svr_scaled.reshape(-1, 1))

plt.plot(X_train_sorted, y_svr_pred, color='blue', 
         linewidth=2, label='SVR Tahmini')

plt.title(f'SVR Regresyon (R² = {r2:.3f})', 
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
print("\n🔧 SVR Parametrelerini Ayarla:")
print("   • Dosyanın 39. satırında SVR(kernel='rbf') var")
print("   • Farklı kerneller dene: 'linear', 'poly', 'sigmoid'")
print("\n🚀 Daha güçlü model:")
print("   • Random Forest (Genelde en iyi): %run 02_regression/random_forest.py")
print("=" * 60)
