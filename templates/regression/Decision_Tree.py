"""
╔══════════════════════════════════════════════════════════════╗
║  🌳 KARAR AĞACI REGRESYONU                                   ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: Basamaklı/Aralıklı tahminler                       ║
║  📥 GİRDİ: X_train, y_train (Bellekten)                      ║
║  📤 ÇIKTI: y_pred, R² skoru, Basamaklı grafik                ║
║  ⏱️  NE ZAMAN KULLAN:                                         ║
║     • Veri kategorik aralıklara bölünebiliyorsa              ║
║     • Örn: Maaş aralıkları, yaş grupları                     ║
║  ℹ️  ÖZELLİK: Scaling gerektirmez (ama zaten yaptık)         ║
╚══════════════════════════════════════════════════════════════╝

⚠️  UYARI: Bu dosyayı çalıştırmadan ÖNCE:
   %run 01_veri_on_isleme.py

✅ ÇALIŞTIR: %run 02_regression/decision_tree.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

print("=" * 60)
print("🌳 KARAR AĞACI REGRESYONU")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1️⃣ MODELİ OLUŞTUR VE EĞİT
# ─────────────────────────────────────────────────────────────
# random_state=0: Her çalıştırmada aynı sonuç
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X_train, y_train)

print("\n✅ Karar Ağacı modeli eğitildi")

# ─────────────────────────────────────────────────────────────
# 2️⃣ TAHMİN YAP
# ─────────────────────────────────────────────────────────────
y_pred = dt_model.predict(X_test)
print("✅ Tahminler yapıldı")

# ─────────────────────────────────────────────────────────────
# 3️⃣ BAŞARI SKORU
# ─────────────────────────────────────────────────────────────
r2 = r2_score(y_test, y_pred)
print("\n" + "─" * 60)
print("📊 SONUÇLAR")
print("─" * 60)
print(f"\n🎯 R² Başarı Skoru: {r2:.4f}")

if r2 > 0.9:
    print("   → 🏆 Mükemmel! Model çok iyi çalışıyor")
elif r2 > 0.7:
    print("   → ✅ İyi! Kabul edilebilir seviye")
elif r2 > 0.5:
    print("   → ⚠️  Orta! Random Forest daha iyi olabilir")
else:
    print("   → ❌ Zayıf! Başka model dene")

# İlk 5 tahmin vs gerçek
print("\n📋 İlk 5 Tahmin vs Gerçek:")
print("─" * 40)
for i in range(min(5, len(y_test))):
    print(f"{i+1}. Tahmin: {y_pred[i]:.2f} | Gerçek: {y_test.iloc[i, 0]:.2f}")

# ─────────────────────────────────────────────────────────────
# 4️⃣ GÖRSELLEŞTİRME (Yüksek Çözünürlük Gerekli!)
# ─────────────────────────────────────────────────────────────
print("\n📊 Grafik çiziliyor...")
print("   ℹ️  Karar ağacı basamaklı yapıdadır")

plt.figure(figsize=(10, 6))

# Veriyi sırala
sort_indices = X_train.ravel().argsort()
X_train_sorted = X_train[sort_indices]
y_train_sorted = y_train.iloc[sort_indices]

# Gerçek veri noktaları
plt.scatter(X_train_sorted, y_train_sorted, color='red', 
            alpha=0.6, s=50, label='Gerçek Veri')

# Yüksek çözünürlüklü grid oluştur (Basamaklar düzgün görünsün)
X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

# Decision Tree tahmini
plt.plot(X_grid, dt_model.predict(X_grid), color='blue', 
         linewidth=2, label='Karar Ağacı Tahmini')

plt.title(f'Karar Ağacı Regresyonu (R² = {r2:.3f})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Bağımsız Değişken (X)', fontsize=12)
plt.ylabel('Hedef Değişken (y)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Grafik gösterildi")
print("   ℹ️  Mavi çizgi basamaklı → Bu normal!")

# ─────────────────────────────────────────────────────────────
# 📝 SONRAKİ ADIMLAR
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("💡 SONRAKİ ADIMLAR")
print("=" * 60)
print("\n🚀 Daha iyi sonuç için:")
print("   • Random Forest (10 ağaç birden): %run 02_regression/random_forest.py")
print("     → Tek ağaçtan daha hassas!")
print("=" * 60)
