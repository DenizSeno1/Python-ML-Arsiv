"""
╔══════════════════════════════════════════════════════════════╗
║  🌲 RANDOM FOREST REGRESYONU                                 ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: Çok ağacın kolektif zekası (En güçlü model!)       ║
║  📥 GİRDİ: X_train, y_train (Bellekten)                      ║
║  📤 ÇIKTI: y_pred, R² skoru, Grafik                          ║
║  ⏱️  NE ZAMAN KULLAN:                                         ║
║     • En hassas sonuç istiyorsan                             ║
║     • Genelde en iyi performansı verir                       ║
║  🔧 AYAR: n_estimators = Ağaç sayısı (10-100 arası)          ║
╚══════════════════════════════════════════════════════════════╝

⚠️  UYARI: Bu dosyayı çalıştırmadan ÖNCE:
   %run 01_veri_on_isleme.py

✅ ÇALIŞTIR: %run 02_regression/random_forest.py

🔧 AYARLAR:
   • n_estimators=10: Ağaç sayısı
     - Fazla ağaç → Yavaş ama hassas
     - Az ağaç → Hızlı ama az hassas
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

print("=" * 60)
print("🌲 RANDOM FOREST REGRESYONU")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1️⃣ MODELİ OLUŞTUR VE EĞİT
# ─────────────────────────────────────────────────────────────
N_TREES = 10  # 🔧 AĞAÇ SAYISI (10-100 arası dene)

rf_model = RandomForestRegressor(
    n_estimators=N_TREES,
    random_state=0
)

rf_model.fit(X_train, y_train.values.ravel())
print(f"\n✅ Random Forest modeli eğitildi ({N_TREES} ağaç)")

# ─────────────────────────────────────────────────────────────
# 2️⃣ TAHMİN YAP
# ─────────────────────────────────────────────────────────────
y_pred = rf_model.predict(X_test)
print("✅ Tahminler yapıldı")

# ─────────────────────────────────────────────────────────────
# 3️⃣ BAŞARI SKORU
# ─────────────────────────────────────────────────────────────
r2 = r2_score(y_test, y_pred)
print("\n" + "─" * 60)
print("📊 SONUÇLAR")
print("─" * 60)
print(f"\n🎯 R² Başarı Skoru: {r2:.4f}")

if r2 > 0.95:
    print("   → 🏆🏆🏆 Müthiş! Mükemmel sonuç!")
elif r2 > 0.9:
    print("   → 🏆 Çok iyi! Model harika çalışıyor")
elif r2 > 0.7:
    print("   → ✅ İyi! Kabul edilebilir seviye")
elif r2 > 0.5:
    print("   → ⚠️  Orta! Ağaç sayısını artır (n_estimators)")
else:
    print("   → ❌ Zayıf! Veri problemi olabilir, veri ön işlemeyi kontrol et")

# İlk 5 tahmin vs gerçek
print("\n📋 İlk 5 Tahmin vs Gerçek:")
print("─" * 40)
for i in range(min(5, len(y_test))):
    print(f"{i+1}. Tahmin: {y_pred[i]:.2f} | Gerçek: {y_test.iloc[i, 0]:.2f}")

# ─────────────────────────────────────────────────────────────
# 4️⃣ ÖZELLİK ÖNEMİ (Feature Importance)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔍 ÖZELLİK ÖNEMİ")
print("─" * 60)
feature_importances = rf_model.feature_importances_
print("\nHangi özellikler en önemli:")
for i, importance in enumerate(feature_importances):
    print(f"   Özellik {i}: {importance:.4f} ({importance*100:.1f}%)")

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

# Yüksek çözünürlüklü grid
X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

# Random Forest tahmini
plt.plot(X_grid, rf_model.predict(X_grid), color='blue', 
         linewidth=2, label=f'Random Forest ({N_TREES} Ağaç)')

plt.title(f'Random Forest Regresyonu (R² = {r2:.3f})', 
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
print(f"\n🔧 Performansı artır:")
print(f"   • Dosyanın 39. satırında N_TREES = {N_TREES}")
print(f"   • N_TREES = {N_TREES*5} veya {N_TREES*10} yapıp tekrar çalıştır")
print(f"     (Daha yavaş ama daha hassas olur)")
print("\n💾 Modeli kaydet:")
print("   import joblib")
print("   joblib.dump(rf_model, 'my_model.pkl')")
print("\n📊 Diğer modellerle karşılaştır:")
print("   • R² skorlarını yan yana koy ve en iyisini seç!")
print("=" * 60)
