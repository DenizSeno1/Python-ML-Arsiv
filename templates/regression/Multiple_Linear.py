"""
╔══════════════════════════════════════════════════════════════╗
║  📊 ÇOKLU DOĞRUSAL REGRESYON                                 ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: Çok sebep → Tek sonuç (Çok boyutlu düzlem)         ║
║  📥 GİRDİ: X_train, y_train (Bellekten)                      ║
║  📤 ÇIKTI: y_pred, R² skoru, İstatistik raporu               ║
║  ⏱️  NE ZAMAN KULLAN:                                         ║
║     • Birden fazla bağımsız değişken varsa                   ║
║     • Örn: Yaş + Boy + Kilo → Sağlık Skoru                   ║
║  🔧 SONRASI: Backward Elimination ile iyileştirilebilir      ║
╚══════════════════════════════════════════════════════════════╝

⚠️  UYARI: Bu dosyayı çalıştırmadan ÖNCE:
   %run 01_veri_on_isleme.py

✅ ÇALIŞTIR: %run 02_regression/multiple_linear.py
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("=" * 60)
print("📊 ÇOKLU DOĞRUSAL REGRESYON")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1️⃣ MODELİ OLUŞTUR VE EĞİT
# ─────────────────────────────────────────────────────────────
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("\n✅ Model eğitildi")

# ─────────────────────────────────────────────────────────────
# 2️⃣ TAHMİN YAP
# ─────────────────────────────────────────────────────────────
y_pred = regressor.predict(X_test)
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
    print("   → ⚠️  Orta! Gereksiz değişkenler olabilir")
    print("   → 💡 utils/backward_elimination.py çalıştır!")
else:
    print("   → ❌ Zayıf! Linear ilişki yok, başka model dene")

# ─────────────────────────────────────────────────────────────
# 4️⃣ TAHMIN vs GERÇEK GRAFİĞİ
# ─────────────────────────────────────────────────────────────
print("\n📊 Grafik çiziliyor...")

plt.figure(figsize=(10, 6))

# Gerçek vs Tahmin scatter
plt.scatter(y_test, y_pred, color='red', alpha=0.6, s=50, 
            label='Tahminler')

# Mükemmel tahmin çizgisi (45 derece)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 
         'k--', linewidth=2, label='Mükemmel Tahmin')

plt.title(f'Gerçek vs Tahmin (R² = {r2:.3f})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Gerçek Değerler', fontsize=12)
plt.ylabel('Tahmin Edilen Değerler', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Grafik gösterildi")
print("   ℹ️  Noktalar çizgiye yakınsa → İyi tahmin")
print("   ℹ️  Noktalar dağınıksa → Zayıf tahmin")

# ─────────────────────────────────────────────────────────────
# 5️⃣ DETAYLI İSTATİSTİK RAPORU (Statsmodels OLS)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📋 DETAYLI İSTATİSTİK RAPORU")
print("=" * 60)

# Sabit terimi (intercept) ekle
X_train_ols = np.append(arr=np.ones((len(X_train), 1)).astype(int), 
                        values=X_train, axis=1)

# OLS modeli
model_ols = sm.OLS(y_train, X_train_ols).fit()
print(model_ols.summary())

print("\n💡 P-VALUE YORUMLAMA:")
print("   • P < 0.05 → Değişken ÖNEMLİ (Kalsın)")
print("   • P > 0.05 → Değişken GEREKSİZ (Atılabilir)")
print("\n   ⚠️  Gereksiz değişkenler varsa:")
print("      %run utils/backward_elimination.py")

# ─────────────────────────────────────────────────────────────
# 📝 SONRAKİ ADIMLAR
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("💡 SONRAKİ ADIMLAR")
print("=" * 60)
print("\n1️⃣  Gereksiz değişkenleri temizle:")
print("   %run utils/backward_elimination.py")
print("\n2️⃣  R² düşükse başka model dene:")
print("   %run 02_regression/polynomial.py")
print("   %run 02_regression/random_forest.py")
print("=" * 60)
