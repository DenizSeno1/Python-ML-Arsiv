"""
╔══════════════════════════════════════════════════════════════╗
║  🧹 BACKWARD ELIMINATION (Değişken Temizleyici)              ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: Gereksiz değişkenleri tespit edip atmak            ║
║  📥 GİRDİ: X_train, y_train (Bellekten)                      ║
║  📤 ÇIKTI: X_opt (Temizlenmiş veri), İstatistik raporu       ║
║  ⏱️  NE ZAMAN KULLAN:                                         ║
║     • Multiple Linear Regression'dan SONRA                   ║
║     • R² skoru düşükse                                       ║
║     • Hangi değişkenlerin gereksiz olduğunu öğrenmek için    ║
║  ⚠️  SADECE: Multiple Linear Regression için!                ║
╚══════════════════════════════════════════════════════════════╝

⚠️  UYARI: Bu dosyayı çalıştırmadan ÖNCE:
   1️⃣  %run 01_veri_on_isleme.py
   2️⃣  %run 02_regression/multiple_linear.py

✅ ÇALIŞTIR: %run utils/backward_elimination.py

💡 NASIL ÇALIŞIR:
   1. En yüksek P-value'ya sahip değişkeni bulur
   2. P > 0.05 ise o değişkeni atar
   3. Kalan değişkenlerle tekrar model kurar
   4. Tüm P-value'lar < 0.05 olana kadar tekrar eder
"""

import numpy as np
import statsmodels.api as sm

print("=" * 60)
print("🧹 BACKWARD ELIMINATION")
print("=" * 60)
print("\nGereksiz değişkenleri otomatik olarak temizliyorum...")
print("─" * 60)

# ─────────────────────────────────────────────────────────────
# 1️⃣ SABİT TERİM (Bias) EKLE
# ─────────────────────────────────────────────────────────────
# Statsmodels OLS için başa 1'lerden oluşan sütun ekle
X_opt = np.append(
    arr=np.ones((len(X_train), 1)).astype(int),
    values=X_train.astype(float),
    axis=1
)

print(f"\n✅ Başlangıç: {X_opt.shape[1]} değişken")

# ─────────────────────────────────────────────────────────────
# 2️⃣ OTOMATİK ELİMİNASYON DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────
iteration = 0

while True:
    iteration += 1
    
    # OLS modelini kur
    model = sm.OLS(y_train, X_opt).fit()
    
    # P-value'ları kontrol et
    p_values = model.pvalues
    max_p = max(p_values)
    max_index = list(p_values).index(max_p)
    
    # P > 0.05 ise değişkeni at
    if max_p > 0.05:
        print(f"\n🗑️  İterasyon {iteration}:")
        print(f"   • Atılan değişken: Index {max_index}")
        print(f"   • P-value: {max_p:.6f} (> 0.05)")
        print(f"   • Kalan değişken sayısı: {X_opt.shape[1] - 1}")
        
        X_opt = np.delete(X_opt, max_index, axis=1)
    else:
        print(f"\n✅ Temizlik Bitti! ({iteration-1} değişken atıldı)")
        break

# ─────────────────────────────────────────────────────────────
# 3️⃣ FİNAL RAPORU
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 FİNAL İSTATİSTİK RAPORU")
print("=" * 60)
print(model.summary())

# ─────────────────────────────────────────────────────────────
# 4️⃣ SONUÇ VE TAVSİYELER
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("💡 SONUÇ VE TAVSİYELER")
print("=" * 60)

remaining_features = X_opt.shape[1] - 1  # -1 (Sabit terimi sayma)
original_features = X_train.shape[1]
removed_features = original_features - remaining_features

print(f"\n📊 Özet:")
print(f"   • Başlangıç değişken sayısı: {original_features}")
print(f"   • Atılan değişken sayısı: {removed_features}")
print(f"   • Kalan değişken sayısı: {remaining_features}")

if removed_features > 0:
    print(f"\n✅ {removed_features} gereksiz değişken tespit edildi ve atıldı!")
    print("\n📝 Şimdi ne yapmalısın:")
    print("   1️⃣  Veri ön işleme dosyasına dön")
    print("   2️⃣  Gereksiz sütunları çıkar")
    print("   3️⃣  Modeli tekrar çalıştır")
    print("   4️⃣  R² skorunun arttığını gör! 🎉")
else:
    print("\n🎉 Tüm değişkenler gerekli! Veri zaten optimize.")

print("\n" + "─" * 60)
print("💾 TEMİZLENMİŞ VERİ:")
print("   • X_opt değişkeni bellekte hazır")
print("   • Bunu modelinde kullanabilirsin:")
print("\n   from sklearn.linear_model import LinearRegression")
print("   model = LinearRegression()")
print("   model.fit(X_opt, y_train)")

# ─────────────────────────────────────────────────────────────
# 📝 SONRAKİ ADIMLAR
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🚀 SONRAKİ ADIMLAR")
print("=" * 60)
print("\n1️⃣  Veri ön işleme dosyasını düzenle:")
print("   • Gereksiz sütunları iloc'tan çıkar")
print("   • Örn: iloc[:, 1:4] → iloc[:, 1:3]")
print("\n2️⃣  Multiple Linear Regression'ı tekrar çalıştır:")
print("   %run 02_regression/multiple_linear.py")
print("\n3️⃣  R² skorunu kontrol et:")
print("   • Önceki: Düşük")
print("   • Sonraki: Daha yüksek! 📈")
print("=" * 60)
