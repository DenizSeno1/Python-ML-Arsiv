import statsmodels.api as sm
import numpy as np

# --- BACKWARD ELIMINATION ŞABLONU ---

# 1. Sabit Değer (Bias) Ekleme ve Float Dönüşümü
# (X ve y değişkenlerinin yukarıda tanımlı olduğundan emin ol)
X_opt = np.append(arr=np.ones((len(X), 1)).astype(int), values=X.astype(float), axis=1)

print("--- Otomatik Eleme Başlıyor ---")

while True:
    # OLS Modelini Oluştur
    model = sm.OLS(y, X_opt).fit()
    
    # P değerlerini kontrol et
    p_values = model.pvalues
    max_p = max(p_values)
    max_index = list(p_values).index(max_p)
    
    # P değeri 0.05'ten büyükse o değişkeni at
    if max_p > 0.05:
        print(f"🗑️ Atıldı: Index {max_index} (P-value: {max_p:.4f})")
        X_opt = np.delete(X_opt, max_index, axis=1)
    else:
        print("✅ Temizlik Bitti! Kalan değişkenler ideal.")
        print(model.summary())
        break

# SONUÇ: Artık elinde en temiz verilerle dolu "X_opt" var.
