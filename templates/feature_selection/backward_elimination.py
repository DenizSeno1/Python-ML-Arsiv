import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- 1. HAZIRLIK ---
# Elindeki X verisinin başına 1'lerden oluşan sütunu (Sabit Değer) ekliyoruz
# X: Senin daha önce oluşturduğun [UnvanSeviyesi, Kidem, Puan] matrisi
X_l = np.append(arr=np.ones((len(X), 1)).astype(int), values=X, axis=1)

# X_l artık: [Sabit(1), UnvanSeviyesi, Kidem, Puan]

print("--- Temizlik Başlıyor ---")

# --- 2. OTOMATİK ELEME DÖNGÜSÜ ---
while True:
    # OLS modelini kur (y: Maaş)
    model = sm.OLS(y, X_l).fit()
    
    # P değerlerini al
    p_values = model.pvalues
    
    # En yüksek P değerini bul
    max_p = max(p_values)
    max_index = list(p_values).index(max_p)
    
    if max_p > 0.05:
        print(f"⚠️ ATILDI: {max_index}. sıradaki değişken (P-value: {max_p:.4f})")
        # O sütunu diziden sil
        X_l = np.delete(X_l, max_index, axis=1)
    else:
        print("✅ Temizlik Bitti! Kalan değişkenler temiz.")
        print("\n--- SON DURUM RAPORU ---")
        print(model.summary())
        break

# --- 3. TEMİZLENMİŞ VERİYLE TEST ---
print("\n--- Final Modeli (Linear Regression) ---")

# Temizlenen X_l ile yeni train/test ayrımı
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(X_l, y, test_size=0.33, random_state=0)

lm = LinearRegression()
lm.fit(x_train_new, y_train_new)

print("Tahminler:", lm.predict(x_test_new))
