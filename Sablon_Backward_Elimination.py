import statsmodels.api as sm
import numpy as np

# --- 1. SABİT DEĞER (BIAS) EKLEME ---
# Statsmodels kütüphanesi, formüldeki sabit sayıyı (y = ax + b'deki b)
# otomatik eklemez. O yüzden en başa "1"lerden oluşan bir sütun ekleriz.
X = np.append(arr=np.ones((len(veri_seti), 1)).astype(int), values=veri_seti, axis=1)

# --- 2. OTOMATİK ELEME DÖNGÜSÜ ---
# Bu döngü:
# 1. Modeli kurar.
# 2. En yüksek P değerine sahip değişkeni bulur.
# 3. Eğer P değeri 0.05'ten büyükse o değişkeni siler.
# 4. Herkes masum (P < 0.05) olana kadar devam eder.

X_l = veri_seti.iloc[:, :].values # Başlangıçta tüm sütunları al
X_l = np.array(X_l, dtype=float)

print("--- Temizlik Başlıyor ---")

while True:
    # OLS (Ordinary Least Squares) modelini kur
    # hedef_degisken: Senin y (sonuç) verin
    model = sm.OLS(hedef_degisken, X_l).fit()
    
    # P değerlerini al
    p_values = model.pvalues
    
    # En yüksek P değerini ve sırasını bul
    max_p = max(p_values)
    max_index = list(p_values).index(max_p)
    
    if max_p > 0.05:
        print(f"ATILDI: {max_index}. sıradaki değişken (P-value: {max_p:.4f})")
        # O sütunu diziden sil
        X_l = np.delete(X_l, max_index, axis=1)
    else:
        print("--- Temizlik Bitti! Kalan Değişkenlerle Devam Ediliyor ---")
        break

print("\n--- SON DURUM RAPORU ---")
print(model.summary())

# --- 3. TEMİZLENMİŞ VERİYLE YENİ MODEL EĞİTİMİ ---
# Bunu kullanarak Linear Regression modelini tekrar eğitebilirsin.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Yeni X (Temizlenmiş) ve y ile tekrar split yap
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(X_l, hedef_degisken, test_size=0.33, random_state=0)

regressor_new = LinearRegression()
regressor_new.fit(x_train_new, y_train_new)

print("\nYeni Modelin Tahminleri:")
print(regressor_new.predict(x_test_new))
