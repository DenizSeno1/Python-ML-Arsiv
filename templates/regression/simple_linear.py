import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- BU ŞABLON NE İŞE YARAR? ---
# Elindeki temizlenmiş, eğitim ve test olarak bölünmüş (X_train, y_train...)
# verileri alıp makineye öğretir ve tahmin yaptırır.

# 1. MODELİ SEÇ VE KUR (Linear Regression)
# Burası "Beyin" kısmıdır. İleride DecisionTree, RandomForest gelirse burası değişir.
regressor = LinearRegression()

# 2. MODELİ EĞİT (FIT)
# Modeli çalıştırır; X (Sorular) ile y (Cevaplar) arasındaki matematiksel ilişkiyi öğrenir.
regressor.fit(X_train, y_train)

# 3. TAHMİN YAP (PREDICT)
# Modeli kullanarak, görmediği test verileri (X_test) için sonuç üretir.
y_pred = regressor.predict(X_test)

# --- SONUÇLARI GÖR ---
print("Tahminler (y_pred):")
print(y_pred)
print("\nGerçek Değerler (y_test):")
print(y_test)


# --- 4. GRAFİK ÇİZME (GÖRSELLEŞTİRME) ---
# Sadece tek değişkenli verilerde (Örn: Sadece Aylar vs Satışlar) çalışır.
# Çoklu veri varsa (Yaş, Boy, Kilo...) bu kısım hata verebilir, dikkat!

# Verileri sırala ki çizgiler zig-zag yapmasın (Opsiyonel ama önerilir)
X_train_sort = X_train.sort_index()
y_train_sort = y_train.sort_index()

plt.scatter(X_train_sort, y_train_sort, color='red', label='Gerçek Veri')
plt.plot(X_train_sort, regressor.predict(X_train_sort), color='blue', label='Tahmin Çizgisi')
plt.title("Tahmin Modeli Grafiği")
plt.xlabel("Bağımsız Değişken (X)")
plt.ylabel("Hedef Değişken (y)")
plt.legend()
plt.show()

# --- 5. BAŞARI PUANI (R-SQUARED) ---
# Modelin başarısını ölçer. 1'e ne kadar yakınsa o kadar iyidir.
print("R-Kare Başarı Puanı:")
print(r2_score(y_test, y_pred))
