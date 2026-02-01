import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# --- DİKKAT ---
# Bu şablon hafızadaki X_train, X_test, y_train, y_test değişkenlerini kullanır.

# 1. POLİNOM DÖNÜŞÜMÜ
# Veriyi bükmek için karesini, küpünü alıyoruz.
# degree=4: Hassasiyet derecesi.
poly_reg = PolynomialFeatures(degree=4)

# Eğitim verisini hem öğren (fit) hem dönüştür (transform)
x_train_poly = poly_reg.fit_transform(X_train)

# Test verisini sadece dönüştür (Asla fit yapma, kopya çekmesin)
x_test_poly = poly_reg.transform(X_test)


# 2. MODELİ KUR VE EĞİT
# Artık elimizde polinomlanmış veri var, LinearRegression'a bunu veriyoruz.
regressor = LinearRegression()
regressor.fit(x_train_poly, y_train)


# 3. TAHMİN YAP
y_pred = regressor.predict(x_test_poly)

# --- SONUÇLARI GÖR ---
print("--- Tahminler (y_pred) ---")
print(y_pred)


# 4. GÖRSELLEŞTİRME (Pürüzsüz Eğri İçin)
# X_train verisini sıralamazsak çizgi karman çorman olur.
# Bu yüzden görselleştirme için sıralı bir liste ve grid oluşturuyoruz.

# Veriyi sırala (Görsel düzgün çıksın diye)
X_train_sort_index = X_train.ravel().argsort()
X_train_sorted = X_train[X_train_sort_index]
y_train_sorted = y_train[X_train_sort_index]

# Tahmin çizgisini çiz
plt.scatter(X_train_sorted, y_train_sorted, color='red', label='Gerçek (Eğitim)')
plt.plot(X_train_sorted, regressor.predict(poly_reg.fit_transform(X_train_sorted)), color='blue', label='Polinom Modeli')

plt.title("Polinom Regresyon Sonucu")
plt.xlabel("Bağımsız Değişken (X)")
plt.ylabel("Hedef Değişken (y)")
plt.legend()
plt.show()


# 5. BAŞARI PUANI (R-SQUARED)
print("\nR-Kare Başarı Puanı:")
print(r2_score(y_test, y_pred))
