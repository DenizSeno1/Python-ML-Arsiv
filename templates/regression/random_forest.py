import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- BU ŞABLON NE İŞE YARAR? ---
# Random Forest: Birden fazla (n_estimators) Karar Ağacı oluşturup,
# bunların tahminlerinin ortalamasını alır. "Kolektif Zeka" mantığıdır.

# 1. MODELİ KUR VE EĞİT
# n_estimators=10: Arka planda 10 tane farklı ağaç kurar. Sayı artarsa işlem uzar ama hassasiyet artabilir.
# random_state=0: Sonuç sabit kalsın diye.
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)

# y_train.ravel(): Veriyi tek boyutlu diziye çevirir (DataConversionWarning yememek için).
rf_reg.fit(X_train, y_train.ravel())


# 2. TAHMİN YAP
y_pred = rf_reg.predict(X_test)

# --- SONUÇLARI GÖR ---
print("--- Tahminler (y_pred) ---")
print(y_pred)


# 3. GÖRSELLEŞTİRME (Yüksek Çözünürlüklü)
# Random Forest da basamaklı yapıdadır (Ağaçların ortalaması olduğu için).
# Düz çizgi çizdirmemek için araları dolduruyoruz (Grid).

# Veri aralığını 0.01 hassasiyetle parçala
X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X_train, y_train, color='red', label='Gerçek Veri')
plt.plot(X_grid, rf_reg.predict(X_grid), color='blue', label='Random Forest (10 Ağaç)')

plt.title('Random Forest Regresyonu')
plt.xlabel('Bağımsız Değişken (X)')
plt.ylabel('Hedef Değişken (y)')
plt.legend()
plt.show()


# 4. BAŞARI PUANI (R-SQUARED)
# Genelde Decision Tree'den daha yüksek çıkar çünkü tek bir ağacın hatasını diğerleri kapatır.
print("\nR-Kare Başarı Puanı:")
print(r2_score(y_test, y_pred))
