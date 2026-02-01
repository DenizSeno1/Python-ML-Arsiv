import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# --- BU ŞABLON NE İŞE YARAR? ---
# Hafızadaki X_train ve y_train verilerini alıp Karar Ağacı oluşturur.
# DİKKAT: Decision Tree aslında Scaling (Ölçekleme) istemez.
# Ancak 'Sablon_Veri_On_Isleme.py' dosyasından X_train ölçekli geliyorsa
# model mecburen ölçekli veriyi öğrenir. Sonuç değişmez, sadece eksen sayıları değişir.

# 1. MODELİ KUR VE EĞİT
# random_state=0: Her çalıştırdığında aynı sonuç çıksın diye.
r_dt = DecisionTreeRegressor(random_state=0)

# Modeli eğit
r_dt.fit(X_train, y_train)


# 2. TAHMİN YAP
y_pred = r_dt.predict(X_test)

# --- SONUÇLARI GÖR ---
print("--- Tahminler (y_pred) ---")
print(y_pred)


# 3. GÖRSELLEŞTİRME (KRİTİK BÖLÜM!)
# Karar ağacı, aralıklara göre sabit değer atar. 
# Düz çizgi çizmemesi için araya sıklaştırılmış noktalar (Grid) koyuyoruz.
# Yoksa grafik yanlış (zikzaklı) görünür.

# Veri aralığını alıp 0.01 hassasiyetle parçalıyoruz
X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1)) # Sütun haline getir

plt.scatter(X_train, y_train, color='red', label='Gerçek Veri')

# Tahmin çizgisini bu yüksek çözünürlüklü grid üzerinden çizdiriyoruz
plt.plot(X_grid, r_dt.predict(X_grid), color='blue', label='Karar Ağacı')

plt.title('Karar Ağacı Regresyonu (Basamaklı Yapı)')
plt.xlabel('Bağımsız Değişken (X)')
plt.ylabel('Hedef Değişken (y)')
plt.legend()
plt.show()


# 4. BAŞARI PUANI (R-SQUARED)
print("\nR-Kare Başarı Puanı:")
print(r2_score(y_test, y_pred))
