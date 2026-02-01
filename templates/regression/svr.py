import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- DİKKAT ---
# Bu şablonu kullanmadan önce "Sablon_Veri_On_Isleme.py" dosyasını çalıştırıp
# hafızaya X_train, X_test, y_train, y_test değişkenlerini yüklemiş olmalısın!

# --- BU ŞABLON NE İŞE YARAR? ---
# Support Vector Regression (SVR) ile kıvrımlı ilişkileri modellemek için.
# ÖNEMLİ: SVR hem X hem y verilerinin ölçeklenmesini ister!

# 1. HEDEF DEĞİŞKENİ (y) ÖLÇEKLEME
# X_train zaten Veri Ön İşleme'de ölçeklendi, şimdi y'yi de ölçekliyoruz.
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1))

# 2. MODELİ KUR VE EĞİT
# kernel='rbf': Kıvrımlı veriler için en iyi seçenek
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train, y_train_scaled.ravel())

# 3. TAHMİN YAP
y_pred_scaled = svr_reg.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# --- SONUÇLARI GÖR ---
print("Tahminler (y_pred):")
print(y_pred)
print("\nGerçek Değerler (y_test):")
print(y_test)


# --- 4. GRAFİK ÇİZME (GÖRSELLEŞTİRME) ---
# Veriyi sırala ki çizgi düzgün çıksın
sort_index = X_train.ravel().argsort()
X_train_sorted = X_train[sort_index]
y_train_sorted = y_train[sort_index]

# Tahmin çizgisini oluştur
tahmin_scaled = svr_reg.predict(X_train_sorted)
tahmin_real = sc_y.inverse_transform(tahmin_scaled.reshape(-1, 1))

plt.scatter(X_train_sorted, y_train_sorted, color='red', label='Gerçek Veri')
plt.plot(X_train_sorted, tahmin_real, color='blue', label='SVR Tahmini')
plt.title("SVR Regresyon Grafiği")
plt.xlabel("Bağımsız Değişken (X)")
plt.ylabel("Hedef Değişken (y)")
plt.legend()
plt.show()

# --- 5. BAŞARI PUANI (R-SQUARED) ---
print("R-Kare Başarı Puanı:")
print(r2_score(y_test, y_pred))
