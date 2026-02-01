import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- DİKKAT ---
# Bu dosyayı çalıştırmadan önce "Sablon_Veri_On_Isleme.py" dosyasını çalıştırıp
# hafızaya X_train, X_test, y_train, y_test değişkenlerini yüklemiş olmalısın!

# --- BU ŞABLON NE İŞE YARAR? ---
# Çoklu Doğrusal Regresyon (Multiple Linear Regression) modelini eğitir.
# Birden fazla sebep (X) ile bir sonucu (y) tahmin etmeye çalışır.

# 1. MODELİ SEÇ VE KUR
regressor = LinearRegression()

# 2. MODELİ EĞİT (FIT)
# Modeli elindeki eğitim verileriyle (X_train, y_train) besle.
regressor.fit(X_train, y_train)

# 3. TAHMİN YAP (PREDICT)
# Test setindeki soruları modele sor, bakalım ne tahmin edecek?
y_pred = regressor.predict(X_test)

# --- SONUÇLARI GÖR ---
# Pandas Series formatındaysa .values ile temizleyip yan yana görebilirsin
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# --- 4. GRAFİK ÇİZME (GÖRSELLEŞTİRME) ---
# DİKKAT: Çoklu regresyonda (Birden fazla X sütunu varsa) klasik çizgi grafiği ÇİZİLEMEZ.
# Çünkü 3-4 boyutlu bir düzlemi ekrana sığdıramazsın.
# O yüzden burada "Gerçek Değerler vs Tahmin Değerleri" grafiği çiziyoruz.

plt.scatter(y_test, y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Tam uyum çizgisi
plt.title("Gerçek vs Tahmin (Ne kadar çizgiye yakınsa o kadar iyi)")
plt.xlabel("Gerçek Değerler (y_test)")
plt.ylabel("Tahmin Edilenler (y_pred)")
plt.show()

# --- 5. BAŞARI PUANI (R-SQUARED) ---
# Modelin başarısını ölçer. 1'e ne kadar yakınsa model o kadar "kral"dır.
print("R-Kare Başarı Puanı:")
print(r2_score(y_test, y_pred))


# --- 6. DETAYLI İSTATİSTİK RAPORU (OLS) ---
# Hangi değişkenin işe yaradığını, hangisinin çöp olduğunu görmek için.
# LinearRegression() bize p-value vermez, o yüzden statsmodels kullanıyoruz.

# Statsmodels için başa "1"lerden oluşan sütun (Bias/Sabit) eklenmesi gerekir.
# (Bunu Veri Ön İşleme aşamasında yaptıysan tekrar yapma)
import numpy as np
X_train_ols = np.append(arr=np.ones((len(X_train), 1)).astype(int), values=X_train, axis=1)

# Modeli kur ve özeti bas
model_ols = sm.OLS(y_train, X_train_ols).fit()
print(model_ols.summary())
