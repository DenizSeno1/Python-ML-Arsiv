import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. VERİ YÜKLEME ---
# Dosya adını kendine göre değiştir!
df = pd.read_csv("veriler.csv")

# --- 2. EKSİK VERİLERİ DOLDURMA (IMPUTER) ---
# Sayısal verilerde eksik (NaN) varsa ortalama ile doldurur.
# Örnek: 1. ile 4. sütunlar arasını doldurmak için:
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
yas = df.iloc[:, 1:4].values
imputer = imputer.fit(yas[:, 1:4])
yas[:, 1:4] = imputer.transform(yas[:, 1:4])

# --- 3. YAZIYI SAYIYA ÇEVİRME (ENCODING) ---
# Label Encoding (Sıralı veriler veya 2 seçenekli durumlar için)
le = preprocessing.LabelEncoder()
cinsiyet = df.iloc[:, -1].values
cinsiyet = le.fit_transform(cinsiyet) # Sonuç: [0, 1, 0, 1...]

# One Hot Encoding (Ülke gibi sıralamanın önemsiz olduğu durumlar için)
ohe = preprocessing.OneHotEncoder()
ulke = df.iloc[:, 0:1].values
ulke = ohe.fit_transform(ulke).toarray() # Sonuç: [[1,0,0], [0,1,0]...]

# --- 4. VERİLERİ BİRLEŞTİRME (DATAFRAME) ---
# Numpy dizilerini tekrar tablo (DataFrame) yapıp birleştirme
sonuc = pd.DataFrame(data=ulke, index=range(len(df)), columns=["fr","tr","us"])
sonuc2 = pd.DataFrame(data=yas, index=range(len(df)), columns=["boy","kilo","yas"])
sonuc3 = pd.DataFrame(data=cinsiyet, index=range(len(df)), columns=["cinsiyet"])

# Tek bir X (Bağımsız Değişkenler) tablosu oluşturma
s = pd.concat([sonuc, sonuc2], axis=1) 

# --- 5. EĞİTİM VE TEST OLARAK BÖLME (SPLIT) ---
# Verinin %33'ü test, %67'si eğitim
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

# --- 6. ÖLÇEKLEME (SCALING) ---
# Sayıları aynı hizaya getirme (Standartlaştırma)
sc = StandardScaler()
X_train = sc.fit_transform(x_train) # Eğitimi öğren ve uygula
X_test = sc.transform(x_test)       # Testi sadece uygula!
