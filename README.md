# 🧠 Python & Machine Learning Arşivim

Bu depo (repository), Veri Bilimi ve Makine Öğrenmesi öğrenim sürecimde oluşturduğum notları, hazır kod şablonlarını ve projeleri içerir.

İlerideki projelerimde "tekerleği yeniden icat etmemek" ve hızlıca kod parçalarını alıp kullanmak için kişisel bir kütüphane amacı taşır.

---

## 📂 İçerik

### 1. Veri Ön İşleme (Data Preprocessing)
Veriyi temizleme, dönüştürme ve analize hazır hale getirme adımları.
- **Eksik Veriler (Missing Values):** `SimpleImputer` ile ortalama/medyan doldurma.
- **Encoding:** Kategorik verileri sayısal forma çevirme (Label & One-Hot Encoding).
- **Scaling:** Verileri standartlaştırma (StandardScaler).
- **Splitting:** Veri setini Eğitim (Train) ve Test setlerine ayırma.

### 2. Makine Öğrenmesi Modelleri
Şu ana kadar uygulanan ve şablon haline getirilen algoritmalar:

#### 📈 Regresyon (Tahmin) Modelleri
- **Linear Regression:** Basit ve Çoklu Doğrusal Regresyon modelleri.
  - *Özellikler:* R² skoru hesaplama, Gerçek vs Tahmin görselleştirme grafikleri.
- **Backward Elimination (Değişken Eleme):**
  - `Statsmodels` kütüphanesi ile OLS raporu çıkarma.
  - P-value değerine göre gereksiz değişkenleri (Feature Selection) tespit etme.

#### 🤖 Sınıflandırma (Classification) Modelleri
*(Yakında eklenecek: Logistic Regression, KNN, SVM...)*

---

## 🚀 Nasıl Kullanılır?

1. **Şablon Seç:** İhtiyacın olan konunun `.py` dosyasını aç (Örn: `Sablon_Veri_On_Isleme.py`).
2. **Kopyala:** Kodu kendi çalışma dosyana yapıştır.
3. **Düzenle:**
   - `pd.read_csv("dosya_yolu.csv")` kısmını kendi veri setine göre güncelle.
   - Sütun indekslerini (Örn: `iloc[:, 1:4]`) verine uygun hale getir.
4. **Çalıştır:** Modelini eğit ve sonuçları gözlemle!

---

## 🛠 Kullanılan Teknolojiler & Kütüphaneler

* ![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
* ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=flat&logo=pandas)
* ![NumPy](https://img.shields.io/badge/NumPy-Matrix%20Ops-blue?style=flat&logo=numpy)
* ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=flat&logo=scikit-learn)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red?style=flat&logo=plotly)
* ![Statsmodels](https://img.shields.io/badge/Statsmodels-Statistics-purple?style=flat&logo=python)

---
*Bu depo, öğrenim sürecimle paralel olarak sürekli güncellenmektedir.*
