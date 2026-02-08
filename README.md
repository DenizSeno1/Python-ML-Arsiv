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

📄 **Şablon:** `utils/Veri_On_Isleme.ipynb`

### 2. Makine Öğrenmesi Modelleri
Şu ana kadar uygulanan ve şablon haline getirilen algoritmalar:

#### 🧹 Yardımcı Araçlar
| Araç | Dosya | Açıklama |
|------|-------|----------|
| Veri Ön İşleme | `utils/Veri_On_Isleme.ipynb` | Veri temizleme ve hazırlama |
| Backward Elimination | `utils/Backward_Elimination.ipynb` | P-value ile gereksiz değişken eleme |
| Görselleştirme | `utils/Matplotlib_Sablonlari.ipynb` | 7 farklı grafik şablonu |

#### 📈 Regresyon (Tahmin) Modelleri
| Model | Dosya | Açıklama |
|-------|-------|----------|
| Simple Linear | `templates/regression/Simple_Linear.ipynb` | Tek değişkenli doğrusal regresyon |
| Multiple Linear | `templates/regression/Multiple_Linear.ipynb` | Çoklu değişkenli regresyon + OLS raporu |
| Polynomial | `templates/regression/Polynomial_Regression.ipynb` | Kıvrımlı ilişkiler için polinom regresyon |
| SVR | `templates/regression/SVR.ipynb` | Support Vector Regression |
| Decision Tree | `templates/regression/Decision_Tree_Regression.ipynb` | Karar ağacı regresyonu |
| Random Forest | `templates/regression/Random_Forest_Regression.ipynb` | Topluluk öğrenmesi (En güçlü!) |


#### 🤖 Sınıflandırma (Classification) Modelleri
| Model | Dosya | Açıklama |
|-------|-------|----------|
| Logistic Regression | `templates/classification/Logistic_Regression.ipynb` | Lojistik Regresyon |
| KNN | `templates/classification/KNN.ipynb` | K-En Yakın Komşu |
| SVM | `templates/classification/SVM.ipynb` | Destek Vektör Makineleri |
| Naive Bayes | `templates/classification/Gaussian_NB.ipynb` | Naive Bayes |
| Random Forest | `templates/classification/Random_Forest_Classifier.ipynb` | Random Forest Sınıflandırıcı |

---

## 🚀 Nasıl Kullanılır?

Tüm şablonlar **Jupyter Notebook (.ipynb)** formatına dönüştürülmüştür.

1. **Önce Veri Ön İşlemeyi Çalıştırın:**
   `utils/Veri_On_Isleme.ipynb` dosyasını açın ve hücreleri sırasıyla çalıştırın. Bu, verinizi temizleyip belleğe (`X_train`, `y_train` olarak) yükleyecektir.

2. **Bir Model Seçin:**
   Örneğin `templates/regression/Random_Forest_Regression.ipynb` dosyasını açın.

3. **Çalıştırın:**
   Veriler zaten bellekte olduğu için (veya notebook içinde `%run` ile çağırıldığı için) doğrudan model eğitilecek ve sonuçlar görünecektir.

### Kendi Projene Uyarlama:
1. `utils/Veri_On_Isleme.ipynb` içinde `pd.read_csv("veriler.csv")` satırını kendi veri setinle değiştir.
2. Sütun indekslerini ayarla.
3. Sonra istediğin model notebook'unu çalıştır!

---

## 🛠 Kullanılan Teknolojiler & Kütüphaneler

* ![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
* ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=flat&logo=pandas)
* ![NumPy](https://img.shields.io/badge/NumPy-Matrix%20Ops-blue?style=flat&logo=numpy)
* ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=flat&logo=scikit-learn)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red?style=flat&logo=plotly)
* ![Statsmodels](https://img.shields.io/badge/Statsmodels-Statistics-purple?style=flat&logo=python)

---

## 📚 Kaynaklar
- Şadi Evren Şeker - Machine Learning Dersleri
- [Scikit-Learn Dokümantasyonu](https://scikit-learn.org/stable/documentation.html)

---

*Bu depo, öğrenim sürecimle paralel olarak sürekli güncellenmektedir.*
