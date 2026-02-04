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

📄 **Şablon:** `Veri_On_Isleme.py`

### 2. Makine Öğrenmesi Modelleri
Şu ana kadar uygulanan ve şablon haline getirilen algoritmalar:

#### 📈 Regresyon (Tahmin) Modelleri
| Model | Dosya | Açıklama |
|-------|-------|----------|
| Simple Linear | `Simple_Linear.py` | Tek değişkenli doğrusal regresyon |
| Multiple Linear | `Multiple_Linear.py` | Çoklu değişkenli regresyon + OLS raporu |
| Polynomial | `Polinomial.py` | Kıvrımlı ilişkiler için polinom regresyon |
| SVR | `SVR.py` | Support Vector Regression |
| Decision Tree | `Decision_Tree.py` | Karar ağacı regresyonu |
| Random Forest | `Random_Forest.py` | Topluluk öğrenmesi (En güçlü!) |

#### 🧹 Yardımcı Araçlar
| Araç | Dosya | Açıklama |
|------|-------|----------|
| Backward Elimination | `Backward_Elimination.py` | P-value ile gereksiz değişken eleme |
| Görselleştirme | `Matplotlib_Sablonlari.py` | 7 farklı grafik şablonu |

#### 🤖 Sınıflandırma (Classification) Modelleri
*(Yakında eklenecek: Logistic Regression, KNN, SVM, XGBoost...)*

---

## 🚀 Nasıl Kullanılır?

```python
# 1. Veriyi hazırla (HER ZAMAN İLK ADIM)
%run Veri_On_Isleme.py

# 2. İstediğin modeli çalıştır
%run Random_Forest.py

# 3. Sonuçları gör: R² skoru + Grafik
```

### Şablonları Kendi Projene Uyarlama:
1. **Şablon Seç:** İhtiyacın olan konunun `.py` dosyasını aç.
2. **Kopyala:** Kodu kendi çalışma dosyana yapıştır.
3. **Düzenle:**
   - `pd.read_csv("veriler.csv")` kısmını kendi veri setine göre güncelle.
   - Sütun indekslerini (`iloc[:, 1:4]`) verine uygun hale getir.
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

## 📚 Kaynaklar
- Şadi Evren Şeker - Machine Learning Dersleri
- [Scikit-Learn Dokümantasyonu](https://scikit-learn.org/stable/documentation.html)

---

*Bu depo, öğrenim sürecimle paralel olarak sürekli güncellenmektedir.*
