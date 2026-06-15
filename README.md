# 🚀 Python Makine Öğrenmesi (ML) Arşivi

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Grandmaster%20Path-20BEFF.svg?logo=kaggle)

Bu depo, uçtan uca makine öğrenmesi projeleri geliştirmek ve Kaggle yarışmalarında üst sıralara çıkmak için özenle hazırlanmış, kopyala-yapıştır yapısına uygun **modüler şablonlar** içerir. Eski, karmaşık ve tek parça notebook'lar yerine, modern veri bilimi süreçlerine uygun **adım adım** bir mimari üzerine kurulmuştur.

## 📂 Klasör Mimarisi

Kaggle projelerindeki (örn: House Prices, Spaceship Titanic) iş akışına göre numaralandırılmış klasörler:

### `00_Setup_and_Libraries`
- Proje başlangıcında ihtiyaç duyulan temel ve ileri seviye tüm kütüphaneler (`All_Imports.ipynb`).
- Pandas, Numpy, Scikit-Learn, Optuna, XGBoost, SHAP ve daha fazlası.

### `01_EDA_and_Visualization`
- Keşifsel Veri Analizi (EDA) için gelişmiş görselleştirme şablonları.
- Histogramlar, kutu grafikleri (Boxplots), değişken ilişkileri ve korelasyon matrisleri (`EDA_Templates.ipynb`).

### `02_Feature_Engineering`
Kaggle'da yarışma kazandıran en kritik aşama:
- **Missing Value Handling:** KNNImputer, IterativeImputer (MICE) ve grup bazlı doldurma yöntemleri.
- **Outlier Analysis:** Z-Score, IQR ve çok boyutlu **Isolation Forest** aykırı değer tespiti.
- **Encoding & Scaling:** Target Encoding, Frequency Encoding, Label/One-Hot Encoding ve Robust/MinMax/Standard Scalers.
- **Feature Creation:** Polinomsal özellikler (PolynomialFeatures) ve alan bilgisi (Domain Knowledge) ile yeni sütunlar üretme şablonları.

### `03_Models_and_Tuning`
Makine öğrenmesi modelleri ve hiperparametre optimizasyonu:
- **Tree-Based Models:** XGBoost, LightGBM, CatBoost ve Random Forest kullanım şablonları.
- **Linear Models:** Ridge, Lasso ve ElasticNet ile GridSearchCV cezalandırmalı regresyon şablonları.
- **Optuna Tuning:** Bayesian yaklaşımı ile modellerden maksimum performansı almak için **Optuna** şablonları.
- **Feature Selection:** Gürültü yapan sütunları temizlemek ve modeli anlamak için **SHAP**, Permutation Importance ve Tree Importance.

### `04_Ensemble_and_Evaluation`
Modellerin gücünü birleştirme ve değerlendirme:
- **Stacking & Voting:** Birden fazla güçlü modeli birleştirerek daha dayanıklı ve isabetli tahminler üretme (StackingRegressor, VotingClassifier).
- **Evaluation Metrics:** Cross Validation (K-Fold, Stratified K-Fold), Karmaşıklık Matrisi (Confusion Matrix), ROC-AUC eğrileri ve Regresyon hata grafikleri.

## 🚀 Hızlı Başlangıç

Yeni bir projeye başlarken bu arşivi nasıl kullanacağınızı öğrenmek için `docs/` klasöründeki [**QUICKSTART.md**](docs/QUICKSTART.md) rehberine göz atın!

## 📦 Kurulum

Bu arşivdeki gelişmiş algoritmaların (Optuna, SHAP, XGBoost vb.) bilgisayarınızda sorunsuz çalışması için gerekli paketleri kurun:

```bash
pip install -r requirements.txt
```

---
*Geliştirici: [DenizSeno1](https://github.com/DenizSeno1) - Sürekli güncellenmekte ve geliştirilmektedir.*
