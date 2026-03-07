# 🔄 ML PROJE WORKFLOW

## 📋 Tam Pipeline Adımları

```
1. 🔍 EDA
   └── templates/utils/EDA_Template.ipynb
       ├── Genel bakış (shape, dtype)
       ├── Eksik değer analizi
       ├── İstatistiksel özet
       ├── Hedef değişken dağılımı
       ├── Feature dağılımları
       ├── Korelasyon matrisi
       └── Outlier tespiti (IQR)

2. 🩹 EKSİK DEĞER
   └── templates/preprocessing/Handling_Missing_Values.ipynb
       ├── >%60 eksik → sil
       ├── %20-60 → KNNImputer
       └── <%20 → median/mod

3. 🏷️ ENCODING
   └── templates/preprocessing/Encoding.ipynb
       ├── Binary → LabelEncoder
       ├── Nominal (≤10) → One-Hot
       ├── Ordinal → OrdinalEncoder
       └── Yüksek cardinality → Target Encoding

4. 🏗️ FEATURE ENGINEERING (opsiyonel)
   └── templates/preprocessing/Feature_Engineering.ipynb
       ├── Log/sqrt/kare dönüşümleri
       ├── Etkileşim özellikleri
       ├── Zaman özellikleri
       └── Binning/gruplama

5. ⚖️ SCALING
   └── templates/preprocessing/Scaling.ipynb
       ├── StandardScaler → normal dağılım
       ├── MinMaxScaler → neural network
       └── RobustScaler → outlier çok
       ⚠️ KURAL: fit sadece X_train!

6. 🏆 FEATURE IMPORTANCE (opsiyonel)
   └── templates/preprocessing/Feature_Importance.ipynb
       ├── MDI (tree-based)
       ├── Permutation Importance
       └── SHAP (opsiyonel)

7. 🤖 MODEL EĞİTİMİ
   └── templates/classification/ veya regression/
       ├── Random Forest → güçlü başlangıç
       ├── XGBoost / LightGBM → fine-tune
       └── Voting_Ensemble → modelleri birleştir

8. 📊 DEĞERLENDİRME
   └── templates/evaluation/
       ├── Cross_Validation.ipynb    → güvenilir skor
       ├── ROC_AUC.ipynb             → eşik analizi
       ├── Classification_Report.ipynb → P/R/F1
       └── Confusion_Matrix.ipynb    → hata analizi
```

---

## ⚠️ Kritik Kurallar

1. **Train/Test sızıntısı önle:**
   - Scaler/Encoder/Imputer → sadece X_train'de `fit_transform`
   - X_test'e sadece `transform`
   - Feature Engineering → train'de hesapla, test'e uygula

2. **Sıra önemli:**
   EDA → Eksik Değer → Encoding → Engineering → Scaling → Model → Evaluation

3. **Model seçimi:**
   - Az veri, hızlı deney → `Random Forest`
   - Yarışma / maksimum skor → `XGBoost` veya `LightGBM`
   - Son hamle → `Voting_Ensemble`

---

## 🎯 Karar Ağacı

```
Veri geldi!
  │
  ├─ Hedef sayısal mı?  → regression/
  ├─ Hedef kategorik mi? → classification/
  └─ Etiket yok mu?     → clustering/

Model seç:
  ├─ Hız önemli?        → LightGBM
  ├─ Yorumlanabilirlik? → Logistic / Decision Tree
  └─ Maksimum skor?     → XGBoost + Voting Ensemble

Değerlendir:
  ├─ Sınıf dengeli?     → Accuracy + Confusion Matrix
  └─ Sınıf dengesiz?   → ROC-AUC + F1 (macro)
```

---
**💪 Pipeline'ı takip et, sonuçlar gelir!**
