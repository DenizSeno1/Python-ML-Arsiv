# 🎯 ML ŞABLON KULLANIM AKIŞI

## 📊 Tam ML Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                 BAŞLANGIÇ: VERİ KEŞFİ                       │
│            templates/utils/EDA_Template.ipynb               │
│                                                             │
│  📥 CSV → 📊 Genel Bakış → 🩹 Eksik → 📈 Dağılım → 🔗 Korelasyon │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  VERİ ÖN İŞLEME                             │
│              templates/preprocessing/                       │
│                                                             │
│  🩹 Handling_Missing_Values → 🏷️ Encoding → ⚖️ Scaling      │
│  🏗️ Feature_Engineering → 🏆 Feature_Importance            │
└─────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────┼──────────────────┐
         ↓                  ↓                  ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ SINIFLANDIRMA│   │  REGRESYON   │   │  KÜMELEME    │
│classification│   │  regression/ │   │  clustering/ │
└──────────────┘   └──────────────┘   └──────────────┘
         ↓
  ┌─────────────────────────────────┐
  │  Random Forest / XGBoost /      │
  │  LightGBM / Logistic / SVM ...  │
  └─────────────────────────────────┘
         ↓
  ┌──────────────────┐
  │  Voting Ensemble │  ← Modelleri birleştir!
  └──────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    DEĞERLENDİRME                            │
│               templates/evaluation/                        │
│                                                             │
│  📋 Classification_Report                                  │
│  🟥 Confusion_Matrix                                        │
│  📈 ROC_AUC                                                 │
│  🔄 Cross_Validation                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Notebook Kataloğu

### 📁 classification/
| Notebook | Açıklama | Ne zaman? |
|----------|----------|-----------|
| `Random_Forest_Classifier` | Güçlü, versatile | Default başlangıç |
| `XGBoost` | Gradient boosting | Tablo veri, yarışmalar |
| `LightGBM` | Hızlı XGBoost | Büyük veri |
| `Logistic_Regression` | Yorumlanabilir | Proba gerekince |
| `SVM_Classifier` | Yüksek boyut | Az veri, metin |
| `Decision_Tree` | Görsel | Açıklanabilirlik |
| `Voting_Ensemble` | Model birleştirme | Skor artırmak |

### 📁 evaluation/
| Notebook | Açıklama |
|----------|----------|
| `Cross_Validation` | k-fold ile güvenilir ölçüm |
| `ROC_AUC` | Eşik analizi, dengesiz veri |
| `Classification_Report` | Precision/Recall/F1 |
| `Confusion_Matrix` | TP/TN/FP/FN görselleştirme |

### 📁 preprocessing/
| Notebook | Açıklama |
|----------|----------|
| `Handling_Missing_Values` | NaN analiz + doldurma |
| `Encoding` | Label/OHE/Ordinal/Target |
| `Scaling` | Standard/MinMax/Robust |
| `Feature_Engineering` | Yeni özellik türetme |
| `Feature_Importance` | MDI + Permutation |

### 📁 utils/
| Notebook | Açıklama |
|----------|----------|
| `EDA_Template` | 8 adımlı sistematik keşif |

---

## 🔄 Model Seçim Tablosu

| Özellik | RF | XGB | LGB | LR | SVM |
|---------|----|----|-----|----|-----|
| Hız | ⚡⚡ | ⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡ | ⚡ |
| Hassasiyet | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Yorumlanabilirlik | 🟡 | 🔴 | 🔴 | 🟢 | 🔴 |
| Büyük Veri | 🟡 | 🟡 | 🟢 | 🟢 | 🔴 |

---
**💪 Doğru araçla doğru işi yap!**
