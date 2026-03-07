# 🚀 HIZLI BAŞLANGIÇ REHBERİ

## 📦 Kurulum

```bash
git clone https://github.com/DenizSeno1/Python-ML-Arsiv.git
cd Python-ML-Arsiv
pip install -r requirements.txt
```

---

## 🗂️ Arşiv Yapısı

```
templates/
├── classification/     # Sınıflandırma modelleri
├── regression/         # Regresyon modelleri
├── clustering/         # Kümeleme modelleri
├── evaluation/         # Değerlendirme araçları
├── preprocessing/      # Veri ön işleme
└── utils/              # EDA, yardımcı araçlar
```

---

## ⚡ 5 Dakikada İlk Proje

### Adım 1: EDA ile keşfet
```python
# templates/utils/EDA_Template.ipynb
VERI_YOLU = 'veri.csv'   # <--- değiştir
HEDEF    = 'target'      # <--- değiştir
```

### Adım 2: Veriyi temizle
```python
# templates/preprocessing/Handling_Missing_Values.ipynb
# templates/preprocessing/Encoding.ipynb
# templates/preprocessing/Scaling.ipynb
```

### Adım 3: Model eğit
```python
# templates/classification/Random_Forest_Classifier.ipynb
# templates/regression/Random_Forest_Regressor.ipynb
```

### Adım 4: Değerlendir
```python
# templates/evaluation/Classification_Report.ipynb
# templates/evaluation/Confusion_Matrix.ipynb
# templates/evaluation/ROC_AUC.ipynb
```

---

## 🔄 Tipik Senaryolar

### "Hangi model en iyi?"
```python
# templates/evaluation/Cross_Validation.ipynb
# → Tüm modelleri k-fold ile karşılaştır
```

### "Feature'lar önemli mi?"
```python
# templates/preprocessing/Feature_Importance.ipynb
# → MDI + Permutation Importance grafikleri
```

### "Performansı artır"
```python
# templates/classification/Voting_Ensemble.ipynb
# → Hard/Soft voting ile modelleri birleştir
```

---

## 🐛 Sık Karşılaşılan Hatalar

### "NameError: X_train is not defined"
→ Önce veri ön işleme notebook'unu çalıştır!

### "ValueError: could not convert string to float"
→ `templates/preprocessing/Encoding.ipynb` çalıştır

### "FileNotFoundError: veri.csv"
→ Her notebook'ta `VERI_YOLU` değişkenini güncelle

---

## 💡 İpuçları

- `# <--- DAYI BURAYI DOLDUR (***)` etiketlerini ara
- Scaler/Encoder'ı **SADECE** X_train'de `fit` et, X_test'e `transform`!
- EDA'dan başla → eksik değer → encoding → scaling → model → evaluation

---

## 📚 Daha Fazla

- Akış diyagramı: `docs/VISUAL_GUIDE.md`
- Detaylı workflow: `docs/WORKFLOW.md`

---
**Mutlu Kodlamalar! 🚀**
