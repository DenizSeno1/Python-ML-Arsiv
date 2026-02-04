# 🚀 HIZLI BAŞLANGIÇ REHBERİ

## 📝 İlk Defa Kullanacaklar İçin

### 1. Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/DenizSeno1/Python-ML-Arsiv.git
cd Python-ML-Arsiv

# Gereksinimleri yükle
pip install -r requirements.txt
```

---

### 2. İlk Projen (5 Dakika)

#### Adım 1: Veri dosyasını hazırla

`veriler.csv` adında bir CSV dosyası oluştur veya var olanı kullan.

Örnek:
```csv
ulke,boy,kilo,yas,cinsiyet
tr,180,75,25,e
us,170,65,30,k
fr,175,70,28,e
```

#### Adım 2: Jupyter Notebook veya Python IDE'yi aç

```bash
jupyter notebook
# veya
python
```

#### Adım 3: Veriyi hazırla

```python
%run 01_veri_on_isleme.py
```

**Çıktı:**
```
🔧 VERİ ÖN İŞLEME BAŞLIYOR...
✅ Veri yüklendi: 100 satır, 5 sütun
✅ Eksik veriler dolduruldu
✅ Veri bölündü: 67 eğitim, 33 test
✅ Veriler ölçeklendirildi
🎉 VERİ ÖN İŞLEME TAMAMLANDI!
```

#### Adım 4: Model çalıştır

```python
%run 02_regression/random_forest.py
```

**Çıktı:**
```
🌲 RANDOM FOREST REGRESYONU
✅ Random Forest modeli eğitildi (10 ağaç)
✅ Tahminler yapıldı

📊 SONUÇLAR
🎯 R² Başarı Skoru: 0.9234
   → 🏆 Çok iyi! Model harika çalışıyor
```

#### Adım 5: Sonucu gör!

Grafik otomatik açılır ve tahminleri görebilirsin! 🎉

---

## 🔄 Tipik Kullanım Senaryoları

### Senaryo 1: "Hangi model en iyi?"

```python
# Veriyi hazırla
%run 01_veri_on_isleme.py

# Tüm modelleri dene
models = {
    'Simple Linear': '%run 02_regression/simple_linear.py',
    'Polynomial': '%run 02_regression/polynomial.py',
    'Random Forest': '%run 02_regression/random_forest.py'
}

# Her birini çalıştır ve R² skorlarını karşılaştır
```

**Sonuç:**
- Simple Linear: R² = 0.65
- Polynomial: R² = 0.85
- Random Forest: R² = 0.94 ← **Kazanan!**

---

### Senaryo 2: "Gereksiz değişkenler var mı?"

```python
# Veriyi hazırla
%run 01_veri_on_isleme.py

# Multiple Linear çalıştır
%run 02_regression/multiple_linear.py
# R² = 0.73 (Orta seviye)

# Backward Elimination ile temizle
%run utils/backward_elimination.py
# 🗑️ 2 gereksiz değişken bulundu ve atıldı!

# Şimdi 01_veri_on_isleme.py'yi düzenle
# Gereksiz sütunları iloc'tan çıkar
# Tekrar çalıştır → R² = 0.89 (Çok daha iyi!)
```

---

### Senaryo 3: "Kendi verimle test"

```python
# 1. Veri dosyasını değiştir
# 01_veri_on_isleme.py → Satır 23
df = pd.read_csv("benim_verim.csv")

# 2. Sütun indekslerini ayarla
# Satır 29: numeric_data = df.iloc[:, 1:4]  # Senin sütunların
# Satır 35: target = df.iloc[:, -1]         # Hedef değişken
# Satır 38: categorical = df.iloc[:, 0:1]   # Kategorik sütun

# 3. Çalıştır
%run 01_veri_on_isleme.py
%run 02_regression/random_forest.py
```

---

## 🐛 Sık Karşılaşılan Hatalar

### Hata 1: "NameError: name 'X_train' is not defined"

**Sebep:** Veri ön işleme yapılmadı

**Çözüm:**
```python
%run 01_veri_on_isleme.py  # Bunu unutma!
```

---

### Hata 2: "ValueError: could not convert string to float"

**Sebep:** Kategorik veriler sayıya çevrilmedi

**Çözüm:** `01_veri_on_isleme.py` dosyasındaki encoding kısmını kontrol et

---

### Hata 3: "FileNotFoundError: veriler.csv"

**Sebep:** CSV dosyası bulunamadı

**Çözüm:**
```python
# 01_veri_on_isleme.py → Satır 23
df = pd.read_csv("doğru/dosya/yolu.csv")
```

---

## 💡 İpuçları

### 1. Bellekteki Verileri Kontrol Et

```python
# Veri ön işleme sonrası
print(X_train.shape)  # (67, 6) gibi bir şey görmeli
print(y_train.shape)  # (67, 1) gibi bir şey görmeli
```

### 2. Birden Fazla Model Dene

```python
# Hepsini çalıştır, en yüksek R² olanı seç
%run 02_regression/simple_linear.py     # R² = ?
%run 02_regression/polynomial.py        # R² = ?
%run 02_regression/random_forest.py     # R² = ? ← Genelde en iyi
```

### 3. Grafikleri Kaydet

```python
# Her model dosyasının sonuna ekle:
import matplotlib.pyplot as plt
plt.savefig('my_model_result.png', dpi=300, bbox_inches='tight')
```

---

## 📚 Daha Fazla Bilgi

- 📖 Detaylı rehber: `WORKFLOW.md`
- 🔍 Model karşılaştırması: `README.md` → Model Comparison
- 🛠 Şablonları özelleştirme: Her dosyanın başındaki `🔧 AYARLAR` kısmına bak

---

## 🎓 Öğrenme Yolu

1. **Başla:** Simple Linear → Basit ve anlaşılır
2. **Derinleş:** Polynomial → Kıvrımlı ilişkileri öğren
3. **Ustalaş:** Random Forest → En güçlü aracı kullan
4. **İyileştir:** Backward Elimination → Optimizasyon öğren

---

## ❓ Yardım

Sorun mu var? GitHub Issues'a yaz:
https://github.com/DenizSeno1/Python-ML-Arsiv/issues

---

**Mutlu Kodlamalar! 🚀**
