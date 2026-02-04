# 🎯 ML Şablon Kullanım Rehberi

## 📋 Hızlı Başlangıç

### Adım 1: Veri Hazırlığı
```bash
python 01_veri_on_isleme.py
```
✅ Bu dosyayı çalıştır → X_train, X_test, y_train, y_test oluşur
✅ Bellekte kalır, diğer dosyalar bunları kullanır

---

### Adım 2: Model Seçimi

#### 🔹 Doğrusal İlişkiler İçin
```bash
# Tek değişken → Basit düz çizgi
python 02_regression/simple_linear.py

# Çok değişken → Çok boyutlu düzlem
python 02_regression/multiple_linear.py
```

#### 🔹 Kıvrımlı İlişkiler İçin
```bash
# Polinom (x², x³ gibi)
python 02_regression/polynomial.py

# Karmaşık kıvrımlar
python 02_regression/svr.py
```

#### 🔹 Ağaç Tabanlı Modeller
```bash
# Tek ağaç (basamaklı)
python 02_regression/decision_tree.py

# Çok ağaç (daha hassas)
python 02_regression/random_forest.py
```

---

### Adım 3: İyileştirme (Opsiyonel)

#### Gereksiz Değişkenleri Temizle
```bash
python utils/backward_elimination.py
```
✅ Sadece Multiple Linear Regression'dan SONRA kullan
✅ Hangi sütunların gereksiz olduğunu gösterir

---

## 🧭 Hangi Modeli Ne Zaman Kullanmalıyım?

| Durum | Model | Örnek |
|-------|-------|-------|
| Tek sebep → Tek sonuç (düz çizgi) | Simple Linear | Reklam bütçesi → Satış |
| Çok sebep → Tek sonuç (düz düzlem) | Multiple Linear | Yaş+Boy+Kilo → Sağlık skoru |
| İlişki kıvrımlı ama yumuşak | Polynomial | Sıcaklık → Dondurma satışı |
| İlişki çok karmaşık ve kıvrımlı | SVR | Borsa tahmini |
| Veri basamaklı/kategorik | Decision Tree | Maaş aralıklarına göre |
| En hassas sonuç istiyorum | Random Forest | Ev fiyat tahmini |

---

## 🔄 Tam İş Akışı Örneği

```python
# 1️⃣ VERİYİ HAZIRLA
%run 01_veri_on_isleme.py
# → X_train, X_test, y_train, y_test bellekte

# 2️⃣ BASİT MODEL DENE
%run 02_regression/simple_linear.py
# → R² = 0.65 (Kötü değil ama daha iyisi var mı?)

# 3️⃣ POLİNOM DENE
%run 02_regression/polynomial.py
# → R² = 0.92 (Çok daha iyi!)

# 4️⃣ RANDOM FOREST İLE KONTROL ET
%run 02_regression/random_forest.py
# → R² = 0.95 (En iyisi bu!)

# 5️⃣ GEREKSIZ DEĞİŞKENLERİ TEMİZLE
%run utils/backward_elimination.py
# → "Yaş" sütunu gereksizmiş, çıkar
```

---

## ⚠️ Önemli Notlar

### Sıralama ZORUNLU
1. **İLK:** `01_veri_on_isleme.py` (Mutlaka!)
2. **SONRA:** İstediğin model
3. **OPSİYONEL:** Backward Elimination (Sadece Multiple Linear için)

### Dosyalar Arası Bağlantı
- Tüm modeller **aynı Jupyter Notebook'ta** veya **aynı Python session'ında** çalışmalı
- Çünkü `X_train, y_train` gibi değişkenler **bellekte paylaşılıyor**

### Hızlı Test
```python
# Bellekteki verileri kontrol et
print(X_train.shape)  # (satır, sütun) görmeli
print(y_train.shape)  # (satır,) görmeli
```

---

## 📝 Kendi Projene Uyarlama

1. `01_veri_on_isleme.py` dosyasını aç
2. `pd.read_csv("veriler.csv")` → Kendi CSV'ni yaz
3. `iloc[:, 1:4]` → Kendi sütun numaralarını ayarla
4. Çalıştır ve akışa devam et!

---

**🎓 İpucu:** Her model dosyasının başında "NE ZAMAN KULLAN" açıklaması var. Oradan da bakabilirsin!
