# 🎯 ML ŞABLON KULLANIM AKIŞI

```
┌─────────────────────────────────────────────────────────────┐
│                     BAŞLANGIČ NOKTASI                       │
│                  01_veri_on_isleme.py                       │
│                                                             │
│  📥 CSV Dosyası → 🔧 Temizle → 📊 Böl → 📈 Ölçekle         │
│                                                             │
│  ÇIKTI: X_train, X_test, y_train, y_test                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL SEČİMİ                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
          ┌─────────────────┼─────────────────┐
          ↓                 ↓                 ↓
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  BASIT   │      │   ORTA   │      │  GÜČLÜ   │
    └──────────┘      └──────────┘      └──────────┘
          ↓                 ↓                 ↓
          │                 │                 │
    Simple Linear    Polynomial          Random Forest
          │                │                 │
          │          Decision Tree          SVR
          │                │                 │
          │         Multiple Linear          │
          │                │                 │
          ↓                ↓                 ↓
    ┌──────────────────────────────────────────┐
    │         📊 SONUČ + GRAFİK                │
    │                                          │
    │  🎯 R² Skoru                             │
    │  📈 Görselleštirme                       │
    │  📋 Tahmin vs Gerçek                     │
    └──────────────────────────────────────────┘
                      ↓
                      ↓
            ┌─────────────────┐
            │   R² < 0.7 ?    │
            └─────────────────┘
               ↓           ↓
            EVET          HAYIR
               ↓           ↓
    ┌──────────────┐   ┌────────────┐
    │ Başka Model  │   │  TAMAMDIR! │
    │    Dene      │   │  Kullan!   │
    └──────────────┘   └────────────┘
          ↓
          │
   utils/backward_elimination.py
          │ (Sadece Multiple Linear için)
          ↓
   Gereksiz değiškenleri temizle
          ↓
   Tekrar čalıštır → R² Artar! 🎉
```

---

## 📊 MODEL SEČİM TABLOSU

| Özellik | Simple | Multiple | Polynomial | SVR | Decision | Random |
|---------|--------|----------|------------|-----|----------|--------|
| Hız | ⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡ | ⚡ | ⚡⚡ | ⚡ |
| Hassasiyet | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Kullanım | 🟢 | 🟢 | 🟡 | 🔴 | 🟢 | 🟢 |
| Değišken | Tek | Čok | Tek/Čok | Tek/Čok | Tek/Čok | Tek/Čok |

---

## 🚀 ÖRNEK KULLANIM

### Başlangıč (Her Proje İčin Zorunlu)
```python
%run 01_veri_on_isleme.py
```

### Model Seči ve Čalıštır
```python
# Basit dene
%run 02_regression/simple_linear.py      # R² = 0.65

# Daha iyisini dene
%run 02_regression/polynomial.py         # R² = 0.85

# En iyisini dene
%run 02_regression/random_forest.py      # R² = 0.94 ✅
```

### İyileštirme (Opsiyonel)
```python
# Sadece Multiple Linear kullandıysan
%run utils/backward_elimination.py
# → Gereksiz sütunları gösterir
# → 01_veri_on_isleme.py'yi düzenle
# → Tekrar čalıštır
```

---

## 💡 KARAR AĞACI

```
Başla
  │
  ├─ Tek değišken mi?
  │   └─ EVET → simple_linear.py
  │
  ├─ Čok değišken mi?
  │   └─ EVET → multiple_linear.py
  │              │
  │              └─ R² düšük mü?
  │                  └─ EVET → backward_elimination.py
  │
  ├─ Kıvrımlı iliški mi?
  │   └─ EVET → polynomial.py veya svr.py
  │
  └─ En iyi sonuč mu istiyorsun?
      └─ EVET → random_forest.py ⭐
```

---

## 🔄 İTERATİF SÜREČ

```
1. Veriyi hazırla
    ↓
2. Basit model dene (Simple Linear)
    ↓
3. Sonuca bak (R²)
    ↓
    ├─ İyi mi? (R² > 0.9)
    │   └─ EVET → Bitir! ✅
    │
    └─ Kötü mü? (R² < 0.9)
        └─ EVET → Daha karmaşık model dene
                   (Polynomial → Random Forest)
        ↓
        Tekrar 3'e dön
```

---

**💪 GÜČLÜ YAPIYI KULLA, ZAMANINDAN KAZAN!**
