# 🚀 Hızlı Başlangıç Rehberi (Quickstart)

Bu rehber, Kaggle veya kendi projelerine başlarken `Python-ML-Arsiv` klasör mimarisini en verimli nasıl kullanacağını anlatır.

## Proje Akışı (Adım Adım)

Yeni bir projeye başladığında, boş bir notebook aç ve şu adımları takip et:

### Adım 1: Kütüphaneleri Çek (Setup)
İlk olarak `00_Setup_and_Libraries/All_Imports.ipynb` dosyasına git. Oradaki tüm import kodlarını kopyala ve kendi projendeki ilk hücreye yapıştır. Bu, sonradan "modül bulunamadı" hatası almanı önler ve sana hazır bir ortam sunar.

### Adım 2: Veriyi Görselleştir (EDA)
Veriyi yükledikten hemen sonra `01_EDA_and_Visualization/EDA_Templates.ipynb` dosyasına bak.
- Hedef değişkeninin (örn: SalePrice) dağılımını görmek için `plot_distribution()` fonksiyonunu kopyala.
- Bir değişkenin hedefe olan etkisini görmek için `plot_relation_with_target()` kullan.
- Çoklu bağlantı (multicollinearity) durumunu görmek için `plot_correlation_matrix()` fonksiyonunu al.

### Adım 3: Temizlik (Missing Values & Outliers)
Bu aşama yarışma kazandıran yerdir (`02_Feature_Engineering`).
- Eksik veriler için `Missing_Value_Handling.ipynb` dosyasından uygun yöntemi seç (Basit doldurma mı yoksa Iterative Imputer mı?).
- Aykırı değerler için `Outlier_Analysis.ipynb` dosyasındaki IQR veya Isolation Forest kodunu kopyalayıp temizliği yap.

### Adım 4: Özellik Üretme ve Encoding
- Sütunları çarpıp/bölüp veya mantıksal çıkarımlar yapacaksan `Feature_Creation.ipynb` şablonundan yararlan.
- Kategorik sütunları sayısallaştırırken Lineer model kullanacaksan One-Hot, Ağaç modeli kullanacaksan Label Encoding veya `Encoding_and_Scaling.ipynb` içindeki Target/Frequency Encoding yöntemlerini kullan.

### Adım 5: Modeli Eğit ve Optimize Et (Tuning)
Sıra `03_Models_and_Tuning` aşamasında.
- `Tree_Based_Models.ipynb` içinden XGBoost veya LightGBM fonksiyonlarını çekip basit bir model kur (Baseline).
- Modeli iyileştirmek için `Optuna_Tuning_Template.ipynb` dosyasını kopyala ve hiperparametre optimizasyonu başlat.
- Hangi sütunun gerçekten işe yaradığını görmek için `Feature_Selection.ipynb` içindeki SHAP grafiklerini çizdir.

### Adım 6: Güçleri Birleştir (Ensemble)
En iyi skor için tek modele güvenme (`04_Ensemble_and_Evaluation`).
- Eğittiğin farklı modellerin (örn: XGBoost + LightGBM + Ridge) tahminlerini birleştirmek için `Stacking_and_Voting.ipynb` şablonundaki Meta-Model mantığını projene ekle.
- Sonucu değerlendirmek için de `Evaluation_Metrics.ipynb` grafikleriyle sunumunu yap.

---
**Tebrikler!** Eski karmaşık yöntemler yerine, artık tam bir Kaggle ustası gibi adım adım ve modüler bir proje inşa ettin. 🎯
