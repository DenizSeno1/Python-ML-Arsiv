"""
╔══════════════════════════════════════════════════════════════╗
║  📊 MATPLOTLIB GÖRSELLEŞTİRME ŞABLONLARI                     ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 AMAÇ: ML modelleri için hazır grafik şablonları          ║
║  📋 İÇERİK:                                                  ║
║     1. Scatter Plot (Gerçek vs Tahmin)                       ║
║     2. Regresyon Çizgisi                                     ║
║     3. Residual Plot (Hata Analizi)                          ║
║     4. Feature Importance (Özellik Önemi)                    ║
║     5. Learning Curve                                        ║
║     6. Confusion Matrix (Classification için)                ║
║     7. Çoklu Model Karşılaştırma                             ║
╚══════════════════════════════════════════════════════════════╝

✅ KULLANIM: İhtiyacın olan bölümü kopyala, değiştir, kullan!
"""

import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# 1️⃣ SCATTER PLOT: GERÇEK vs TAHMİN
# ═══════════════════════════════════════════════════════════════
def plot_actual_vs_predicted(y_test, y_pred, title="Gerçek vs Tahmin"):
    """
    🎯 NE ZAMAN: Her regresyon modelinden sonra
    📊 GÖSTERIR: Tahminlerin gerçeğe ne kadar yakın olduğunu
    """
    plt.figure(figsize=(10, 6))
    
    # Noktalar
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='navy', s=50)
    
    # Mükemmel tahmin çizgisi (45 derece)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Mükemmel Tahmin')
    
    plt.xlabel('Gerçek Değerler', fontsize=12)
    plt.ylabel('Tahmin Edilen Değerler', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Kullanım:
# plot_actual_vs_predicted(y_test, y_pred, "Random Forest Sonuçları")


# ═══════════════════════════════════════════════════════════════
# 2️⃣ REGRESYON ÇİZGİSİ (Tek Değişken İçin)
# ═══════════════════════════════════════════════════════════════
def plot_regression_line(X, y, model, title="Regresyon"):
    """
    🎯 NE ZAMAN: Simple Linear, Polynomial için
    📊 GÖSTERIR: Veri noktaları + Model çizgisi
    """
    plt.figure(figsize=(10, 6))
    
    # Veriyi düzleştir ve sırala
    X_flat = X.ravel() if hasattr(X, 'ravel') else np.array(X).ravel()
    y_flat = y.ravel() if hasattr(y, 'ravel') else np.array(y).ravel()
    sort_idx = X_flat.argsort()
    
    # Noktalar
    plt.scatter(X_flat[sort_idx], y_flat[sort_idx], color='red', alpha=0.6, s=50, label='Gerçek Veri')
    
    # Tahmin çizgisi (pürüzsüz)
    X_line = np.linspace(X_flat.min(), X_flat.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='blue', linewidth=2, label='Model Tahmini')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Kullanım:
# plot_regression_line(X_train, y_train, model, "Linear Regression")


# ═══════════════════════════════════════════════════════════════
# 3️⃣ RESIDUAL PLOT (Hata Analizi)
# ═══════════════════════════════════════════════════════════════
def plot_residuals(y_test, y_pred, title="Residual Plot"):
    """
    🎯 NE ZAMAN: Model hatalarını analiz etmek için
    📊 GÖSTERIR: Hataların dağılımı (0 etrafında olmalı)
    """
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sol: Scatter
    axes[0].scatter(y_pred, residuals, alpha=0.6, color='purple', s=50)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Tahmin Edilen Değerler', fontsize=12)
    axes[0].set_ylabel('Hatalar (Residuals)', fontsize=12)
    axes[0].set_title('Hata Dağılımı', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Sağ: Histogram
    axes[1].hist(residuals, bins=30, color='teal', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Hata Değeri', fontsize=12)
    axes[1].set_ylabel('Frekans', fontsize=12)
    axes[1].set_title('Hata Histogramı', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# Kullanım:
# plot_residuals(y_test, y_pred, "Random Forest Hata Analizi")


# ═══════════════════════════════════════════════════════════════
# 4️⃣ FEATURE IMPORTANCE (Özellik Önemi)
# ═══════════════════════════════════════════════════════════════
def plot_feature_importance(model, feature_names=None, top_n=10, title="Özellik Önemi"):
    """
    🎯 NE ZAMAN: Random Forest, XGBoost, Decision Tree için
    📊 GÖSTERIR: Hangi özellikler en önemli
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    if feature_names is None:
        feature_names = [f"Özellik {i}" for i in range(len(importances))]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='navy')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Önem Derecesi', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # En önemli üstte
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# Kullanım:
# plot_feature_importance(rf_model, feature_names=['Yaş', 'Gelir', 'Eğitim'])


# ═══════════════════════════════════════════════════════════════
# 5️⃣ MODEL KARŞILAŞTIRMA (Bar Chart)
# ═══════════════════════════════════════════════════════════════
def plot_model_comparison(model_names, scores, title="Model Karşılaştırması"):
    """
    🎯 NE ZAMAN: Birden fazla modeli karşılaştırmak için
    📊 GÖSTERIR: R² skorlarını yan yana
    """
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, color=colors, edgecolor='black')
    
    # Değerleri barların üstüne yaz
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('R² Skoru', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# Kullanım:
# plot_model_comparison(
#     ['Linear', 'Polynomial', 'Random Forest', 'SVR'],
#     [0.65, 0.82, 0.94, 0.88]
# )


# ═══════════════════════════════════════════════════════════════
# 6️⃣ CONFUSION MATRIX (Classification İçin)
# ═══════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    🎯 NE ZAMAN: Classification modelleri için
    📊 GÖSTERIR: Doğru/Yanlış tahminlerin dağılımı
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    
    if labels is None:
        labels = [f"Sınıf {i}" for i in range(len(cm))]
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Değerleri hücrelere yaz
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    plt.xlabel('Tahmin Edilen', fontsize=12)
    plt.ylabel('Gerçek', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Kullanım:
# plot_confusion_matrix(y_test, y_pred, labels=['Negatif', 'Pozitif'])


# ═══════════════════════════════════════════════════════════════
# 7️⃣ LEARNING CURVE
# ═══════════════════════════════════════════════════════════════
def plot_learning_curve(train_sizes, train_scores, test_scores, title="Learning Curve"):
    """
    🎯 NE ZAMAN: Overfitting/Underfitting analizi için
    📊 GÖSTERIR: Veri arttıkça model performansı nasıl değişiyor
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    # Eğitim skoru
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Eğitim Skoru')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    # Test skoru
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Test Skoru')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
    
    plt.xlabel('Eğitim Seti Boyutu', fontsize=12)
    plt.ylabel('Skor', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Kullanım:
# from sklearn.model_selection import learning_curve
# train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
# plot_learning_curve(train_sizes, train_scores, test_scores)


# ═══════════════════════════════════════════════════════════════
# 🎨 BONUS: STIL AYARLARI (Tüm Grafikleri Güzelleştir)
# ═══════════════════════════════════════════════════════════════
def set_plot_style():
    """Profesyonel görünüm için stil ayarları"""
    plt.style.use('seaborn-v0_8-whitegrid')  # veya 'ggplot', 'dark_background'
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

# Dosyanın başında çağır:
# set_plot_style()


print("=" * 60)
print("📊 MATPLOTLIB ŞABLONLARI HAZIR!")
print("=" * 60)
print("\n📋 Kullanılabilir Fonksiyonlar:")
print("   1. plot_actual_vs_predicted(y_test, y_pred)")
print("   2. plot_regression_line(X, y, model)")
print("   3. plot_residuals(y_test, y_pred)")
print("   4. plot_feature_importance(model)")
print("   5. plot_model_comparison(names, scores)")
print("   6. plot_confusion_matrix(y_true, y_pred)")
print("   7. plot_learning_curve(sizes, train, test)")
print("=" * 60)
