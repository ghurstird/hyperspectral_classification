# Hiperspektral Görüntü Sınıflandırması

Hiperspektral görüntüleme teknolojisini kullanarak arazi örtüsü (land cover) sınıflandırması yapan bir Python projesidir.

## Proje Hakkında

Bu proje, AVIRIS sensörü tarafından toplanan **Indian Pines** veri setini kullanarak hiperspektral sınıflandırma gerçekleştirmektedir. Hiperspektral görüntüler, yüzlerce spektral banttan oluşan çok boyutlu veri setleridir ve her piksel için detaylı spektral bilgi sağlar.

### Veri Seti: Indian Pines
- **Çözünürlük**: 145 × 145 piksel
- **Spektral Bant Sayısı**: 200 (orijinal 220'den su emilim bölgeleri çıkarıldı)
- **Sınıf Sayısı**: 16 ürün + 1 tanımsız sınıf
- **Sınıflar**:
  - Alfalfa, Corn (notill/mintill/normal)
  - Grass (pasture/trees/pasture-mowed)
  - Hay-windrowed, Oats
  - Soybean (notill/mintill/clean)
  - Wheat, Woods
  - Buildings-grass-trees-drives, Stone-steel-towers

## Kurulum

### Gereksinimler
- Python 3.7 veya üzeri
- pip paket yöneticisi

### Adımlar

1. Depoyu klonlayın:
```bash
git clone <repository_url>
cd hyperspectral_classification
```

2. Sanal ortam oluşturun (önerilir):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Jupyter notebook'u başlatın:
```bash
jupyter notebook
```

## Kullanım

### 1. Veri Yükleme ve Görselleştirme
`notebooks/data_loader.ipynb` notebook'unu açın ve hücreleri sırasıyla çalıştırın:
- Gerekli kütüphaneleri içe aktarma
- MAT dosyalarından veri yükleme
- Veri yapısı ve boyutlarının doğrulanması
- Örnek spektral bandları ve ground truth haritasının görselleştirilmesi
- Veri istatistiklerinin hesaplanması

## Dizin Yapısı

```
hyperspectral_classification/
├── README.md              # Proje dokumentasyon
├── requirements.txt       # Python bağımlılıkları
├── data/
│   ├── Indian_pines_corrected.mat    # Hiperspektral görüntü data (145x145x200)
│   └── Indian_pines_gt.mat           # Ground truth etiketleri (145x145)
├── notebooks/
│   └── data_loader.ipynb             # Veri yükleme ve analiz notebook'u
├── src/                   # Sınıflandırma modülleri (geliştirme aşaması)
└── outputs/               # Sonuçlar ve çıktılar
```

## Teknolojiler

- **NumPy**: Sayısal hesaplamalar
- **SciPy**: MAT dosya işlemleri
- **Matplotlib**: Veri görselleştirme
- **Scikit-learn**: Makine öğrenmesi algoritmaları
- **Jupyter**: İnteraktif analiz

## Sonraki Adımlar

- [ ] Veri ön işleme (normalizasyon, boyut indirgeme)
- [ ] Eğitim/test bölümlemesi
- [ ] Sınıflandırma modelleri (SVM, Random Forest, CNN, vb.)
- [ ] Model performans değerlendirmesi
- [ ] Sonuçların görselleştirilmesi

## Kaynaklar

- [Indian Pines Dataset](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)
- [Hiperspektral Görüntü İşleme](https://en.wikipedia.org/wiki/Hyperspectral_imaging)

## Lisans

MIT License

## Yazarlar

Hiperspektral Classification Projesi