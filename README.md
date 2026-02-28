# Hiperspektral Görüntülerin Makine Öğrenmesi ve Derin Öğrenme Yöntemleriyle Sınıflandırılması

## 📚 Proje Bilgileri

**Proje Türü:** Bilgisayar Mühendisliği Bitirme Projesi  
**Üniversite:** Yıldız Teknik Üniversitesi  
**Öğrenci Numaraları:** 22011083, 22011084  
**Öğrenciler:** Tan Erciyas, Kerem Baydar  
**İletişim:**
- tan.erciyas@std.yildiz.edu.tr
- kerem.baydar@std.yildiz.edu.tr

---

## 🎯 Proje Hedefleri

Projenin temel amacı, yüksek boyutlu ve karmaşık spektral veriye sahip hiperspektral görüntülerin (HSI) sınıflandırılmasında, **eğiticili (supervised)** ve **yarı eğiticili (semi-supervised)** öğrenme yaklaşımlarının performans farklarını nicel olarak ortaya koymaktır.

### Ana Hedefler:
1. **Performans Karşılaştırması:** Farklı makine öğrenmesi ve derin öğrenme yöntemlerinin hiperspektral görüntü sınıflandırmadaki etkinliğini karşılaştırmak
2. **Boyutsallık Sorunu Çözümü:** "Boyutluluğun laneti" (curse of dimensionality) problemini temel ön işleme teknikleri ile çözmek
3. **Pratik Uygulamalar:** Geliştirililen modelleri gerçek zamanlı test edebilecek kullanıcı dostu bir arayüz geliştirmek
4. **Etiketli Veri Kısıtlılığı:** Sınırlı etiketli örnek sayısında model performansını optimize etmek

---

## 📊 Veri Setleri

Proje kapsamında aşağıdaki üç temel hiperspektral veri seti kullanılacaktır:

### 1. Indian Pines
- **Boyut:** 145×145 piksel
- **Bant Sayısı:** 200 spektral bant
- **Sınıf Sayısı:** 16 sınıf
- **Kaynak:** [Kaggle - Indian Pines Hyperspectral Dataset](https://www.kaggle.com/datasets/abhijeetgo/indian-pines-hyperspectral-dataset)

### 2. Pavia University
- **Boyut:** 610×340 piksel
- **Bant Sayısı:** 103 spektral bant
- **Sınıf Sayısı:** 9 sınıf
- **Kaynak:** [Kaggle - Pavia University HSI](https://www.kaggle.com/datasets/syamkakarla/pavia-university-hsi)

### 3. Salinas
- **Boyut:** 512×217 piksel
- **Bant Sayısı:** 204 spektral bant
- **Sınıf Sayısı:** 16 sınıf
- **Kaynak:** [Kaggle - Salinas Dataset](https://www.kaggle.com/datasets/wangyijialili/salinas)

---

## 🔧 Proje Kapsamı ve Yöntemler

### Veri Ön İşleme
- **Gürültü Azaltma:** Gaussian filtreleme
- **Boyut İndirgeme:** Temel Bileşen Analizi (PCA)

### Makine Öğrenmesi Modelleri
Aşağıdaki yöntemler karşılaştırmalı olarak analiz edilecektir:

1. **Eğiticili Öğrenme (Supervised Learning)**
   - Evrişimli Sinir Ağları (CNN)
   - Vision Transformer (ViT)

2. **Yarı Eğiticili Öğrenme (Semi-Supervised Learning)**
   - Self-Training yöntemi

### Eğitim Stratejileri
- **Curriculum Learning:** Eğitimin kararlılığını ve yakınsamasını artırmak için uygulanacak

### Model Değerlendirme Metrikleri
- **Accuracy (Doğruluk):** Genel sınıflandırma başarısı
- **Precision (Kesinlik):** Pozitif tahminlerin doğruluğu
- **Recall (Duyarlılık):** Pozitif örneklerin bulunabilirliği
- **F1-Score:** Kesinlik ve duyarlılık dengesini ölçer
- **Kappa Katsayısı:** Sınıflar arası anlaşma derecesi

---

## 💻 Teknoloji Yığını

### Programlama ve Kütüphaneler
- **Dil:** Python
- **Görüntü İşleme:** OpenCV
- **Makine Öğrenmesi:** Scikit-learn
- **Derin Öğrenme:** PyTorch veya TensorFlow

### Grafik Kullanıcı Arayüzü (GUI)
- **Ön Yüz:** React
- **Arka Uç:** Python
- **İşlem Yöntemi:** Multithreading (Çok İzlekli yapı)
  - Yüklenen hiperspektral görüntünün aynı anda birden fazla model (CNN, ViT, Self-Training) ile sınıflandırma tahmini yapabilmesini sağlar
  - Her model tahminini ayrı thread'de çalıştırarak, kullanıcı arayüzünün duyarlı (responsive) kalmasını garantiler
  - Eşzamanlı işlem gerçekleştirilerek tüm model sonuçları paralel şekilde alınabilir

### Geliştirme Ortamları
- VS Code
- Google Colab

### Versiyon Yönetimi
- GitHub

---

## 📋 Proje Çıktıları

### 1. Teknik Dokümantasyon
Kapsamlı bir teknik rapor sunan:
- Eğiticili ve yarı eğiticili öğrenme yöntemlerinin performans verileri
- Accuracy, Precision, Recall, F1-Score ve Kappa metriklerini içeren karşılaştırmalı tablolar
- Karışıklık Matrisleri (Confusion Matrices)
- Eğitim/Doğrulama Accuracy ve Loss grafikleri
- Curriculum Learning stratejisinin etkinlik analizi

### 2. Grafik Kullanıcı Arayüzü (GUI)
Kullanıcılar tarafından kolayca kullanılabilecek uygulama:
- Hiperspektral görüntü yükleme ve görselleştirme
- **Multithreaded Paralel Tahmin:** Yüklenen görüntü aynı anda tüm modeller tarafından işlenerek çoklu sınıflandırma tahmini
  - CNN modeli ile tahmin (Thread-1)
  - ViT modeli ile tahmin (Thread-2)
  - Self-Training modeli ile tahmin (Thread-3)
  - Tüm tahminler eşzamanlı olarak gerçekleşir ve sonuçlar anında görüntülenir
- Tahmin sonuçlarının karşılaştırmalı görünümü
- Eğitim geçmişi ve istatistik paneli
- Kesintisiz kullanıcı deneyimi (UI thread'inin engellenmemesi)

### 3. Kaynak Kodlar
- GitHub deposunda sürüm kontrol sistemine uygun şekilde yapılandırılmış tüm kodlar
- Detaylı inline documentation ve docstring'ler
- Replicable ve reproducible araştırma yapısı

---

## 📈 Beklenen Çıktılar

1. **Yüksek Performanslı Modeller:** Hiperspektral görüntü sınıflandırmasında %95+ doğruluk oranı
2. **Etkili GUI:** Eğitim görmemiş kullanıcıların model tahmini yapabilmesi
3. **Bilimsel Katkı:** Eğiticili vs yarı eğiticili yöntemlerin karşılaştırmalı analizi
4. **Açık Kaynak Kod:** Geçmiş ve gelecek araştırmacılar için referans kaynağı

---

## 🚀 Hızlı Başlangıç

```bash
# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# Veri setlerini indir
python scripts/download_datasets.py

# Modeli eğit
python train.py --config config/config.yaml

# GUI'yi başlat
python gui/app.py
```

---

## 📞 İletişim

- **Tan Erciyas:** tan.erciyas@std.yildiz.edu.tr | 05053536058
- **Kerem Baydar:** kerem.baydar@std.yildiz.edu.tr | 05447354100

---

## 📄 Lisans

Bu proje Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği bitirme projesidir.

---

**Son Güncelleme:** 28 Şubat 2026
