# Mask → Trapezoid → OBB Görselleştirme UI

Segmantasyon mask poligonlarından trapezoid fit ve sonrasında Oriented Bounding Box (OBB) üretim sürecini adım adım görselleştiren hafif bir Flask arayüzü. Dönüşümün her adımını net biçimde görmenizi, IoU karşılaştırmalarını canlı hesaplamanızı ve toplu veri üretimi yapmanızı sağlar.

## Özellikler

- Tek görsel için adım adım görselleştirme:
  1) Orijinal + doldurulmuş mask (yeşil) + mevcut YOLO OBB (mor)
  2) Mask’ten trapezoid fit (kırmızı)
  3) Yeni OBB’ler (mavi) + raf bazlı IoU’lar ve mean IoU
- IoU’lar gerçek zamanlı hesaplanır 
- Her adım için indirmeler: görseller ve üretilen yeni OBB JSON
- Viewer sayfası: Görsel + Mask + Yeni OBB JSON yükleyip sonuçları yeniden inceleyin
- Batch sayfası:
  - ZIP dataset (görseller + mask JSON’ları + YOLO TXT’leri) yükleyin
  - Tüm dataset’i işleyin, istatistik ve grafikler alın (mean IoU gelişimi, dağılımlar)
  - En iyi/En kötü örnek görselleri görün, yeni OBB JSON’larını ZIP olarak indirin

Kurulum (örnek):
```bash
pip install Flask numpy opencv-python numba matplotlib
```

## Çalıştırma

```bash
cd mask_to_obb_ui
python app.py
```

- Adres: http://127.0.0.1:5001/


## Sayfalar

### 1) Dönüşüm (Ana Sayfa)
- URL: `/`
- Yükleme:
  - Mask JSON (segmantasyon poligonları)
  - Görsel dosya
  - YOLO TXT (opsiyonel)
- Çıktılar:
  - Adım 1: Orijinal + doldurulmuş mask (yeşil) + YOLO OBB (mor)
  - Adım 2: Trapezoid fit (kırmızı)
  - Adım 3: Yeni OBB (mavi) + raf bazlı IoU ve mean IoU
- İndirmeler: adım görselleri + yeni OBB JSON

### 2) Görüntüleyici (Viewer)
- URL: `/viewer`
- Yükleme: Görsel + Mask JSON + Yeni OBB JSON (Adım 3 veya Batch çıktısı)
- Sol: Görsel + doldurulmuş mask
- Sağ: Görsel + doldurulmuş mask + OBB + raf bazlı IoU metinleri
- Mean IoU ve raf bazlı IoU tablosu gösterilir

### 3) Toplu İşlem (Batch)
- URL: `/batch`
- Tek bir ZIP yükleyin; içinde şu yapılar olmalı:
  - Görseller (örn. `*.jpg`)
  - Mask JSON’ları (tercihen `image_name.jpg.json`, yoksa `stem.json`)
  - YOLO TXT etiketleri (destek: `stem.txt`, `stem.jpg.txt`, vb.)
- Uygulama:
  - Tüm dataset’i işler, IoU gelişimlerini hesaplar (old → new)
  - Özet istatistik: mean IoU (old → new), iyileşen/kötüleşen/aynı sayıları
  - Grafikler: IoU dağılımı, iyileşme (New-Old) dağılımı
  - Örnek görseller: en çok iyileşen ve en çok kötüleşen ilk 3
  - Yeni OBB JSON’larını ZIP olarak indirmenizi sağlar

## Veri Formatları

### YOLO TXT (satır başına 1 OBB):
```
class_id x1 y1 x2 y2 x3 y3 x4 y4   # normalize (0..1)
```
- Sadece `class_id = 1` (shelf) işlenir/görselleştirilir.

### Mask JSON (özet yapı)
```json
{
  "metadata": { "height": 2000, "width": 1500, "name": "00001.jpg" },
  "instances": [
    { "type": "polygon", "className": "shelf", "points": [x1, y1, x2, y2, ...] },
    { "type": "polygon", "className": "shelf-space", "points": [...] }
  ]
}
```
- `points` dizisi piksel koordinatlarıdır.

### Renkler
- Mask: yeşil (içi dolu)
- YOLO OBB: mor kontur
- Trapezoid: kırmızı kontur
- Yeni OBB: mavi kontur

