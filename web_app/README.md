# OBB Dataset Viewer - Web Uygulaması

Modern web tabanlı OBB (Oriented Bounding Box) dataset görselleştirme uygulaması. Bu uygulama, eski ve yeni OBB'leri karşılaştırmalı olarak görselleştirmenizi sağlar.

## 🚀 Özellikler

### 📊 Görselleştirme
- **Sol Panel**: Eski OBB + Ground Truth Mask
- **Sağ Panel**: Yeni OBB + Ground Truth Mask
- **Mean IoU Göstergesi**: Sağ panelin üzerinde şık badge
- **Renk Kodlaması**: 
  - 🟢 **Yeşil**: Ground Truth Mask
  - 🔴 **Kırmızı**: Eski OBB
  - 🔵 **Mavi**: Yeni OBB

### 📈 IoU Analizi
- **Gerçek Zamanlı Hesaplama**: Her görsel için mean IoU
- **Detaylı Gösterim**: Her shelf için IoU değeri
- **Karşılaştırmalı Analiz**: Eski vs Yeni OBB performansı
- **Akıllı Sıralama**: Yeni OBB'ye göre en kötüden en iyiye sıralama

## 🛠️ Kurulum

### Yerel Kurulum

1. **Gereksinimler**:
```bash
pip install -r requirements.txt
```

2. **Uygulamayı Çalıştır**:
```bash
python app.py
```

3. **Tarayıcıda Aç**:
```
http://localhost:5000
```

### Docker ile Kurulum

1. **Docker Image Build**:
```bash
docker build -t obb-web-viewer .
```

2. **Container Çalıştır**:
```bash
docker run -p 5000:5000 obb-web-viewer
```

3. **Tarayıcıda Aç**:
```
http://localhost:5000
```

## 📁 Dataset Formatı

### ZIP Dosya Yapısı
```
dataset.zip
├── 00011.jpg
├── 00011.json
├── 00012.jpg
├── 00012.json
└── ...
```

### JSON Formatı
```json
{
  "image_name": "00011.jpg",
  "instances": [
    {
      "segmentation_polygon": {
        "coordinates": [[x1,y1], [x2,y2], ...]
      },
      "old_obb": {
        "coordinates": [[x1,y1], [x2,y2], ...],
        "center": [cx, cy],
        "iou_with_gt": 0.856
      },
      "new_obb": {
        "coordinates": [[x1,y1], [x2,y2], ...],
        "center": [cx, cy],
        "iou_with_gt": 0.923
      }
    }
  ]
}
```

## 🎮 Kullanım

### 1. Dataset Yükleme
- **ZIP Dosyası**: Dataset klasörünü ZIP olarak sıkıştırın
- **Drag & Drop**: Dosyayı sürükleyip bırakın
- **Dosya Seç**: Manuel olarak dosya seçin

### 2. Görsel Navigasyon
- **Dropdown**: Görsel listesinden seçim (mean IoU ile sıralı)
- **Önceki/Sonraki**: Butonlarla gezinme
- **Klavye**: Sol/Sağ ok tuşları

### 3. Analiz
- **Mean IoU**: Sağ panelin üzerinde badge
- **Shelf Detayları**: Her shelf için IoU değeri
- **Karşılaştırma**: Sol panelde eski OBB, sağ panelde yeni OBB
- **Sıralama**: En düşük yeni OBB IoU'lu görseller önce gösterilir


