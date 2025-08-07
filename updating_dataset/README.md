# OBB Güncelleme ve JSON Üretim Scripti

Bu script, `new_dataset/worst_dataset_1000` klasörü altındaki tüm görseller için:
- Eski OBB'leri (YOLO formatından)
- Ground truth segmentation poligonlarını (segmentation_labels klasöründen)
- Yeni OBB'leri (`cv2.minAreaRect` ile)
hesaplar ve her bir raf (shelf) için sadeleştirilmiş yeni bir JSON dosyası üretir.

## Özellikler

- Tüm görselleri otomatik işler.
- Her raf için:
  - Eski OBB koordinatları ve özellikleri
  - Ground truth segmentation poligon koordinatları
  - Yeni OBB koordinatları ve özellikleri
- Sonuçları `new_dataset/new_updated_worst_1000` klasörüne kaydeder.
- Görselleştirme yapmaz, sadece JSON ve görsel kopyalar.

## JSON Formatı

Her görsel için örnek çıktı:
```json
{
  "image_name": "00011.jpg",
  "instances": [
    {
      "shelf_id": 1,
      "old_obb": {
        "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "center": [x, y],
        "width": w,
        "height": h,
        "angle": angle,
        "iou_with_gt": 0.xxxx
      },
      "segmentation_polygon": {
        "coordinates": [[x1, y1], [x2, y2], ...],
        "points": [x1, y1, x2, y2, ...]
      },
      "new_obb": {
        "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "center": [x, y],
        "width": w,
        "height": h,
        "angle": angle,
        "iou_with_gt": 0.xxxx
      }
    }
  ]
}
```
