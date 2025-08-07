import cv2
import numpy as np
import json
import os
from pathlib import Path

def process_single_image(img_path, json_path, segmentation_json_path, yolo_txt_path, output_dir):
    """Tek bir görseli işle ve yeni JSON oluştur"""
    
    # Görseli yükle
    image = cv2.imread(img_path)
    if image is None:
        print(f"Görsel yüklenemedi: {img_path}")
        return None
    
    height, width = image.shape[:2]
    
    # segmentation mask poligonlarını yükle
    if not os.path.exists(segmentation_json_path):
        print(f"Segmentation JSON bulunamadı: {segmentation_json_path}")
        return None
    
    with open(segmentation_json_path, "r") as f:
        seg_data = json.load(f)
    
    # Sadece "shelf" class'ındaki poligonları al
    ground_truth_polygons = []
    for instance in seg_data["instances"]:
        if instance["className"] == "shelf":
            points = instance["points"]
            polygon = np.array(points).reshape(-1, 2).astype(np.int32)
            ground_truth_polygons.append(polygon)
    
    # YOLO format dosyasından tüm shelf'leri oku
    orig_obb_list = []
    if os.path.exists(yolo_txt_path):
        with open(yolo_txt_path, 'r') as f:
            for line in f:
                if line.strip() and line.startswith('1 '):  # Shelf class_id=1
                    parts = line.strip().split()
                    coords = list(map(float, parts[1:]))  # 8 koordinat
                    points = np.array(coords, dtype=np.float32).reshape(4, 2)
                    # Denormalize et
                    points[:, 0] *= width
                    points[:, 1] *= height
                    orig_obb_list.append(points.astype(np.int32))
        print(f"  YOLO dosyasından {len(orig_obb_list)} shelf okundu")
    else:
        print(f"  YOLO dosyası bulunamadı: {yolo_txt_path}")
    
    print(f"  Ground truth poligonları: {len(ground_truth_polygons)}")
    
    # Yeni JSON için veri yapısı
    new_data = {
        "image_name": os.path.basename(img_path),
        "instances": []
    }
    
    # Her shelf için işlem yap
    for shelf_idx, orig_obb_points in enumerate(orig_obb_list):
        # Orijinal OBB mask'i oluştur
        orig_obb_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(orig_obb_mask, [orig_obb_points], 255)
        
        # Contour'ları bul
        contours, _ = cv2.findContours(orig_obb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # En büyük contour'u al
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Yeni Oriented Bounding Box hesapla
            oriented_bbox = cv2.minAreaRect(largest_contour)
            center, (w, h), angle = oriented_bbox
            
            # Yeni OBB köşe noktalarını al
            box_points = cv2.boxPoints(oriented_bbox)
            box_points = box_points.astype(np.int32)
            
            # Yeni OBB mask'i oluştur
            new_obb_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(new_obb_mask, [box_points], 255)
            
            # En yakın ground truth poligonunu bul (merkez mesafesine göre)
            best_gt_polygon = None
            min_distance = float('inf')
            
            for gt_polygon in ground_truth_polygons:
                # GT poligonun merkezini hesapla
                gt_center = np.mean(gt_polygon, axis=0)
                # OBB merkezi ile GT merkezi arasındaki mesafe
                distance = np.linalg.norm(center - gt_center)
                if distance < min_distance:
                    min_distance = distance
                    best_gt_polygon = gt_polygon
            
            if best_gt_polygon is not None:
                # En yakın ground truth poligonunun mask'ini oluştur
                gt_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(gt_mask, [best_gt_polygon], 255)
                
                # IoU hesaplama
                intersection_orig = np.logical_and(gt_mask > 0, orig_obb_mask > 0).sum()
                union_orig = np.logical_or(gt_mask > 0, orig_obb_mask > 0).sum()
                orig_iou = intersection_orig / union_orig if union_orig > 0 else 0
                
                intersection_new = np.logical_and(gt_mask > 0, new_obb_mask > 0).sum()
                union_new = np.logical_or(gt_mask > 0, new_obb_mask > 0).sum()
                new_iou = intersection_new / union_new if union_new > 0 else 0
                
                
                # Orijinal OBB'nin özelliklerini hesapla
                orig_center = np.mean(orig_obb_points, axis=0)
                # Orijinal OBB'nin boyutlarını hesapla
                orig_width = np.linalg.norm(orig_obb_points[1] - orig_obb_points[0])
                orig_height = np.linalg.norm(orig_obb_points[2] - orig_obb_points[1])
                # Orijinal OBB'nin açısını hesapla
                orig_angle = np.degrees(np.arctan2(orig_obb_points[1][1] - orig_obb_points[0][1], 
                                                  orig_obb_points[1][0] - orig_obb_points[0][0]))
                
                instance_data = {
                    "shelf_id": shelf_idx + 1,
                    "old_obb": {
                        "coordinates": orig_obb_points.tolist(),
                        "center": [float(orig_center[0]), float(orig_center[1])],
                        "width": float(orig_width),
                        "height": float(orig_height),
                        "angle": float(orig_angle),
                        "iou_with_gt": float(orig_iou)
                    },
                    "segmentation_polygon": {
                        "coordinates": best_gt_polygon.tolist(),
                        "points": best_gt_polygon.flatten().tolist()
                    },
                    "new_obb": {
                        "coordinates": box_points.tolist(),
                        "center": [float(center[0]), float(center[1])],
                        "width": float(w),
                        "height": float(h),
                        "angle": float(angle),
                        "iou_with_gt": float(new_iou)
                    }
                }
                
                new_data["instances"].append(instance_data)
    
    return new_data

def main():
    # Dizin yolları
    source_dir = "new_dataset/worst_dataset_1000"
    segmentation_dir = "segmentation_labels"
    yolo_dir = "yolo_format"
    output_dir = "new_dataset/new_updated_worst_1000"
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Tüm görselleri bul
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    total_images = len(image_files)
    
    print(f"Toplam {total_images} görsel işlenecek...")
    
    for idx, img_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{total_images}] İşleniyor: {img_file}")
        
        # Dosya yolları
        img_path = os.path.join(source_dir, img_file)
        json_path = os.path.join(source_dir, img_file.replace('.jpg', '.json'))
        segmentation_json_path = os.path.join(segmentation_dir, img_file + '.json')
        yolo_txt_path = os.path.join(yolo_dir, img_file + '.txt')
        
        # Görseli işle
        new_data = process_single_image(img_path, json_path, segmentation_json_path, yolo_txt_path, output_dir)
        
        if new_data is not None:
            # Yeni JSON dosyasını kaydet
            output_json_path = os.path.join(output_dir, img_file.replace('.jpg', '.json'))
            with open(output_json_path, 'w') as f:
                json.dump(new_data, f, indent=2)
            
            # Görseli kopyala
            output_img_path = os.path.join(output_dir, img_file)
            import shutil
            shutil.copy2(img_path, output_img_path)
            
            print(f"✓ {img_file} işlendi - {len(new_data['instances'])} shelf bulundu")
        else:
            print(f"✗ {img_file} işlenemedi")
    
    print(f"\nİşlem tamamlandı! {output_dir} klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 