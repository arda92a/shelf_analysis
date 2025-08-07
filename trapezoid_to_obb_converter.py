"""
Bu script, trapezoid köşe noktalarını optimize edilmiş OBB'ye dönüştürür:

+ Algoritma: Hull Sweep
    -Convex Hull Oluşturma: Trapezoid köşe noktalarından convex hull çıkarır
    -Açı Sweep: -90° ile +90° arasında her açıda hull'u döndürür
    -Axis-Aligned BBox: Döndürülmüş hull'dan axis-aligned bounding box bulur
    -IoU Optimizasyonu: En yüksek IoU'ya sahip OBB'yi seçer
    -Fallback: Eğer hiçbir OBB bulunamazsa, direkt minAreaRect kullanır

Çıktı Formatı:
{
  "obb_coordinates": [center_x, center_y, width, height, angle],
  "trapezoid_iou": 0.85,
  "obb_iou": 0.82,
  "iou_loss": 0.03
}
"""


import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrapezoidToOBBConverter:
    def __init__(self, visualize: bool = True):
        """
        Trapezoid köşe noktalarını OBB'ye çeviren converter
        """
        self.visualize = visualize
        logger.info("Trapezoid to OBB Converter initialized")
    
    def trapezoid_to_obb(self, corners: List[Tuple[float, float]], img_shape: tuple = (640, 640), angle_step: float = 1.0) -> Tuple[float, float, float, float, float]:
        """
        Trapezoid köşe koordinatlarını OBB'ye çevirir. Convex hull + döndürmeli sweep ile en iyi OBB'yi bulur.
        corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] - Normalize edilmiş koordinatlar
        img_shape: mask boyutu (yükseklik, genişlik)
        angle_step: sweep için derece aralığı
        returns: (x_center, y_center, width, height, angle) - Normalize edilmiş OBB
        """
        h, w = img_shape
        pts = np.array([(x * w, y * h) for x, y in corners], dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

        # Convex hull + rotated rectangle sweep yöntemi
        best_iou = -1
        best_obb = None
        pts_int = pts.astype(np.int32)
        hull = cv2.convexHull(pts_int)
        hull_points = hull.reshape(-1, 2)  # shape: (N, 2)
        
        for angle in np.arange(-90, 90, angle_step):
            # Hull'u döndür
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            hull_rot = cv2.transform(np.array([hull_points], dtype=np.float32), M)[0]
            # Axis-aligned bounding box bul
            x, y, bw, bh = cv2.boundingRect(hull_rot.astype(np.int32))
            box = np.array([
                [x, y],
                [x + bw, y],
                [x + bw, y + bh],
                [x, y + bh]
            ], dtype=np.float32)
            # Orijinal açıya geri döndür
            M_inv = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
            box_rot = cv2.transform(np.array([box]), M_inv)[0]
            obb_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obb_mask, [box_rot.astype(np.int32)], 255)
            intersection = np.logical_and(mask > 0, obb_mask > 0).sum()
            union = np.logical_or(mask > 0, obb_mask > 0).sum()
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                rect = cv2.minAreaRect(box_rot)
                (center_x, center_y), (width, height), best_angle = rect
                best_obb = (center_x, center_y, width, height, best_angle)
        
        # Fallback
        if best_obb is None:
            rect = cv2.minAreaRect(hull_points)
            (center_x, center_y), (width, height), angle = rect
            best_obb = (center_x, center_y, width, height, angle)
        
        return best_obb
    
    def obb_to_polygon(self, obb: Tuple[float, float, float, float, float]) -> List[Tuple[float, float]]:
        """
        OBB parametrelerinden polygon köşe noktalarını hesapla
        """
        center_x, center_y, width, height, angle = obb
        
        # Radyan'a çevir
        angle_rad = np.deg2rad(angle)
        
        # Köşe noktalarını hesapla (merkez etrafında)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Width ve height'ı yarıya böl
        w_half = width / 2
        h_half = height / 2
        
        # Döndürülmemiş köşe noktaları
        corners = [
            (-w_half, -h_half),
            (w_half, -h_half), 
            (w_half, h_half),
            (-w_half, h_half)
        ]
        
        # Döndür ve merkeze taşı
        rotated_corners = []
        for x, y in corners:
            rot_x = x * cos_a - y * sin_a + center_x
            rot_y = x * sin_a + y * cos_a + center_y
            rotated_corners.append((rot_x, rot_y))
        
        return rotated_corners
    
    def calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        İki mask arasındaki IoU hesapla
        """
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def obb_to_mask(self, obb: Tuple[float, float, float, float, float], img_shape: Tuple[int, int]) -> np.ndarray:
        """
        OBB'den mask oluştur
        """
        mask = np.zeros(img_shape, dtype=np.uint8)
        corners = self.obb_to_polygon(obb)
        
        if len(corners) >= 3:
            corners_array = np.array(corners, dtype=np.int32)
            cv2.fillPoly(mask, [corners_array], 255)
        
        return mask
    
    def create_comparison_visualization(self, image: np.ndarray, original_mask: np.ndarray,
                                      orig_obb_corners: list, trapezoid_corners: list, obb_corners: list,
                                      shelf_id: int, orig_obb_iou: float, trapezoid_iou: float, obb_iou: float,
                                      save_path: str) -> None:
        """
        Orijinal OBB, Trapezoid ve Optimize OBB karşılaştırmalı görselleştirme oluştur
        """
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f'Shelf {shelf_id + 1}\nOriginal OBB IoU: {orig_obb_iou:.3f} | Trapezoid IoU: {trapezoid_iou:.3f} | Optimized OBB IoU: {obb_iou:.3f}',
                        fontsize=16, fontweight='bold')
            # Sol: Orijinal OBB
            ax1.imshow(image)
            ax1.set_title(f'Original OBB (IoU: {orig_obb_iou:.3f})', fontsize=14, fontweight='bold')
            ax1.axis('off')
            if original_mask.sum() > 0:
                mask_rgba = np.zeros((*original_mask.shape, 4))
                mask_rgba[original_mask > 0] = [1, 0.2, 0.2, 0.4]
                ax1.imshow(mask_rgba)
            if len(orig_obb_corners) >= 3:
                orig_obb_polygon = Polygon(np.array(orig_obb_corners, dtype=np.int32), fill=False, edgecolor='blue', linewidth=2, linestyle='-', alpha=0.9)
                ax1.add_patch(orig_obb_polygon)
            # Orta: Trapezoid
            ax2.imshow(image)
            ax2.set_title(f'Trapezoid (IoU: {trapezoid_iou:.3f})', fontsize=14, fontweight='bold')
            ax2.axis('off')
            if original_mask.sum() > 0:
                ax2.imshow(mask_rgba)
            if len(trapezoid_corners) >= 3:
                trap_polygon = Polygon(np.array(trapezoid_corners, dtype=np.int32), fill=False, edgecolor='darkgreen', linewidth=2, linestyle='-', alpha=0.9)
                ax2.add_patch(trap_polygon)
            # Sağ: Optimize OBB
            ax3.imshow(image)
            ax3.set_title(f'Optimized OBB (IoU: {obb_iou:.3f})', fontsize=14, fontweight='bold')
            ax3.axis('off')
            if original_mask.sum() > 0:
                ax3.imshow(mask_rgba)
            if len(obb_corners) >= 3:
                obb_polygon = Polygon(np.array(obb_corners, dtype=np.int32), fill=False, edgecolor='darkred', linewidth=2, linestyle='-', alpha=0.9)
                ax3.add_patch(obb_polygon)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Shelf {shelf_id + 1} comparison saved: {os.path.basename(save_path)}")
        except Exception as e:
            logger.error(f"Error creating shelf {shelf_id + 1} comparison: {e}")
    
    def parse_yolo_obb_points(self, yolo_line: str, img_width: int, img_height: int) -> list:
        """
        YOLO formatındaki bir satırı (8 koordinat) köşe noktalarına çevirir (piksel cinsinden)
        """
        parts = yolo_line.strip().split()
        coords = list(map(float, parts[1:]))
        points = np.array(coords, dtype=np.float32).reshape(4, 2)
        points[:, 0] *= img_width
        points[:, 1] *= img_height
        return points.tolist()

    def get_yolo_obb_points_for_image(self, yolo_txt_path: str, img_width: int, img_height: int) -> list:
        """
        Bir yolo txt dosyasındaki tüm OBB köşe noktalarını (sadece shelf class_id=1) döndürür
        """
        obb_points = []
        if not os.path.exists(yolo_txt_path):
            return obb_points
        with open(yolo_txt_path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                if line.startswith('1 '):
                    pts = self.parse_yolo_obb_points(line, img_width, img_height)
                    obb_points.append(pts)
        return obb_points
    
    def process_single_json(self, json_path: str, output_dir: str, image_dir: str = None) -> Dict[str, Any]:
        """
        Tek bir JSON dosyasını işle
        """
        try:
            # JSON'u yükle
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_name = data.get("image_name", "")
            if not image_name:
                logger.warning(f"No image_name found in {json_path}")
                return {"error": "No image_name found"}
            
            # Görüntüyü yükle (eğer image_dir verilmişse)
            image = None
            img_height, img_width = 640, 640  # Default değerler
            if image_dir:
                image_path = os.path.join(image_dir, image_name)
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        img_height, img_width = image.shape[:2]
            
            enhanced_obb_labels = data.get("enhanced_obb_labels", [])
            if not enhanced_obb_labels:
                logger.warning(f"No enhanced_obb_labels found in {json_path}")
                return {"error": "No enhanced_obb_labels found"}
            
            converted_results = {
                "image_name": image_name,
                "conversion_method": "trapezoid_to_obb_hull_sweep",
                "converted_obb_labels": [],
                "conversion_stats": {
                    "total_objects": len(enhanced_obb_labels),
                    "successfully_converted": 0,
                    "failed_conversions": 0,
                    "average_trapezoid_iou": 0.0,
                    "average_obb_iou": 0.0,
                    "average_iou_loss": 0.0
                }
            }
            
            trapezoid_ious = []
            obb_ious = []
            
            # Görselleştirme klasörü
            if self.visualize and image is not None:
                vis_dir = os.path.join(output_dir, "visualizations", Path(image_name).stem)
                os.makedirs(vis_dir, exist_ok=True)
            
            # YOLO OBB'lerini oku (köşe noktası olarak)
            yolo_txt_path = os.path.join('yolo_format', Path(image_name).stem + '.jpg.txt')
            yolo_obb_points = self.get_yolo_obb_points_for_image(yolo_txt_path, img_width, img_height)
            
            for idx, label in enumerate(enhanced_obb_labels):
                try:
                    # Trapezoid köşe noktalarını al
                    trapezoid_corners = label.get("obb_coordinates", [])
                    if len(trapezoid_corners) != 4:
                        logger.warning(f"Invalid trapezoid corners for object {idx}, skipping")
                        continue

                    # Trapezoid köşe noktalarını denormalize et (görselleştirme ve mask için)
                    denorm_corners = [(x * img_width, y * img_height) for x, y in trapezoid_corners]
                    # Optimize OBB'yi hull_sweep ile bul
                    obb = self.trapezoid_to_obb(trapezoid_corners, img_shape=(img_height, img_width))
                    center_x, center_y, width, height, angle = obb
                    obb_px = (center_x, center_y, width, height, angle)
                    obb_corners = self.obb_to_polygon(obb_px)
                    # IoU hesaplamaları 
                    trapezoid_iou = label.get("new_iou", 0.0)
                    obb_iou = 0.0
                    if image is not None:
                        # Orijinal maskeyi oluştur (trapezoid'den)
                        trapezoid_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.fillPoly(trapezoid_mask, [np.array(denorm_corners, dtype=np.int32)], 255)
                        # OBB maskesini oluştur
                        obb_mask = self.obb_to_mask(obb_px, (img_height, img_width))
                        # IoU hesapla
                        obb_iou = self.calculate_mask_iou(trapezoid_mask, obb_mask)
                        # Orijinal OBB'yi yolo labelstan al 
                        if idx < len(yolo_obb_points):
                            orig_obb_corners = yolo_obb_points[idx]
                            orig_obb_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            cv2.fillPoly(orig_obb_mask, [np.array(orig_obb_corners, dtype=np.int32)], 255)
                            orig_obb_iou = self.calculate_mask_iou(trapezoid_mask, orig_obb_mask)
                        else:
                            orig_obb_corners = []
                            orig_obb_iou = 0.0
                    else:
                        orig_obb_corners = []
                        orig_obb_iou = 0.0
                    # Görselleştirme oluştur
                    if self.visualize and image is not None:
                        vis_filename = f"shelf_{idx + 1}_trapezoid_vs_obb.png"
                        vis_path = os.path.join(vis_dir, vis_filename)
                        self.create_comparison_visualization(
                            image, trapezoid_mask, orig_obb_corners, denorm_corners, obb_corners,
                            idx, orig_obb_iou, trapezoid_iou, obb_iou, vis_path
                        )
                    # OBB formatında label oluştur 
                    norm_center_x = center_x / img_width
                    norm_center_y = center_y / img_height
                    norm_width = width / img_width
                    norm_height = height / img_height
                    converted_obb_label = {
                        "class_id": label.get("class_id", 1),
                        "class_name": label.get("class_name", "shelf"),
                        "obb_coordinates": [norm_center_x, norm_center_y, norm_width, norm_height, angle],
                        "confidence": label.get("confidence", 1.0),
                        "trapezoid_iou": float(trapezoid_iou),
                        "obb_iou": float(obb_iou),
                        "iou_loss": float(trapezoid_iou - obb_iou)
                    }

                    converted_results["converted_obb_labels"].append(converted_obb_label)
                    trapezoid_ious.append(trapezoid_iou)
                    obb_ious.append(obb_iou)
                    converted_results["conversion_stats"]["successfully_converted"] += 1

                except Exception as e:
                    logger.error(f"Error converting object {idx}: {e}")
                    converted_results["conversion_stats"]["failed_conversions"] += 1
                    continue
            
            # İstatistikleri tamamla
            if trapezoid_ious:
                converted_results["conversion_stats"]["average_trapezoid_iou"] = np.mean(trapezoid_ious)
                converted_results["conversion_stats"]["average_obb_iou"] = np.mean(obb_ious)
                converted_results["conversion_stats"]["average_iou_loss"] = np.mean(trapezoid_ious) - np.mean(obb_ious)
            
            # Sonuçları kaydet
            output_filename = os.path.splitext(os.path.basename(json_path))[0] + "_converted_to_obb.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Converted OBB results saved to: {output_path}")
            return converted_results
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            return {"error": str(e)}
    
    def process_dataset(self, input_dir: str, output_dir: str, image_dir: str = None) -> Dict[str, Any]:
        """
        Tüm dataset'i işle
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.visualize:
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # JSON dosyalarını bul
        json_files = []
        for file in os.listdir(input_dir):
            if file.endswith('_enhanced_trapezoid.json'):
                json_path = os.path.join(input_dir, file)
                json_files.append(json_path)
        
        logger.info(f"Found {len(json_files)} trapezoid JSON files to convert")
        
        total_results = {
            "conversion_timestamp": datetime.now().isoformat(),
            "total_files": len(json_files),
            "successfully_converted": 0,
            "failed_conversions": 0,
            "overall_stats": {
                "total_objects_converted": 0,
                "average_trapezoid_iou": 0.0,
                "average_obb_iou": 0.0,
                "average_iou_loss": 0.0
            },
            "converted_files": []
        }
        
        all_trapezoid_ious = []
        all_obb_ious = []
        
        for idx, json_path in enumerate(json_files):
            logger.info(f"\nConverting {idx + 1}/{len(json_files)}: {os.path.basename(json_path)}")
            
            result = self.process_single_json(json_path, output_dir, image_dir)
            
            if "error" not in result:
                total_results["successfully_converted"] += 1
                total_results["converted_files"].append({
                    "filename": os.path.basename(json_path),
                    "converted_objects": result["conversion_stats"]["successfully_converted"],
                    "failed_objects": result["conversion_stats"]["failed_conversions"],
                    "average_trapezoid_iou": result["conversion_stats"]["average_trapezoid_iou"],
                    "average_obb_iou": result["conversion_stats"]["average_obb_iou"],
                    "average_iou_loss": result["conversion_stats"]["average_iou_loss"]
                })
                
                total_results["overall_stats"]["total_objects_converted"] += result["conversion_stats"]["successfully_converted"]
                
                if result["conversion_stats"]["average_trapezoid_iou"] > 0:
                    all_trapezoid_ious.append(result["conversion_stats"]["average_trapezoid_iou"])
                    all_obb_ious.append(result["conversion_stats"]["average_obb_iou"])
                    
            else:
                total_results["failed_conversions"] += 1
                logger.error(f"Failed to convert {json_path}: {result['error']}")
        
        # Genel ortalamalar
        if all_trapezoid_ious:
            total_results["overall_stats"]["average_trapezoid_iou"] = np.mean(all_trapezoid_ious)
            total_results["overall_stats"]["average_obb_iou"] = np.mean(all_obb_ious)
            total_results["overall_stats"]["average_iou_loss"] = np.mean(all_trapezoid_ious) - np.mean(all_obb_ious)
        
        # Toplam sonuçları kaydet
        summary_path = os.path.join(output_dir, "trapezoid_to_obb_conversion_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(total_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAPEZOID TO OBB CONVERSION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total files converted: {total_results['successfully_converted']}/{total_results['total_files']}")
        logger.info(f"Total objects converted: {total_results['overall_stats']['total_objects_converted']}")
        logger.info(f"Average trapezoid IoU: {total_results['overall_stats']['average_trapezoid_iou']:.4f}")
        logger.info(f"Average OBB IoU: {total_results['overall_stats']['average_obb_iou']:.4f}")
        logger.info(f"Average IoU loss: {total_results['overall_stats']['average_iou_loss']:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return total_results

def main():
    """
    Ana fonksiyon
    """
    # Konfigürasyon
    INPUT_DIR = "new_dataset/trapezoid_dataset"  # Trapezoid JSON'ların bulunduğu klasör
    OUTPUT_DIR = "new_dataset/trapezoid_to_obb_converted"  # Çıktı klasörü
    IMAGE_DIR = "new_dataset/worst_dataset_1000"  # Görüntülerin bulunduğu klasör 
    VISUALIZE = True  # Karşılaştırmalı görselleştirme oluştur
    
    try:
        # Converter'ı başlat
        logger.info("Initializing Trapezoid to OBB Converter...")
        converter = TrapezoidToOBBConverter(visualize=VISUALIZE)
        
        # Dataset'i işle
        logger.info(f"Starting trapezoid to OBB conversion from: {INPUT_DIR}")
        results = converter.process_dataset(INPUT_DIR, OUTPUT_DIR, IMAGE_DIR)
        
        logger.info("Trapezoid to OBB conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 