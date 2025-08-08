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

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OBBGenerator:
    def __init__(self, visualize: bool = False):
        """
        Oriented Bounding Box üretici
        
        Args:
            visualize: Karşılaştırmalı görselleştirme oluştur
        """
        self.visualize = visualize
        logger.info("OBB Generator initialized")
    
    def polygon_to_mask(self, polygon_points: List[int], img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Polygon koordinatlarını binary mask'e çevir
        """
        points = []
        for i in range(0, len(polygon_points), 2):
            if i + 1 < len(polygon_points):
                points.append([polygon_points[i], polygon_points[i + 1]])
        
        if len(points) < 3:
            logger.warning("Not enough points for polygon, skipping")
            return np.zeros(img_shape, dtype=np.uint8)
        
        mask = np.zeros(img_shape, dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        
        return mask
    
    def mask_to_obb(self, mask: np.ndarray) -> Tuple[Tuple[float, float, float, float, float], List[Tuple[int, int]]]:
        """
        Binary mask'ten Oriented Bounding Box çıkar
        
        Args:
            mask: Binary mask
            
        Returns:
            (center_x, center_y, width, height, angle), corner_points
        """
        # Mask'te beyaz pikselleri bul
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            return (0, 0, 0, 0, 0), []
        
       
        points = np.array([(coord[1], coord[0]) for coord in coords])
        
        # Minimum area rectangle bul
        rect = cv2.minAreaRect(points.astype(np.float32))
        
        # Rectangle bilgilerini çıkar
        center, (width, height), angle = rect
        
        # Corner points'leri hesapla
        box_points = cv2.boxPoints(rect)
        box_points = np.int32(box_points)
        
        return (center[0], center[1], width, height, angle), box_points.tolist()
    
    def obb_to_polygon(self, obb: Tuple[float, float, float, float, float]) -> List[Tuple[int, int]]:
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
            rotated_corners.append((int(rot_x), int(rot_y)))
        
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
    
    def calculate_obb_iou(self, obb1: Tuple[float, float, float, float, float], 
                         obb2: Tuple[float, float, float, float, float], 
                         img_shape: Tuple[int, int]) -> float:
        """
        İki OBB arasındaki IoU hesapla
        """
        mask1 = self.obb_to_mask(obb1, img_shape)
        mask2 = self.obb_to_mask(obb2, img_shape)
        
        return self.calculate_mask_iou(mask1, mask2)
    
    def normalize_obb(self, obb: Tuple[float, float, float, float, float], 
                     img_width: int, img_height: int) -> Tuple[float, float, float, float, float]:
        """
        OBB'yi normalize et (0-1 arası)
        """
        center_x, center_y, width, height, angle = obb
        
        norm_center_x = center_x / img_width
        norm_center_y = center_y / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        
        return (norm_center_x, norm_center_y, norm_width, norm_height, angle)
    
    
    def mask_to_trapezoid(self, mask: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Maskeye en iyi uyan yamuk (trapezoid) fit eder. 
        """
        import numpy as np
        import cv2
        def is_trapezoid(pts):
            def direction(p1, p2):
                v = p2 - p1
                norm = np.linalg.norm(v)
                if norm == 0:
                    return v
                return v / norm
            d1 = direction(pts[0], pts[1])
            d2 = direction(pts[2], pts[3])
            d3 = direction(pts[1], pts[2])
            d4 = direction(pts[3], pts[0])
            return (np.abs(np.dot(d1, d2)) > 0.99) or (np.abs(np.dot(d3, d4)) > 0.99)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((4, 2)), []
        contour = max(contours, key=cv2.contourArea)
        contour = contour.reshape(-1, 2)
        best_iou = 0
        best_trap = None
        hull = cv2.convexHull(contour)
        hull = hull.reshape(-1, 2)
        # 1) Hull sadeleştirme
        hull = cv2.approxPolyDP(hull, 0.001 * cv2.arcLength(hull, True), True)
        hull = hull.reshape(-1, 2)
        n = len(hull)
        found_early = False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        pts = np.array([hull[i], hull[j], hull[k], hull[l]])
                        if is_trapezoid(pts):
                            trap_mask = np.zeros_like(mask)
                            cv2.fillPoly(trap_mask, [pts.astype(np.int32)], 255)
                            intersection = np.logical_and(mask > 0, trap_mask > 0).sum()
                            union = np.logical_or(mask > 0, trap_mask > 0).sum()
                            iou = intersection / union if union > 0 else 0
                            if iou > best_iou:
                                best_iou = iou
                                best_trap = pts
                                # 2) Early stopping: Yeterince iyi bir fit bulunduysa çık
                                if best_iou >= 0.98:
                                    found_early = True
                                    break
                    if found_early:
                        break
                if found_early:
                    break
            if found_early:
                break
        if best_trap is None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            best_trap = box.astype(np.int32)
        return best_trap, best_trap.tolist()

    def create_single_shelf_visualization(self, image: np.ndarray, original_mask: np.ndarray,
                                        old_obb: Tuple[float, float, float, float, float],
                                        new_poly: list,
                                        shelf_id: int, old_iou: float, new_iou: float,
                                        save_path: str, fit_method: str = 'obb') -> None:
        """
        Tek bir shelf objesi için karşılaştırmalı görselleştirme oluştur
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'Shelf {shelf_id + 1} - Old IoU: {old_iou:.3f} | New IoU: {new_iou:.3f} | Gain: +{new_iou-old_iou:.3f}', 
                        fontsize=16, fontweight='bold')
            # Sol taraf - Eski OBB
            ax1.imshow(image)
            ax1.set_title(f'Old OBB (IoU: {old_iou:.3f})', fontsize=14, fontweight='bold')
            ax1.axis('off')
            # Sağ taraf - Yeni OBB veya Quadrilateral
            ax2.imshow(image)
            ax2.set_title(f'New {fit_method.upper()} (IoU: {new_iou:.3f})', fontsize=14, fontweight='bold')
            ax2.axis('off')
            # Maskeleri göster
            if original_mask.sum() > 0:
                mask_rgba = np.zeros((*original_mask.shape, 4))
                mask_rgba[original_mask > 0] = [1, 0.2, 0.2, 0.4]
                ax1.imshow(mask_rgba)
                ax2.imshow(mask_rgba)
            # Eski OBB çiz (sol)
            if old_obb is not None:
                old_corners = self.obb_to_polygon(old_obb)
                if len(old_corners) >= 3:
                    obb_polygon = Polygon(old_corners, fill=False, edgecolor='darkgreen', 
                                        linewidth=2, linestyle='-', alpha=0.9)
                    ax1.add_patch(obb_polygon)
            # Yeni OBB veya Quadrilateral çiz (sağ)
            if new_poly is not None and len(new_poly) >= 3:
                poly = Polygon(new_poly, fill=False, edgecolor='darkgreen', 
                            linewidth=2, linestyle='-', alpha=0.9)
                ax2.add_patch(poly)
            # IoU değerlerini yaz
            ax1.text(0.02, 0.02, f'IoU: {old_iou:.3f}', transform=ax1.transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='darkred', alpha=0.8))
            ax2.text(0.02, 0.02, f'IoU: {new_iou:.3f}', transform=ax2.transAxes, 
                    fontsize=14, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='darkgreen', alpha=0.8))
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Shelf {shelf_id + 1} visualization saved: {os.path.basename(save_path)}")
        except Exception as e:
            logger.error(f"Error creating shelf {shelf_id + 1} visualization: {e}")
    
    def parse_yolo_obb(self, yolo_line: str, img_width: int, img_height: int) -> Tuple[float, float, float, float, float]:
        """
        YOLO formatındaki bir satırı (8 koordinat) OBB'ye çevirir (center_x, center_y, width, height, angle)
        """
        parts = yolo_line.strip().split()
        # Sınıf id'sini atla, kalan 8 değer köşe noktaları
        coords = list(map(float, parts[1:]))
        # 4 köşe noktası (x1,y1,x2,y2,x3,y3,x4,y4) [normalize edilmiş]
        points = np.array(coords, dtype=np.float32).reshape(4, 2)
        # Denormalize et
        points[:, 0] *= img_width
        points[:, 1] *= img_height
        # cv2.minAreaRect ile OBB parametrelerini bul
        rect = cv2.minAreaRect(points)
        (center_x, center_y), (width, height), angle = rect
        return (center_x, center_y, width, height, angle)

    def get_yolo_obbs_for_image(self, yolo_txt_path: str, img_width: int, img_height: int) -> list:
        """
        Bir yolo txt dosyasındaki tüm OBB'leri (sadece shelf class_id=1) döndürür
        """
        obbs = []
        with open(yolo_txt_path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                if line.startswith('1 '):  # Sadece shelf class_id=1
                    obb = self.parse_yolo_obb(line, img_width, img_height)
                    obbs.append(obb)
        return obbs
    
    def process_single_image(self, image_path: str, json_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Tek bir görüntüyü yamuk (trapezoid) fit ile işle
        """
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image.shape[:2]
            # JSON'u yükle
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Sadece shelf objelerini filtrele
            shelf_labels = [label for label in data.get("segmentation_labels", []) if label.get("className", "").lower() == "shelf"]
            if len(shelf_labels) == 0:
                logger.info("No shelf objects found, skipping")
                return {"error": "No shelf objects found"}
            # Yolo OBB'leri oku
            yolo_txt_path = os.path.join('yolo_format', Path(image_path).stem + '.jpg.txt')
            yolo_obbs = self.get_yolo_obbs_for_image(yolo_txt_path, img_width, img_height)
            if len(yolo_obbs) < len(shelf_labels):
                logger.warning(f"Yolo OBB sayısı ile shelf label sayısı eşleşmiyor: {len(yolo_obbs)} vs {len(shelf_labels)}")
            enhanced_results = {
                "image_name": data.get("image_name", os.path.basename(image_path)),
                "fit_method": "trapezoid",
                "enhanced_obb_labels": [],
                "obb_improvements": [],
                "processing_stats": {
                    "total_objects": len(shelf_labels),
                    "successfully_enhanced": 0,
                    "failed_enhancements": 0,
                    "average_original_iou": 0.0,
                    "average_improved_iou": 0.0,
                    "average_iou_improvement": 0.0
                }
            }
            original_ious = []
            improved_ious = []
            # Görselleştirme klasörü
            if self.visualize:
                vis_dir = os.path.join(output_dir, "visualizations", Path(image_path).stem)
                os.makedirs(vis_dir, exist_ok=True)
            # Her shelf label'ını işle
            for idx, seg_label in enumerate(shelf_labels):
                try:
                    logger.info(f"Processing shelf object {idx + 1}/{len(shelf_labels)}")
                    # Polygon'dan mask oluştur
                    points = seg_label.get("points", [])
                    if len(points) < 6:
                        logger.warning(f"Not enough points in segment {idx}, skipping")
                        continue
                    original_mask = self.polygon_to_mask(points, (img_height, img_width))
                    if original_mask.sum() == 0:
                        logger.warning(f"Empty mask for segment {idx}, skipping")
                        continue
                    # Eski OBB (YOLO format) - index eşleşmesiyle
                    if idx < len(yolo_obbs):
                        old_obb = yolo_obbs[idx]
                    else:
                        logger.warning(f"Yolo OBB bulunamadı, maskten üretilecek (idx={idx})")
                        old_obb, _ = self.mask_to_obb(original_mask)
                    old_obb_mask = self.obb_to_mask(old_obb, (img_height, img_width))
                    old_iou = self.calculate_mask_iou(original_mask, old_obb_mask)
                    # Sadece trapezoid fit
                    trap, trap_list = self.mask_to_trapezoid(original_mask)
                    new_obb_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    cv2.fillPoly(new_obb_mask, [np.array(trap, dtype=np.int32)], 255)
                    new_iou = self.calculate_mask_iou(original_mask, new_obb_mask)
                    new_poly = trap_list
                    normalized_obb = [(float(x)/img_width, float(y)/img_height) for x, y in trap]
                    # Görselleştirme oluştur
                    if self.visualize:
                        vis_filename = f"shelf_{idx + 1}_obb_comparison.png"
                        vis_path = os.path.join(vis_dir, vis_filename)
                        self.create_single_shelf_visualization(
                            image_rgb, original_mask, old_obb, new_poly, 
                            idx, old_iou, new_iou, vis_path, fit_method="trapezoid"
                        )
                    # OBB formatında label oluştur
                    enhanced_obb_label = {
                        "class_id": seg_label.get("classId", 1),
                        "class_name": seg_label.get("className", "shelf"),
                        "obb_coordinates": list(normalized_obb),
                        "confidence": float(seg_label.get("probability", 100) / 100.0),
                        "old_iou": float(old_iou),
                        "new_iou": float(new_iou),
                        "iou_gain": float(new_iou - old_iou)
                    }
                    enhanced_results["enhanced_obb_labels"].append(enhanced_obb_label)
                    # İstatistikler
                    improvement = {
                        "object_id": idx,
                        "class_name": seg_label.get("className", "shelf"),
                        "old_iou": float(old_iou),
                        "new_iou": float(new_iou),
                        "iou_gain": float(new_iou - old_iou)
                    }
                    enhanced_results["obb_improvements"].append(improvement)
                    original_ious.append(old_iou)
                    improved_ious.append(new_iou)
                    enhanced_results["processing_stats"]["successfully_enhanced"] += 1
                except Exception as e:
                    logger.error(f"Error processing segment {idx}: {e}")
                    enhanced_results["processing_stats"]["failed_enhancements"] += 1
                    continue
            # İstatistikleri tamamla
            if original_ious:
                enhanced_results["processing_stats"]["average_original_iou"] = np.mean(original_ious)
                enhanced_results["processing_stats"]["average_improved_iou"] = np.mean(improved_ious)
                enhanced_results["processing_stats"]["average_iou_improvement"] = np.mean(improved_ious) - np.mean(original_ious)
            # Sonuçları kaydet
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_enhanced_trapezoid.json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Enhanced trapezoid results saved to: {output_path}")
            return enhanced_results
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {"error": str(e)}
    
    def process_dataset(self, dataset_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Tüm dataset'i sadece yamuk (trapezoid) fit ile işle. Daha önce işlenmiş görselleri ve jsonları atlar.
        """
        if output_dir is None:
            output_dir = os.path.join(dataset_dir, "enhanced_obb")
        os.makedirs(output_dir, exist_ok=True)
        if self.visualize:
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for file in os.listdir(dataset_dir):
            if Path(file).suffix.lower() in image_extensions:
                image_path = os.path.join(dataset_dir, file)
                json_path = os.path.join(dataset_dir, Path(file).stem + '.json')
                if os.path.exists(json_path):
                    image_files.append((image_path, json_path))
                else:
                    logger.warning(f"JSON file not found for {file}")
        logger.info(f"Found {len(image_files)} image-json pairs to process")
       
        already_processed = set()
        for file in os.listdir(output_dir):
            if file.endswith('_enhanced_trapezoid.json'):
                stem = file.replace('_enhanced_trapezoid.json', '')
                already_processed.add(stem)
        logger.info(f"{len(already_processed)} images already processed, will be skipped.")
        total_results = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "successfully_processed": 0,
            "failed_processing": 0,
            "overall_stats": {
                "total_objects_processed": 0,
                "total_objects_enhanced": 0,
                "average_original_iou": 0.0,
                "average_improved_iou": 0.0,
                "average_iou_improvement": 0.0
            },
            "processed_files": []
        }
        all_original_ious = []
        all_improved_ious = []
        for idx, (image_path, json_path) in enumerate(image_files):
            image_stem = Path(image_path).stem
            if image_stem in already_processed:
                logger.info(f"Skipping already processed image: {image_stem}")
                continue
            logger.info(f"\nProcessing {idx + 1}/{len(image_files)}: {os.path.basename(image_path)}")
            result = self.process_single_image(image_path, json_path, output_dir)
            if "error" not in result:
                total_results["successfully_processed"] += 1
                total_results["processed_files"].append({
                    "filename": os.path.basename(image_path),
                    "enhanced_objects": result["processing_stats"]["successfully_enhanced"],
                    "failed_objects": result["processing_stats"]["failed_enhancements"],
                    "average_original_iou": result["processing_stats"]["average_original_iou"],
                    "average_improved_iou": result["processing_stats"]["average_improved_iou"],
                    "average_improvement": result["processing_stats"]["average_iou_improvement"]
                })
                total_results["overall_stats"]["total_objects_processed"] += result["processing_stats"]["total_objects"]
                total_results["overall_stats"]["total_objects_enhanced"] += result["processing_stats"]["successfully_enhanced"]
                if result["processing_stats"]["average_original_iou"] > 0:
                    all_original_ious.append(result["processing_stats"]["average_original_iou"])
                    all_improved_ious.append(result["processing_stats"]["average_improved_iou"])
            else:
                total_results["failed_processing"] += 1
                logger.error(f"Failed to process {image_path}: {result['error']}")
        if all_original_ious:
            total_results["overall_stats"]["average_original_iou"] = np.mean(all_original_ious)
            total_results["overall_stats"]["average_improved_iou"] = np.mean(all_improved_ious)
            total_results["overall_stats"]["average_iou_improvement"] = np.mean(all_improved_ious) - np.mean(all_original_ious)
        summary_path = os.path.join(output_dir, "obb_processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(total_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n{'='*60}")
        logger.info(f"OBB PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total images processed: {total_results['successfully_processed']}/{total_results['total_images']}")
        logger.info(f"Total objects enhanced: {total_results['overall_stats']['total_objects_enhanced']}")
        logger.info(f"Average original IoU: {total_results['overall_stats']['average_original_iou']:.4f}")
        logger.info(f"Average improved IoU: {total_results['overall_stats']['average_improved_iou']:.4f}")
        logger.info(f"Average IoU improvement: {total_results['overall_stats']['average_iou_improvement']:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Summary saved to: {summary_path}")
        return total_results

def main():
    """
    Ana fonksiyon - script'i çalıştırmak için
    """
    # Konfigürasyon
    DATASET_DIR = "new_dataset/worst_dataset_pure"  # Dataset klasörü yolu
    OUTPUT_DIR = "new_dataset/trapezoid_dataset"  # Çıktı klasörü
    VISUALIZE = False  # Karşılaştırmalı görselleştirme oluştur
    
    try:
        # OBB Generator'ı başlat
        logger.info("Initializing OBB Generator...")
        generator = OBBGenerator(visualize=VISUALIZE)
        
        # Dataset'i işle
        logger.info(f"Starting OBB dataset processing from: {DATASET_DIR}")
        results = generator.process_dataset(DATASET_DIR, OUTPUT_DIR)
        
        logger.info("OBB processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise

if __name__ == "__main__":
    main()