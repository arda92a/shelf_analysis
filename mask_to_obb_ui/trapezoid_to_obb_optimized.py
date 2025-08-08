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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from numba import jit, prange
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@jit(nopython=True, cache=True)
def fast_polygon_area(points):
    """Numba ile hızlandırılmış polygon alan hesabı (Shoelace formula)"""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

@jit(nopython=True, cache=True)
def fast_point_in_polygon(point, polygon):
    """Numba ile hızlandırılmış point-in-polygon testi"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

@jit(nopython=True, cache=True)
def fast_mask_iou(mask1_flat, mask2_flat):
    """Numba ile hızlandırılmış IoU hesabı"""
    intersection = 0
    union = 0
    
    for i in prange(len(mask1_flat)):
        m1 = mask1_flat[i] > 0
        m2 = mask2_flat[i] > 0
        if m1 and m2:
            intersection += 1
        if m1 or m2:
            union += 1
    
    return intersection / union if union > 0 else 0.0

@jit(nopython=True, cache=True)
def rotate_points(points, angle_rad, center_x, center_y):
    """Numba ile hızlandırılmış nokta döndürme"""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotated = np.empty_like(points)
    for i in range(len(points)):
        x = points[i][0] - center_x
        y = points[i][1] - center_y
        rotated[i][0] = x * cos_a - y * sin_a + center_x
        rotated[i][1] = x * sin_a + y * cos_a + center_y
    
    return rotated

class OptimizedTrapezoidToOBBConverter:
    def __init__(self, visualize: bool = True, n_workers: int = None, use_gpu: bool = False):
        """
        Optimize edilmiş Trapezoid köşe noktalarını OBB'ye çeviren converter
        """
        self.visualize = visualize
        self.n_workers = n_workers or min(8, mp.cpu_count())
        self.use_gpu = use_gpu
        
        # Önbellek boyutlarını artır
        self.trapezoid_to_obb_cache = {}
        self.mask_cache = {}
        
        logger.info(f"Optimized Trapezoid to OBB Converter initialized with {self.n_workers} workers")
    
    @lru_cache(maxsize=1024)
    def _cached_rotation_matrix(self, angle: float, center_x: float, center_y: float):
        """Döndürme matrislerini önbellekle"""
        return cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    def trapezoid_to_obb_optimized(self, corners: List[Tuple[float, float]], 
                                 img_shape: tuple = (640, 640), 
                                 angle_step: float = 2.0,  # Daha büyük adım
                                 early_stop_threshold: float = 0.95) -> Tuple[float, float, float, float, float]:
        """
        Optimize edilmiş trapezoid'den OBB'ye çevirme
        """
        # Önbellek kontrolü - corners'ı hassas tuple'a çevir
        corners_tuple = tuple((round(x, 8), round(y, 8)) for x, y in corners)
        cache_key = (corners_tuple, img_shape, angle_step, early_stop_threshold)
        
        # DEBUG: Her unique corner için cache miss olduğundan emin ol
        # Cache'i geçici olarak devre dışı bırak
        # if cache_key in self.trapezoid_to_obb_cache:
        #     logger.debug(f"Cache hit for corners: {corners_tuple[:2]}...")
        #     return self.trapezoid_to_obb_cache[cache_key]
        
        h, w = img_shape
        pts = np.array([(x * w, y * h) for x, y in corners], dtype=np.float32)
        
        # Convex hull yerine direkt minimum enclosing rectangle deneme
        hull = cv2.convexHull(pts.astype(np.int32))
        hull_points = hull.reshape(-1, 2).astype(np.float32)
        
        # İlk tahmin olarak cv2.minAreaRect kullan
        initial_rect = cv2.minAreaRect(hull_points)
        best_obb = initial_rect
        best_iou = 0.0
        
        # Daha az açı deneme ile optimizasyon
        center_x, center_y = w/2, h/2
        
        # Mask'i bir kere oluştur ve cache'le
        mask_key = (corners_tuple, img_shape)
        if mask_key in self.mask_cache:
            mask = self.mask_cache[mask_key]
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            self.mask_cache[mask_key] = mask
        
        mask_flat = mask.flatten()
        
        # Daha az açı ile sweep (performans için)
        for angle in np.arange(-90, 90, angle_step):
            # Hull'u döndür
            angle_rad = np.deg2rad(angle)
            hull_rot = rotate_points(hull_points, angle_rad, center_x, center_y)
            
            # Bounding box bul
            x_min, y_min = np.min(hull_rot, axis=0)
            x_max, y_max = np.max(hull_rot, axis=0)
            
            bw, bh = x_max - x_min, y_max - y_min
            box = np.array([
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ], dtype=np.float32)
            
            # Geri döndür
            box_rot = rotate_points(box, -angle_rad, center_x, center_y)
            
            # Mask oluştur ve IoU hesapla
            obb_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obb_mask, [box_rot.astype(np.int32)], 255)
            
            # Numba ile hızlandırılmış IoU
            iou = fast_mask_iou(mask_flat, obb_mask.flatten())
            
            if iou > best_iou:
                best_iou = iou
                rect = cv2.minAreaRect(box_rot)
                best_obb = rect
                
                # Erken durma - yeterince iyi IoU bulundu
                if iou > early_stop_threshold:
                    break
        
        # Cache'e kaydet - sadece debug amaçlı
        # self.trapezoid_to_obb_cache[cache_key] = best_obb
        
        # Cache boyutunu kontrol et
        if len(self.trapezoid_to_obb_cache) > 2000:
            # En eski %30'u temizle
            keys_to_remove = list(self.trapezoid_to_obb_cache.keys())[:600]
            for key in keys_to_remove:
                del self.trapezoid_to_obb_cache[key]
        
        return best_obb
    
    def obb_to_polygon_fast(self, obb: Tuple[float, float, float, float, float]) -> List[Tuple[float, float]]:
        """
        Hızlandırılmış OBB'den polygon hesaplama
        """
        center_x, center_y, width, height, angle = obb
        angle_rad = np.deg2rad(angle)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        w_half = width * 0.5
        h_half = height * 0.5
        
        # Vektörize edilmiş hesaplama
        corners = np.array([
            [-w_half, -h_half], [w_half, -h_half], 
            [w_half, h_half], [-w_half, h_half]
        ])
        
        # Rotation matrix
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = corners @ rotation.T
        rotated[:, 0] += center_x
        rotated[:, 1] += center_y
        
        return rotated.tolist()
    
    def calculate_mask_iou_fast(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Hızlandırılmış IoU hesabı
        """
        return fast_mask_iou(mask1.flatten(), mask2.flatten())
    
    def process_single_json_parallel(self, args):
        """
        Paralel işleme için wrapper function
        """
        json_path, output_dir, image_dir = args
        return self.process_single_json(json_path, output_dir, image_dir)
    
    def process_single_json(self, json_path: str, output_dir: str, image_dir: str = None) -> Dict[str, Any]:
        """
        Optimize edilmiş tek JSON dosyası işleme
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_name = data.get("image_name", "")
            if not image_name:
                return {"error": "No image_name found"}
            
            # Görüntü boyutlarını dinamik olarak al
            image = None
            img_height, img_width = 2000, 1500  # Varsayılan boyutlar (fallback)
            
            # Görüntüyü yükle ve boyutları dinamik olarak al
            if image_dir:
                image_path = os.path.join(image_dir, image_name)
                if os.path.exists(image_path):
                    temp_image = cv2.imread(image_path)
                    if temp_image is not None:
                        # Görüntü boyutlarını dinamik olarak al
                        img_height, img_width = temp_image.shape[:2]
                        
                        # 1500x2000 boyutundaki görselleri atla
                        if img_width == 1500 and img_height == 2000:
                            logger.info(f"Skipping {image_name} - already correct dimensions (1500x2000)")
                            return {"skipped": f"Image {image_name} has correct dimensions (1500x2000)"}
                        
                        logger.info(f"Processing {image_name} - dimensions: {img_width}x{img_height}")
                        
                        # Sadece görselleştirme gerektiğinde RGB'ye çevir
                        if self.visualize:
                            image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
                        else:
                            image = temp_image  # IoU hesaplaması için BGR kalabilir
                    else:
                        logger.warning(f"Could not load image: {image_path}")
                else:
                    logger.warning(f"Image file not found: {image_path}")
            else:
                logger.warning("No image_dir provided, using default dimensions")
            
            enhanced_obb_labels = data.get("enhanced_obb_labels", [])
            if not enhanced_obb_labels:
                return {"error": "No enhanced_obb_labels found"}
            
            converted_results = {
                "image_name": image_name,
                "conversion_method": "optimized_trapezoid_to_obb_hull_sweep",
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
            
            # YOLO OBB'lerini oku - sadece görselleştirme gerektiğinde
            yolo_obb_points = []
            if self.visualize and image is not None:
                vis_dir = os.path.join(output_dir, "visualizations", Path(image_name).stem)
                os.makedirs(vis_dir, exist_ok=True)
                
                yolo_txt_path = os.path.join('yolo_format', Path(image_name).stem + '.jpg.txt')
                yolo_obb_points = self.get_yolo_obb_points_for_image(yolo_txt_path, img_width, img_height)
            
            # Batch processing için tüm köşe noktalarını topla
            all_corners = []
            valid_indices = []
            
            for idx, label in enumerate(enhanced_obb_labels):
                trapezoid_corners = label.get("obb_coordinates", [])
                if len(trapezoid_corners) == 4:
                    all_corners.append(trapezoid_corners)
                    valid_indices.append(idx)
            
            # Batch OBB conversion
            obbs = []
            for corners in all_corners:
                obb = self.trapezoid_to_obb_optimized(
                    corners, 
                    img_shape=(img_height, img_width),
                    angle_step=3.0,  # Daha büyük adım daha hızlı
                    early_stop_threshold=0.92
                )
                obbs.append(obb)
            
            # Sonuçları işle
            for i, (idx, obb) in enumerate(zip(valid_indices, obbs)):
                try:
                    label = enhanced_obb_labels[idx]
                    trapezoid_corners = all_corners[i]
                    
                    # cv2.minAreaRect döndürdüğü değerleri doğru şekilde al
                    if len(obb) == 3:  # (center, (width, height), angle)
                        (center_x, center_y), (width, height), angle = obb
                    else:  # (center_x, center_y, width, height, angle)
                        center_x, center_y, width, height, angle = obb
                    
                    obb_px = (center_x, center_y, width, height, angle)
                    
                    # IoU hesaplamaları - her zaman yap
                    trapezoid_iou = label.get("new_iou", 0.0)
                    obb_iou = 0.0
                    
                    # IoU hesaplaması için görüntü boyutları gerekli
                    if img_height > 0 and img_width > 0:
                        denorm_corners = [(x * img_width, y * img_height) for x, y in trapezoid_corners]
                        obb_corners = self.obb_to_polygon_fast(obb_px)
                        
                        # Mask'ler
                        trapezoid_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.fillPoly(trapezoid_mask, [np.array(denorm_corners, dtype=np.int32)], 255)
                        
                        obb_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.fillPoly(obb_mask, [np.array(obb_corners, dtype=np.int32)], 255)
                        
                        obb_iou = self.calculate_mask_iou_fast(trapezoid_mask, obb_mask)
                    
                    # Görselleştirme (sadece visualize=True ise)
                    if self.visualize and image is not None:
                        orig_obb_corners = yolo_obb_points[i] if i < len(yolo_obb_points) else []
                        orig_obb_iou = 0.0
                        
                        if orig_obb_corners:
                            orig_obb_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            cv2.fillPoly(orig_obb_mask, [np.array(orig_obb_corners, dtype=np.int32)], 255)
                            orig_obb_iou = self.calculate_mask_iou_fast(trapezoid_mask, orig_obb_mask)
                        
                        vis_filename = f"shelf_{idx + 1}_trapezoid_vs_obb.png"
                        vis_path = os.path.join(vis_dir, vis_filename)
                        self.create_comparison_visualization(
                            image, trapezoid_mask, orig_obb_corners, denorm_corners, obb_corners,
                            idx, orig_obb_iou, trapezoid_iou, obb_iou, vis_path
                        )
                    
                    # Normalize edilmiş koordinatlar
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
            
            # İstatistik hesapları
            if trapezoid_ious:
                converted_results["conversion_stats"]["average_trapezoid_iou"] = float(np.mean(trapezoid_ious))
                converted_results["conversion_stats"]["average_obb_iou"] = float(np.mean(obb_ious))
                converted_results["conversion_stats"]["average_iou_loss"] = float(np.mean(trapezoid_ious) - np.mean(obb_ious))
            
            # Sonuç kaydetme
            output_filename = os.path.splitext(os.path.basename(json_path))[0] + "_converted_to_obb.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
            return converted_results
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            return {"error": str(e)}
    
    def get_yolo_obb_points_for_image(self, yolo_txt_path: str, img_width: int, img_height: int) -> list:
        """YOLO OBB noktalarını hızlı okuma"""
        obb_points = []
        if not os.path.exists(yolo_txt_path):
            return obb_points
            
        with open(yolo_txt_path, 'r') as f:
            for line in f:
                if line.strip() and line.startswith('1 '):
                    pts = self.parse_yolo_obb_points(line, img_width, img_height)
                    obb_points.append(pts)
        return obb_points
    
    def parse_yolo_obb_points(self, yolo_line: str, img_width: int, img_height: int) -> list:
        """YOLO formatı hızlı parsing"""
        parts = yolo_line.strip().split()
        coords = np.array(parts[1:], dtype=np.float32).reshape(4, 2)
        coords[:, 0] *= img_width
        coords[:, 1] *= img_height
        return coords.tolist()
    
    def create_comparison_visualization(self, image: np.ndarray, original_mask: np.ndarray,
                                      orig_obb_corners: list, trapezoid_corners: list, obb_corners: list,
                                      shelf_id: int, orig_obb_iou: float, trapezoid_iou: float, obb_iou: float,
                                      save_path: str) -> None:
        """Optimize edilmiş görselleştirme"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # Daha küçük figür
            fig.suptitle(f'Shelf {shelf_id + 1}\nOrig: {orig_obb_iou:.3f} | Trap: {trapezoid_iou:.3f} | OBB: {obb_iou:.3f}',
                        fontsize=12)
            
            # Mask RGBA'yı bir kere hesapla
            mask_rgba = np.zeros((*original_mask.shape, 4))
            mask_rgba[original_mask > 0] = [1, 0.2, 0.2, 0.4]
            
            for ax, corners, color, title, iou in [
                (ax1, orig_obb_corners, 'blue', f'Original ({orig_obb_iou:.3f})', orig_obb_iou),
                (ax2, trapezoid_corners, 'darkgreen', f'Trapezoid ({trapezoid_iou:.3f})', trapezoid_iou),
                (ax3, obb_corners, 'darkred', f'Optimized ({obb_iou:.3f})', obb_iou)
            ]:
                ax.imshow(image)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                if original_mask.sum() > 0:
                    ax.imshow(mask_rgba)
                if len(corners) >= 3:
                    polygon = Polygon(np.array(corners, dtype=np.int32), 
                                    fill=False, edgecolor=color, linewidth=2, alpha=0.9)
                    ax.add_patch(polygon)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')  # Düşük DPI
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def process_dataset(self, input_dir: str, output_dir: str, image_dir: str = None) -> Dict[str, Any]:
        """
        Paralel dataset işleme
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.visualize:
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # JSON dosyalarını bul
        json_files = [
            os.path.join(input_dir, file) 
            for file in os.listdir(input_dir) 
            if file.endswith('_enhanced_trapezoid.json')
        ]
        
        logger.info(f"Found {len(json_files)} trapezoid JSON files to convert")
        logger.info(f"Using {self.n_workers} parallel workers")
        
        # Paralel işleme argümanları
        args_list = [(json_path, output_dir, image_dir) for json_path in json_files]
        
        total_results = {
            "conversion_timestamp": datetime.now().isoformat(),
            "total_files": len(json_files),
            "successfully_converted": 0,
            "failed_conversions": 0,
            "skipped_files": 0,
            "overall_stats": {
                "total_objects_converted": 0,
                "average_trapezoid_iou": 0.0,
                "average_obb_iou": 0.0,
                "average_iou_loss": 0.0
            },
            "converted_files": [],
            "skipped_files_list": []
        }
        
        all_trapezoid_ious = []
        all_obb_ious = []
        
        # ThreadPoolExecutor kullan (I/O bound işlemler için)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(self.process_single_json_parallel, args_list))
        
        # Sonuçları topla
        for i, result in enumerate(results):
            json_path = json_files[i]
            logger.info(f"Processed {i + 1}/{len(json_files)}: {os.path.basename(json_path)}")
            
            if "skipped" in result:
                total_results["skipped_files"] += 1
                total_results["skipped_files_list"].append({
                    "filename": os.path.basename(json_path),
                    "reason": result["skipped"]
                })
                logger.info(f"Skipped: {result['skipped']}")
            elif "error" not in result:
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
        
        # Genel istatistikler
        if all_trapezoid_ious:
            total_results["overall_stats"]["average_trapezoid_iou"] = float(np.mean(all_trapezoid_ious))
            total_results["overall_stats"]["average_obb_iou"] = float(np.mean(all_obb_ious))
            total_results["overall_stats"]["average_iou_loss"] = float(np.mean(all_trapezoid_ious) - np.mean(all_obb_ious))
        
        # Sonuç kaydetme
        summary_path = os.path.join(output_dir, "optimized_trapezoid_to_obb_conversion_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(total_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZED TRAPEZOID TO OBB CONVERSION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total files processed: {total_results['total_files']}")
        logger.info(f"Successfully converted: {total_results['successfully_converted']}")
        logger.info(f"Skipped files (1500x2000): {total_results['skipped_files']}")
        logger.info(f"Failed conversions: {total_results['failed_conversions']}")
        logger.info(f"Total objects converted: {total_results['overall_stats']['total_objects_converted']}")
        logger.info(f"Average trapezoid IoU: {total_results['overall_stats']['average_trapezoid_iou']:.4f}")
        logger.info(f"Average OBB IoU: {total_results['overall_stats']['average_obb_iou']:.4f}")
        logger.info(f"Average IoU loss: {total_results['overall_stats']['average_iou_loss']:.4f}")
        
        return total_results

def main():
    """
    Optimize edilmiş ana fonksiyon
    """
    INPUT_DIR = "new_dataset/trapezoid_dataset"
    OUTPUT_DIR = "new_dataset/optimized_trapezoid_to_obb_converted"
    IMAGE_DIR = "new_dataset/worst_dataset_pure"
    VISUALIZE = False
    N_WORKERS = min(8, mp.cpu_count())  # CPU çekirdeği kadar worker
    
    try:
        logger.info("Initializing Optimized Trapezoid to OBB Converter...")
        converter = OptimizedTrapezoidToOBBConverter(
            visualize=VISUALIZE, 
            n_workers=N_WORKERS
        )
        
        logger.info(f"Starting optimized conversion from: {INPUT_DIR}")
        results = converter.process_dataset(INPUT_DIR, OUTPUT_DIR, IMAGE_DIR)
        
        logger.info("Optimized conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise

if __name__ == "__main__":
    main()