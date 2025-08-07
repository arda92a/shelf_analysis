import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from glob import glob
from ultralytics import SAM
import torch
from typing import Tuple, Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAM2OBBGenerator:
    def __init__(self, model_path: str , device: str = "auto", visualize: bool = True):
        """
        SAM2 modelini OBB modunda kullanarak maskelerden OBB üreten sınıf
        
        Args:
            model_path: SAM2 model dosyası yolu
            device: Kullanılacak cihaz ("auto", "cpu", "cuda", vs.)
            visualize: Görselleştirme yapılıp yapılmayacağı
        """
        self.device = device
        self.visualize = visualize
        try:
            # SAM2 modelini yükle
            self.model = SAM(model_path)
            self.model = self.model.to(device)
            logger.info(f"SAM2 modeli başarıyla yüklendi: {model_path}")
        except Exception as e:
            logger.error(f"SAM2 model yüklenirken hata: {e}")
            self.model = None
    
    def enhance_sam2_prediction_with_box_prompt(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        SAM2'yi box prompt ile çalıştır ve OBB üret
        """
        if self.model is None:
            return None
            
        try:
            # Maskeden bounding box hesapla
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
                
            # En büyük contour'dan bounding box al
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Box format: [x1, y1, x2, y2]
            box = [x, y, x + w, y + h]
            
            logger.info(f"SAM2'ye gönderilen box: {box}")
            
            results = self.model.predict(
                source=image,
                bboxes=[box],
                save=False,
                verbose=False
            )
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"SAM2 box prediction hatası: {e}")
            return None
    
    def mask_to_obb_sam2(self, mask: np.ndarray, image: np.ndarray = None) -> Tuple[Optional[Tuple], Optional[np.ndarray]]:
        """
        SAM2 modelini kullanarak maskeden direkt OBB üret
        """
        if self.model is None:
            raise ValueError("SAM2 modeli yüklenemedi!")
        
        # Eğer görüntü yoksa maski görüntü olarak kullan
        if image is None:
            image = np.stack([mask, mask, mask], axis=-1).astype(np.uint8) * 255
        
        # Box prompt ile SAM2'yi OBB modunda çalıştır
        logger.info("SAM2 OBB task kullanılıyor...")
        sam2_results = self.enhance_sam2_prediction_with_box_prompt(image, mask)
        
        if sam2_results is None:
            raise ValueError("SAM2 prediction başarısız!")
                
        # SAM2 sonucunu debug et
        logger.info(f"SAM2 sonucu özellikleri: {[attr for attr in dir(sam2_results) if not attr.startswith('_')]}")
        logger.info(f"SAM2 obb mevcut mu: {hasattr(sam2_results, 'obb')}")
        
        if hasattr(sam2_results, 'obb'):
            logger.info(f"SAM2 obb data: {sam2_results.obb}")
            if sam2_results.obb is not None:
                logger.info(f"SAM2 obb data shape: {sam2_results.obb.data.shape if hasattr(sam2_results.obb, 'data') else 'No data attr'}")
        
        # SAM2'den direkt OBB al
        if hasattr(sam2_results, 'obb') and sam2_results.obb is not None:
            obb_data = sam2_results.obb.data
            if len(obb_data) > 0:
                # İlk OBB'yi al
                best_obb = obb_data[0]
                
                # OBB formatını kontrol et
                logger.info(f"OBB data: {best_obb}")
                logger.info(f"OBB data shape: {best_obb.shape}")
                
                if len(best_obb) >= 8:
                    
                    box_points = best_obb[:8].reshape(4, 2).cpu().numpy()
                    
                    # Merkez, boyutlar ve açıyı hesapla
                    center_x = np.mean(box_points[:, 0])
                    center_y = np.mean(box_points[:, 1])
                    
                    # En uzun kenarları bul
                    edge_lengths = []
                    for i in range(4):
                        next_i = (i + 1) % 4
                        length = np.linalg.norm(box_points[next_i] - box_points[i])
                        edge_lengths.append(length)
                    
                    width = max(edge_lengths)
                    height = min(edge_lengths)
                    
                    # Açıyı hesapla
                    max_edge_idx = np.argmax(edge_lengths)
                    edge_vector = box_points[(max_edge_idx + 1) % 4] - box_points[max_edge_idx]
                    angle = np.degrees(np.arctan2(edge_vector[1], edge_vector[0]))
                    
                    logger.info(f"SAM2 direkt OBB ile başarılı: center=({center_x:.1f},{center_y:.1f}), size=({width:.1f},{height:.1f}), angle={angle:.1f}")
                    return (center_x, center_y, width, height, angle), box_points
         
        raise ValueError("SAM2'den OBB alınamadı - OBB data boş veya mevcut değil!")
    
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
        Bir yolo txt dosyasındaki tüm OBB köşe noktalarını döndürür
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
    
    def polygon_to_mask(self, points: List[List[float]], img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Polygon noktalarından binary mask oluştur
        """
        mask = np.zeros(img_shape, dtype=np.uint8)
        pts = np.array(points, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        İki mask arasındaki IoU hesapla
        """
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        if union == 0:
            return 0.0
        return intersection / union
    
    def obb_to_mask(self, box: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        OBB box noktalarından mask oluştur
        """
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [box.astype(np.int32)], 255)
        return mask
    
    def create_comparison_visualization(self, image: np.ndarray, original_mask: np.ndarray,
                                      orig_obb_corners: list, sam2_obb_corners: list,
                                      shelf_id: int, orig_obb_iou: float, sam2_obb_iou: float,
                                      save_path: str, sam2_method: str = "SAM2") -> None:
        """
        Orijinal OBB ve SAM2 OBB karşılaştırmalı görselleştirme oluştur
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'Shelf {shelf_id + 1} ({sam2_method})\nOriginal OBB IoU: {orig_obb_iou:.3f} | {sam2_method} OBB IoU: {sam2_obb_iou:.3f}',
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
            
            # Sağ: SAM2 OBB
            ax2.imshow(image)
            ax2.set_title(f'{sam2_method} OBB (IoU: {sam2_obb_iou:.3f})', fontsize=14, fontweight='bold')
            ax2.axis('off')
            if original_mask.sum() > 0:
                ax2.imshow(mask_rgba)
            if len(sam2_obb_corners) >= 3:
                color = 'green'
                sam2_obb_polygon = Polygon(np.array(sam2_obb_corners, dtype=np.int32), fill=False, edgecolor=color, linewidth=2, linestyle='-', alpha=0.9)
                ax2.add_patch(sam2_obb_polygon)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Shelf {shelf_id + 1} comparison saved: {os.path.basename(save_path)}")
        except Exception as e:
            logger.error(f"Error creating shelf {shelf_id + 1} comparison: {e}")
    
    def save_obb_yolo_format(self, obb_params: Tuple, img_shape: Tuple[int, int], output_path: str):
        """
        OBB'yi YOLO formatında kaydet
        """
        center_x, center_y, width, height, angle = obb_params
        img_h, img_w = img_shape
        
        # Normalize coordinates
        norm_x = center_x / img_w
        norm_y = center_y / img_h
        norm_w = width / img_w
        norm_h = height / img_h
        
        with open(output_path, 'w') as f:
            f.write(f"1 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f} {angle:.6f}\n")
    
    def process_jsons(self, json_dir: str, output_dir: str, image_dir: str = None, 
                     save_yolo_format: bool = True):
        """
        JSON dosyalarını işleyerek SAM2 OBB üret
        """
        os.makedirs(output_dir, exist_ok=True)
        if save_yolo_format:
            yolo_dir = os.path.join(output_dir, "yolo_labels")
            os.makedirs(yolo_dir, exist_ok=True)
        
        # Görselleştirme klasörü
        if self.visualize:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        
        json_paths = glob(os.path.join(json_dir, '*.json'))
        
        logger.info(f"İşlenecek JSON dosya sayısı: {len(json_paths)}")
        
        results_summary = {
            'total_processed': 0,
            'successful_obb': 0,
            'failed_obb': 0,
            'avg_orig_iou': 0.0,
            'avg_sam2_iou': 0.0
        }
        
        orig_ious = []
        sam2_ious = []
        
        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # JSON'dan image_name'i al
                image_name = data.get('metadata', {}).get('name')
                if not image_name:
                    base = os.path.splitext(os.path.basename(json_path))[0]
                    image_name = base + '.jpg'
                
                # Görsel kontrolü
                if image_dir:
                    worst_img_path = os.path.join(image_dir, image_name)
                    if not os.path.exists(worst_img_path):
                        continue
                    else:
                        logger.info(f"Görsel bulundu, işleniyor: {image_name}")
                
                img_h = data['metadata']['height']
                img_w = data['metadata']['width']
                
                image = None
                if image_dir:
                    img_path = os.path.join(image_dir, image_name)
                    if os.path.exists(img_path):
                        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                        logger.info(f"Görsel yüklendi: {img_path} ({img_w}x{img_h})")
                    else:
                        logger.warning(f"Görsel bulunamadı: {img_path}")
                
                # YOLO OBB'lerini oku (köşe noktası olarak)
                yolo_txt_path = os.path.join('yolo_format', image_name + '.txt')
                yolo_obb_points = self.get_yolo_obb_points_for_image(yolo_txt_path, img_w, img_h)
                
                # Görselleştirme klasörü
                if self.visualize:
                    vis_subdir = os.path.join(vis_dir, image_name)
                    os.makedirs(vis_subdir, exist_ok=True)
                
                # Her shelf için SAM2 OBB üret
                shelf_count = 0
                for idx, inst in enumerate(data['instances']):
                    if inst.get('className') != 'shelf':
                        continue
                    
                    shelf_count += 1
                    results_summary['total_processed'] += 1
                    
                    points = inst['points']
                    mask = self.polygon_to_mask(points, (img_h, img_w))
                    
                    # SAM2 ile OBB üret 
                    try:
                        sam2_result = self.mask_to_obb_sam2(mask, image)
                        sam2_obb, sam2_box = sam2_result
                        
                        results_summary['successful_obb'] += 1
                        
                        # SAM2 IoU hesapla
                        sam2_mask = self.obb_to_mask(sam2_box, (img_h, img_w))
                        sam2_iou = self.calculate_mask_iou(mask, sam2_mask)
                        sam2_ious.append(sam2_iou)
                        
                        # Orijinal OBB IoU hesapla
                        orig_obb_iou = 0.0
                        orig_obb_corners = []
                        if idx < len(yolo_obb_points):
                            orig_obb_corners = yolo_obb_points[idx]
                            orig_obb_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                            cv2.fillPoly(orig_obb_mask, [np.array(orig_obb_corners, dtype=np.int32)], 255)
                            orig_obb_iou = self.calculate_mask_iou(mask, orig_obb_mask)
                            orig_ious.append(orig_obb_iou)
                        
                        # Görselleştirme oluştur
                        if self.visualize and image is not None:
                            vis_filename = f"shelf_{idx + 1}_original_vs_sam2_obb.png"
                            vis_path = os.path.join(vis_subdir, vis_filename)
                            logger.info(f"Görselleştirme oluşturuluyor: {vis_path}")
                            self.create_comparison_visualization(
                                image, mask, orig_obb_corners, sam2_box.tolist(),
                                idx, orig_obb_iou, sam2_iou, vis_path, "SAM2"
                            )
                        
                        if save_yolo_format:
                            yolo_path = os.path.join(yolo_dir, f"{image_name}_shelf_{idx}.txt")
                            self.save_obb_yolo_format(sam2_obb, (img_h, img_w), yolo_path)
                        
                        logger.info(f"✓ SAM2 OBB kaydedildi: {image_name}_shelf_{idx} (Original IoU: {orig_obb_iou:.3f}, SAM2 IoU: {sam2_iou:.3f})")
                        
                    except Exception as e:
                        logger.error(f"✗ OBB hatası: {json_path} instance {idx} - {str(e)}")
                        results_summary['failed_obb'] += 1
                        continue
                
                if shelf_count > 0:
                    logger.info(f"✓ {json_path}: {shelf_count} shelf işlendi")
                    
            except Exception as e:
                logger.error(f"Hata {json_path}: {e}")
                continue
        
        # Özet istatistikleri
        if orig_ious:
            results_summary['avg_orig_iou'] = np.mean(orig_ious)
        if sam2_ious:
            results_summary['avg_sam2_iou'] = np.mean(sam2_ious)
        
        logger.info(f"\n=== İŞLEM ÖZETİ ===")
        logger.info(f"Toplam işlenen shelf: {results_summary['total_processed']}")
        logger.info(f"Başarılı SAM2 OBB: {results_summary['successful_obb']}")
        logger.info(f"Başarısız OBB: {results_summary['failed_obb']}")
        logger.info(f"Ortalama Orijinal OBB IoU: {results_summary['avg_orig_iou']:.3f}")
        logger.info(f"Ortalama SAM2 OBB IoU: {results_summary['avg_sam2_iou']:.3f}")
        
        return results_summary

def main():
    """
    Ana fonksiyon - SAM2 direkt OBB Generator'ı çalıştır
    """
    # Konfigürasyon
    MODEL_PATH = "sam2.1_l.pt"  # SAM2 model dosyası
    JSON_DIR = "segmentation_labels"  # Segmentation etiketleri dizini
    OUTPUT_DIR = "pure_sam2_obb_results"  # Çıktı dizini
    IMAGE_DIR = "new_dataset/worst_dataset_1000"  # Orijinal görseller dizini
    DEVICE = "cuda"  
    VISUALIZE = True  # Gelişmiş görselleştirme
    
    logger.info("SAM2 Direct OBB Generator başlatılıyor...")

    # SAM2 OBB Generator'ı oluştur
    generator = SAM2OBBGenerator(MODEL_PATH, device=DEVICE, visualize=VISUALIZE)
    
    # İşlemi başlat
    if not os.path.exists(JSON_DIR):
        logger.error(f"Hata: JSON dizini bulunamadı: {JSON_DIR}")
        return
    
    results = generator.process_jsons(
        json_dir=JSON_DIR,
        output_dir=OUTPUT_DIR,
        image_dir=IMAGE_DIR,
        save_yolo_format=True
    )
    
    logger.info("İşlem tamamlandı!")

if __name__ == "__main__":
    main()