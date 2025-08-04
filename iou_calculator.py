import numpy as np
import cv2
from shapely.geometry import Polygon
from typing import List, Dict, Optional
import json

class IoUCalculator:
    """
    YOLO format OBB ve segmentation polygon'ları arasındaki IoU skorunu hesaplayan
    temel sınıf.
    """
    
    def __init__(self):
        self.class_mapping = {
            0: "shelf-space",
            1: "shelf"
        }
    
    def parse_yolo_obb(self, yolo_line: str, img_width: int, img_height: int) -> Dict:
        """
        YOLO format OBB satırını parse eder.
        
        Args:
            yolo_line: YOLO format satır (class_id x1 y1 x2 y2 x3 y3 x4 y4)
            img_width: Görüntü genişliği
            img_height: Görüntü yüksekliği
            
        Returns:
            Dict: class_id, normalized ve pixel koordinatları içeren dict
        """
        parts = yolo_line.strip().split()
        class_id = int(parts[0])
        
        norm_coords = [float(x) for x in parts[1:]]
        
        corners_norm = np.array(norm_coords).reshape(4, 2)
        
        # Pixel koordinatlarına çevir
        corners_pixel = corners_norm.copy()
        corners_pixel[:, 0] *= img_width
        corners_pixel[:, 1] *= img_height
        
        return {
            'class_id': class_id,
            'class_name': self.class_mapping.get(class_id, f'class_{class_id}'),
            'corners_norm': corners_norm,
            'corners_pixel': corners_pixel
        }
    
    def parse_segmentation_json(self, json_path: str) -> Dict:
        """
        Segmentation JSON dosyasını parse eder.
        
        Args:
            json_path: JSON dosya yolu
            
        Returns:
            Dict: Metadata ve instances bilgileri
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def polygon_to_mask(self, points: List[int], img_width: int, img_height: int) -> np.ndarray:
        """
        Polygon koordinatlarını binary mask'e dönüştürür.
        
        Args:
            points: [x1, y1, x2, y2, ...] format koordinatlar
            img_width: Görüntü genişliği
            img_height: Görüntü yüksekliği
            
        Returns:
            Binary mask (numpy array)
        """
        # Koordinatları (x,y) çiftleri halinde düzenle
        polygon_points = np.array(points).reshape(-1, 2)
        
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points.astype(np.int32)], 1)
        
        return mask
    
    def mask_to_polygon(self, mask: np.ndarray) -> Optional[Polygon]:
        """
        Binary mask'i Shapely Polygon'a dönüştürür.
        """
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # En büyük konturu seç
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 10:
            return None
        
        # Kontur noktalarını Shapely formatına çevir
        points = largest_contour.reshape(-1, 2)
        
        try:
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            return polygon if polygon.is_valid else None
        except Exception as e:
            print(f"Polygon oluşturma hatası: {e}")
            return None
    
    def obb_to_polygon(self, obb_corners: np.ndarray) -> Polygon:
        """
        OBB köşe koordinatlarını Shapely Polygon'a dönüştürür.
        """
        return Polygon(obb_corners)
    
    def calculate_iou(self, mask: np.ndarray, obb_corners: np.ndarray) -> float:
        """
        Segmentation mask ile OBB arasındaki IoU skorunu hesaplar.
        """
        mask_polygon = self.mask_to_polygon(mask)
        obb_polygon = self.obb_to_polygon(obb_corners)
        
        if mask_polygon is None:
            return 0.0
        
        try:
            intersection_area = mask_polygon.intersection(obb_polygon).area
            union_area = mask_polygon.union(obb_polygon).area
            
            if union_area == 0:
                return 0.0
            
            iou = intersection_area / union_area
            return max(0.0, min(1.0, iou))
            
        except Exception as e:
            print(f"IoU hesaplama hatası: {e}")
            return 0.0
    
    def find_matching_instances(self, obb_data: List[Dict], seg_data: Dict) -> List[tuple]:
        """
        OBB ve segmentation instance'larını eşleştirir.
        Aynı class'a sahip olanları merkez noktalarına göre eşleştirir.
        
        Args:
            obb_data: YOLO OBB verileri
            seg_data: Segmentation JSON verileri
            
        Returns:
            List[Tuple]: (obb_instance, seg_instance) eşleşmeleri
        """
        matches = []
        
        # Her OBB için en yakın segmentation instance'ını bul
        for obb in obb_data:
            obb_center = np.mean(obb['corners_pixel'], axis=0)
            obb_class = obb['class_name']
            
            best_match = None
            min_distance = float('inf')
            
            for seg in seg_data['instances']:
                if seg['className'] == obb_class:
                    # Segmentation polygon'unun merkez noktasını hesapla
                    seg_points = np.array(seg['points']).reshape(-1, 2)
                    seg_center = np.mean(seg_points, axis=0)
                    
                    # Mesafeyi hesapla
                    distance = np.linalg.norm(obb_center - seg_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = seg
            
            if best_match is not None:
                matches.append((obb, best_match))
        
        return matches 