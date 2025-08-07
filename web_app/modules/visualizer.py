import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {
            'segmentation': (0, 255, 0),  # Yeşil - Ground truth
            'new_obb': (0, 0, 255),       # Mavi - Yeni OBB
            'text': (255, 255, 255)       # Beyaz
        }
    
    def create_segmentation_visualization(self, image_data):
        """Sol panel: Orijinal görsel + Ground truth segmentation mask"""
        try:
            image = image_data['image'].copy()
            instances = image_data['instances']
            
            # Her instance için ground truth segmentation mask çiz
            for i, instance in enumerate(instances):
                # Ground truth segmentation mask
                if 'segmentation_polygon' in instance:
                    polygon_data = instance['segmentation_polygon']
                    coordinates = polygon_data.get('coordinates', [])
                    
                    if coordinates:
                        # Ground truth poligonunu çiz
                        polygon_points = np.array(coordinates, dtype=np.int32)
                        cv2.polylines(image, [polygon_points], True, self.colors['segmentation'], 2)
                        
                        # Poligon içini yarı şeffaf doldur
                        overlay = image.copy()
                        cv2.fillPoly(overlay, [polygon_points], self.colors['segmentation'])
                        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
                        
                        # Shelf ID'sini yaz
                        text = f"Shelf {i+1}"
                        
                        # Merkez noktasını hesapla
                        center_x = int(np.mean([p[0] for p in coordinates]))
                        center_y = int(np.mean([p[1] for p in coordinates]))
                        
                        # Metin arka planı için
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
                        text_x = center_x + 15
                        text_y = center_y - 15
                        
                        # Siyah arka plan
                        cv2.rectangle(image, 
                                    (text_x - 5, text_y - text_size[1] - 5),
                                    (text_x + text_size[0] + 5, text_y + 5),
                                    (0, 0, 0), -1)
                        
                        # Beyaz metin
                        cv2.putText(image, text, 
                                  (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                                  self.colors['text'], 3)
            
            return image
            
        except Exception as e:
            print(f"Segmentation görselleştirme hatası: {e}")
            return None
    
    def create_new_obb_visualization(self, image_data):
        """Sağ panel: Orijinal görsel + Ground truth + Yeni OBB'ler"""
        try:
            image = image_data['image'].copy()
            instances = image_data['instances']
            
            # Her instance için ground truth ve yeni OBB'yi çiz
            for i, instance in enumerate(instances):
                # Ground truth segmentation mask
                if 'segmentation_polygon' in instance:
                    polygon_data = instance['segmentation_polygon']
                    coordinates = polygon_data.get('coordinates', [])
                    
                    if coordinates:
                        # Ground truth poligonunu çiz
                        polygon_points = np.array(coordinates, dtype=np.int32)
                        cv2.polylines(image, [polygon_points], True, self.colors['segmentation'], 2)
                        
                        # Poligon içini yarı şeffaf doldur
                        overlay = image.copy()
                        cv2.fillPoly(overlay, [polygon_points], self.colors['segmentation'])
                        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
                
                # Yeni OBB
                if 'new_obb' in instance:
                    obb_data = instance['new_obb']
                    coordinates = obb_data.get('coordinates', [])
                    
                    if coordinates:
                        # Yeni OBB poligonunu çiz
                        obb_points = np.array(coordinates, dtype=np.int32)
                        cv2.polylines(image, [obb_points], True, self.colors['new_obb'], 3)
                        
                        # Merkez noktasını çiz
                        center = obb_data.get('center', [0, 0])
                        center_point = (int(center[0]), int(center[1]))
                        cv2.circle(image, center_point, 5, self.colors['new_obb'], -1)
                        
                        # Shelf ID'sini ve IoU'yu yaz - daha büyük ve görünür
                        iou = obb_data.get('iou_with_gt', 0)
                        text = f"Shelf {i+1} (IoU: {iou:.3f})"
                        
                        # Metin arka planı için
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
                        text_x = center_point[0] + 15
                        text_y = center_point[1] - 15
                        
                        # Siyah arka plan
                        cv2.rectangle(image, 
                                    (text_x - 5, text_y - text_size[1] - 5),
                                    (text_x + text_size[0] + 5, text_y + 5),
                                    (0, 0, 0), -1)
                        
                        # Beyaz metin
                        cv2.putText(image, text, 
                                  (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                                  self.colors['text'], 3)
            
            return image
            
        except Exception as e:
            print(f"Yeni OBB görselleştirme hatası: {e}")
            return None 