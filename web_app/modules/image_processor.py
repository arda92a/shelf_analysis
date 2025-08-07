import cv2
import numpy as np
import os

class ImageProcessor:
    def __init__(self):
        self.temp_dir = 'static/temp'
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def save_image(self, numpy_array, filename):
        """Numpy array'i dosyaya kaydet"""
        try:
            if numpy_array is None:
                return None
            
            # RGB'den BGR'ye çevir (OpenCV formatından kaydetme formatına)
            bgr_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
            
            filepath = os.path.join(self.temp_dir, filename)
            cv2.imwrite(filepath, bgr_array)
            
            return filepath
            
        except Exception as e:
            print(f"Görsel kaydetme hatası: {e}")
            return None
    
    def resize_image(self, image, max_width, max_height):
        """Görseli yeniden boyutlandır"""
        try:
            height, width = image.shape[:2]
            
            # Aspect ratio'yu koru
            aspect_ratio = width / height
            
            if width > max_width:
                width = max_width
                height = int(width / aspect_ratio)
            
            if height > max_height:
                height = max_height
                width = int(height * aspect_ratio)
            
            return cv2.resize(image, (width, height))
            
        except Exception as e:
            print(f"Görsel boyutlandırma hatası: {e}")
            return image
    
    def cleanup_temp_files(self):
        """Geçici dosyaları temizle"""
        try:
            for filename in os.listdir(self.temp_dir):
                if filename.endswith('.png'):
                    os.remove(os.path.join(self.temp_dir, filename))
        except Exception as e:
            print(f"Temizlik hatası: {e}") 