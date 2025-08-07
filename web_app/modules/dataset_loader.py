import os
import json
import cv2
import numpy as np
from pathlib import Path

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def load_all_images(self):
        """Tüm görselleri ve JSON dosyalarını yükle"""
        images_data = []
        
        # JSON dosyalarını bul
        json_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(self.dataset_path, json_file)
            image_name = json_file.replace('.json', '.jpg')
            image_path = os.path.join(self.dataset_path, image_name)
            
            # Görsel dosyası var mı kontrol et
            if not os.path.exists(image_path):
                print(f"Görsel bulunamadı: {image_path}")
                continue
            
            try:
                # JSON dosyasını yükle
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                # Görseli yükle
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Görsel yüklenemedi: {image_path}")
                    continue
                
                # RGB'ye çevir
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Veri yapısını oluştur
                image_data = {
                    'image_name': image_name,
                    'image_path': image_path,
                    'json_path': json_path,
                    'image': image_rgb,
                    'json_data': json_data,
                    'instances': json_data.get('instances', [])
                }
                
                images_data.append(image_data)
                
            except Exception as e:
                print(f"Hata: {json_file} - {e}")
                continue
        
        print(f"Toplam {len(images_data)} görsel yüklendi")
        return images_data 