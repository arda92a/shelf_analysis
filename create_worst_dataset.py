"""
En Kötü 1000 Resim İçin Dataset Oluşturma Scripti

Bu script, iou_scores.json dosyasından en kötü 1000 resmi bulur ve
yeni bir klasörde dataset oluşturur.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class WorstDatasetCreator:
    """
    En kötü 1000 resim için dataset oluşturan sınıf
    """
    
    def __init__(self, 
                 json_path: str = "shelf_analysis_results/iou_scores.json",
                 source_images_dir: str = "segmentation_labels",
                 source_yolo_dir: str = "yolo_format",
                 output_dir: str = "worst_1000_dataset"):
        """
        Args:
            json_path: IoU skorları JSON dosyasının yolu
            source_images_dir: Orijinal görüntülerin bulunduğu dizin
            source_yolo_dir: YOLO format txt dosyalarının bulunduğu dizin
            output_dir: Yeni dataset'in oluşturulacağı dizin
        """
        self.json_path = Path(json_path)
        self.source_images_dir = Path(source_images_dir)
        self.source_yolo_dir = Path(source_yolo_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True)
        
        self.load_data()
        
    def load_data(self):
        """JSON verilerini yükler ve mean IoU'ya göre sıralar"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON dosyası bulunamadı: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Mean IoU'ya göre sırala (en kötüden en iyiye)
        self.sorted_data = sorted(self.data, key=lambda x: x['mean_iou'])
        
        print(f"Toplam {len(self.data)} görüntü yüklendi")
        print(f"En kötü mean IoU: {self.sorted_data[0]['mean_iou']:.4f}")
        print(f"En iyi mean IoU: {self.sorted_data[-1]['mean_iou']:.4f}")
    
    def create_annotation_json(self, image_data: Dict, image_name: str) -> Dict:
        """
        Tek bir görüntü için annotation JSON dosyası oluşturur
        
        Args:
            image_data: JSON'dan gelen görüntü verisi
            image_name: Görüntü dosya adı (örn: "00001.jpg")
            
        Returns:
            Annotation JSON verisi
        """
        # Dosya adını çıkar
        base_name = image_name.replace('.jpg', '')
        
        # YOLO ve segmentation dosya yollarını oluştur
        yolo_file = self.source_yolo_dir / f"{base_name}.jpg.txt"
        seg_file = self.source_images_dir / f"{base_name}.jpg.json"
        
        # YOLO verilerini yükle
        yolo_labels = []
        if yolo_file.exists():
            with open(yolo_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        yolo_labels.append({
                            'class_id': class_id,
                            'class_name': 'shelf' if class_id == 1 else 'shelf-space',
                            'coordinates': coords
                        })
        
        # Segmentation verilerini yükle
        seg_labels = []
        if seg_file.exists():
            with open(seg_file, 'r', encoding='utf-8') as f:
                seg_data = json.load(f)
                if 'instances' in seg_data:
                    seg_labels = seg_data['instances']
        
        # Annotation JSON oluştur
        annotation = {
            'image_name': image_name,
            'mean_iou': image_data['mean_iou'],
            'single_iou': image_data['single_iou'],
            'yolo_labels': yolo_labels,
            'segmentation_labels': seg_labels,
            'metadata': {
                'source_json': str(self.json_path),
                'total_instances': len(image_data['single_iou']),
                'rank': self.sorted_data.index(image_data) + 1
            }
        }
        
        return annotation
    
    def copy_image_file(self, image_name: str) -> bool:
        """
        Görüntü dosyasını yeni klasöre kopyalar
        
        Args:
            image_name: Görüntü dosya adı
            
        Returns:
            Başarı durumu
        """
        source_path = self.source_images_dir / image_name
        target_path = self.output_dir / image_name
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            return True
        else:
            print(f"Görüntü bulunamadı: {source_path}")
            return False
    
    def create_dataset(self, num_images: int = 1000):
        """
        En kötü N görüntü için dataset oluşturur
        
        Args:
            num_images: Oluşturulacak dataset boyutu
        """
        print(f"\nEn kötü {num_images} görüntü için dataset oluşturuluyor...")
        print(f"Hedef klasör: {self.output_dir}")
        
        success_count = 0
        total_count = min(num_images, len(self.sorted_data))
        
        for i in range(total_count):
            image_data = self.sorted_data[i]
            image_path = Path(image_data['image_path'])
            image_name = image_path.name
            
            print(f"İşleniyor: {i+1}/{total_count} - {image_name} (Mean IoU: {image_data['mean_iou']:.4f})")
            
            # Görüntü dosyasını kopyala
            if self.copy_image_file(image_name):
                # Annotation JSON oluştur
                annotation = self.create_annotation_json(image_data, image_name)
                
                # JSON dosyasını kaydet
                json_name = image_name.replace('.jpg', '.json')
                json_path = self.output_dir / json_name
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                print(f"  ✓ {image_name} ve {json_name} oluşturuldu")
            else:
                print(f"  ✗ {image_name} kopyalanamadı")
        
        print(f"\nTamamlandı! {success_count}/{total_count} görüntü başarıyla işlendi.")
        print(f"Dataset '{self.output_dir}' klasöründe oluşturuldu.")    
    

def main():
    """Ana fonksiyon"""
    print("=" * 60)
    print("EN KÖTÜ 1000 GÖRÜNTÜ DATASET OLUŞTURMA SCRIPTİ")
    print("=" * 60)
    
    try:
        # Dataset creator'ı başlat
        creator = WorstDatasetCreator()
        
        # En kötü 1000 görüntü için dataset oluştur
        creator.create_dataset(num_images=1000)
        
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    main() 