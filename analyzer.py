import numpy as np
import cv2
import pandas as pd
import json
from shapely.geometry import Polygon
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from iou_calculator import IoUCalculator
from visualizer import VisualizationHelper

class ShelfDatasetAnalyzer:
    """
    YOLO format OBB ve segmentation polygon'ları arasındaki IoU skorunu hesaplayan
    ve görselleştiren ana sınıf.
    """
    
    def __init__(self, 
                 yolo_format_dir: str,
                 segmentation_labels_dir: str,
                 output_dir: str = "shelf_analysis_results"):
        """
        Args:
            yolo_format_dir: YOLO format txt dosyalarının bulunduğu dizin
            segmentation_labels_dir: Segmentation etiketlerinin bulunduğu dizin
            output_dir: Sonuçların kaydedileceği dizin
        """
        self.yolo_dir = Path(yolo_format_dir)
        self.seg_dir = Path(segmentation_labels_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Alt dizinler oluştur
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "low_iou_cases").mkdir(exist_ok=True)
        (self.output_dir / "analysis_reports").mkdir(exist_ok=True)
        
        self.results = []
        
        # Yardımcı sınıfları başlat
        self.iou_calculator = IoUCalculator()
        self.visualizer = VisualizationHelper()
    
    def process_single_image(self, image_name: str) -> List[Dict]:
        """
        Tek bir görüntü için tüm instance'ları işler.
        
        Args:
            image_name: Görüntü dosya adı (örn: "00001.jpg")
            
        Returns:
            Bu görüntüdeki tüm sonuçların listesi
        """
        image_results = []
        
        # Dosya yollarını oluştur
        base_name = image_name.replace('.jpg', '')
        yolo_file = self.yolo_dir / f"{image_name}.txt"  
        seg_json_file = self.seg_dir / f"{image_name}.json"
        image_file = self.seg_dir / image_name
        
        # Dosyaların varlığını kontrol et
        if not all([yolo_file.exists(), seg_json_file.exists(), image_file.exists()]):
            print(f"Eksik dosyalar: {image_name}")
            return image_results
        
        try:
            # Görüntüyü yükle
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Görüntü yüklenemedi: {image_file}")
                return image_results
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image.shape[:2]
            
            # YOLO OBB verilerini yükle
            obb_data = []
            with open(yolo_file, 'r') as f:
                for line in f:
                    if line.strip():
                        obb_data.append(self.iou_calculator.parse_yolo_obb(line, img_width, img_height))
            
            # Segmentation JSON verilerini yükle
            seg_data = self.iou_calculator.parse_segmentation_json(seg_json_file)
            
            # Instance'ları eşleştir
            matches = self.iou_calculator.find_matching_instances(obb_data, seg_data)
            
            # Her eşleşme için IoU hesapla
            for idx, (obb, seg) in enumerate(matches):
                # Segmentation mask oluştur
                mask = self.iou_calculator.polygon_to_mask(seg['points'], img_width, img_height)
                
                # IoU hesapla
                iou_score = self.iou_calculator.calculate_iou(mask, obb['corners_pixel'])
                
                # Sonucu kaydet
                result = {
                    'image_id': base_name,
                    'instance_id': f"{idx:02d}",
                    'class_name': obb['class_name'],
                    'IoU_score': iou_score,
                    'obb_area': Polygon(obb['corners_pixel']).area,
                    'mask_area': np.sum(mask)
                }
                
                image_results.append(result)
                self.results.append(result)
                
                # Individual görselleştirmeleri kaydet
                save_path = (self.output_dir / "visualizations" / 
                           f"{base_name}_{idx:02d}_{obb['class_name']}_iou{iou_score:.3f}.png")
                self.visualizer.visualize_comparison(image, mask, obb['corners_pixel'], 
                                        iou_score, base_name, f"{idx:02d}",
                                        obb['class_name'], str(save_path))
                
                # Düşük IoU skorları için özel kayıt (low_iou_cases klasörüne)
                if iou_score < 0.5:
                    save_path = (self.output_dir / "low_iou_cases" / 
                               f"{base_name}_{idx:02d}_{obb['class_name']}_iou{iou_score:.3f}.png")
                    self.visualizer.visualize_comparison(image, mask, obb['corners_pixel'], 
                                            iou_score, base_name, f"{idx:02d}",
                                            obb['class_name'], str(save_path))
            
            # Her görsel için tek bir özet görselleştirme oluştur
            if image_results:
                mean_iou = np.mean([r['IoU_score'] for r in image_results])
                summary_path = (self.output_dir / "visualizations" / 
                              f"{base_name}_summary_mean_iou{mean_iou:.3f}.png")
                
                # Gerçek OBB ve mask verilerini hazırla
                summary_data = []
                for idx, (obb, seg) in enumerate(matches):
                    mask = self.iou_calculator.polygon_to_mask(seg['points'], img_width, img_height)
                    summary_data.append({
                        'obb_corners': obb['corners_pixel'],
                        'mask': mask,
                        'iou_score': image_results[idx]['IoU_score'],
                        'class_name': obb['class_name']
                    })
                
                self.visualizer.visualize_image_summary(image, summary_data, base_name, mean_iou, str(summary_path))
            
        except Exception as e:
            print(f"Hata işlenirken {image_name}: {e}")
        
        return image_results
    
    def process_dataset(self, max_images: Optional[int] = None) -> pd.DataFrame:
        """
        Tüm veri kümesini işler.
        
        Args:
            max_images: İşlenecek maksimum görüntü sayısı (None = hepsini işle)
            
        Returns:
            Sonuçları içeren pandas DataFrame
        """
        print("Dataset taranıyor...")
        
        # Segmentation dizinindeki JPG dosyalarını bul
        image_files = list(self.seg_dir.glob("*.jpg"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Toplam {len(image_files)} görüntü bulundu")
        
        for image_file in tqdm(image_files, desc="Görüntüler işleniyor"):
            self.process_single_image(image_file.name)
        
        # Sonuçları JSON formatında kaydet
        if len(self.results) > 0:
            # Her görsel için mean IoU hesapla
            image_summaries = {}
            for result in self.results:
                image_id = result['image_id']
                if image_id not in image_summaries:
                    image_summaries[image_id] = {
                        'image_id': image_id,
                        'instances': [],
                        'mean_iou': 0.0,
                        'instance_count': 0
                    }
                
                image_summaries[image_id]['instances'].append({
                    'instance_id': result['instance_id'],
                    'class_name': result['class_name'],
                    'iou_score': float(result['IoU_score']),
                    'obb_area': float(result['obb_area']),
                    'mask_area': int(result['mask_area'])
                })
            
            # Mean IoU'ları hesapla
            for image_id, summary in image_summaries.items():
                iou_scores = [inst['iou_score'] for inst in summary['instances']]
                summary['mean_iou'] = np.mean(iou_scores)
                summary['instance_count'] = len(summary['instances'])
            
            # JSON formatında kaydet
            json_data = {
                'analysis_summary': {
                    'total_images': len(image_summaries),
                    'total_instances': len(self.results),
                    'overall_mean_iou': float(np.mean([r['IoU_score'] for r in self.results])),
                    'overall_median_iou': float(np.median([r['IoU_score'] for r in self.results]))
                },
                'image_results': list(image_summaries.values())
            }
            
            json_path = self.output_dir / "iou_scores.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"Sonuçlar kaydedildi: {json_path}")
            
            df_results = pd.DataFrame(self.results)
        else:
            print("Hiç sonuç bulunamadı!")
            df_results = pd.DataFrame()
        
        return df_results
    
    def analyze_results(self, df_results: pd.DataFrame) -> None:
        """
        Sonuçları analiz eder ve istatistikleri yazdırır.
        """
        if len(df_results) == 0:
            print("Analiz edilecek sonuç bulunamadı!")
            return
        
        print("\n=== GENEL PERFORMANS ANALİZİ ===")
        print(f"Toplam örnek sayısı: {len(df_results)}")
        print(f"Ortalama IoU: {df_results['IoU_score'].mean():.3f}")
        print(f"Medyan IoU: {df_results['IoU_score'].median():.3f}")
        print(f"Standart sapma: {df_results['IoU_score'].std():.3f}")
        print(f"Min IoU: {df_results['IoU_score'].min():.3f}")
        print(f"Max IoU: {df_results['IoU_score'].max():.3f}")
        
        # Class bazında analiz
        print("\n=== CLASS BAZINDA ANALİZ ===")
        for class_name in df_results['class_name'].unique():
            class_data = df_results[df_results['class_name'] == class_name]
            print(f"\n{class_name.upper()}:")
            print(f"  Örnek sayısı: {len(class_data)}")
            print(f"  Ortalama IoU: {class_data['IoU_score'].mean():.3f}")
            print(f"  Medyan IoU: {class_data['IoU_score'].median():.3f}")
            print(f"  Min IoU: {class_data['IoU_score'].min():.3f}")
            print(f"  Max IoU: {class_data['IoU_score'].max():.3f}")
        
        # Düşük performans gösteren örnekler
        low_iou_threshold = 0.5
        low_iou_count = len(df_results[df_results['IoU_score'] < low_iou_threshold])
        print(f"\n=== DÜŞÜK PERFORMANS ANALİZİ ===")
        print(f"Düşük IoU skorlu örnekler (< {low_iou_threshold}): {low_iou_count}")
        print(f"Düşük performans oranı: {low_iou_count/len(df_results)*100:.1f}%")
        
        # En düşük skorlu örnekleri listele
        print(f"\nEn düşük 10 IoU skoru:")
        worst_cases = df_results.nsmallest(10, 'IoU_score')
        for _, row in worst_cases.iterrows():
            print(f"  {row['image_id']}_{row['instance_id']} ({row['class_name']}): {row['IoU_score']:.3f}")
        
        self.visualizer.plot_analysis_charts(df_results, self.output_dir)
        self.save_detailed_report(df_results)
    
    def save_detailed_report(self, df_results: pd.DataFrame) -> None:
        """
        Detaylı analiz raporunu kaydeder.
        """
        report_path = self.output_dir / "analysis_reports" / "detailed_report.txt"
        (self.output_dir / "analysis_reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SHELF DATASET OBB-SEGMENTATION IoU ANALİZ RAPORU\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Toplam örnek sayısı: {len(df_results)}\n")
            f.write(f"Ortalama IoU: {df_results['IoU_score'].mean():.4f}\n")
            f.write(f"Medyan IoU: {df_results['IoU_score'].median():.4f}\n")
            f.write(f"Standart sapma: {df_results['IoU_score'].std():.4f}\n\n")
            
            # Class bazında istatistikler
            f.write("CLASS BAZINDA İSTATİSTİKLER:\n")
            f.write("-" * 30 + "\n")
            for class_name in df_results['class_name'].unique():
                class_data = df_results[df_results['class_name'] == class_name]
                f.write(f"\n{class_name.upper()}:\n")
                f.write(f"  Örnek sayısı: {len(class_data)}\n")
                f.write(f"  Ortalama IoU: {class_data['IoU_score'].mean():.4f}\n")
                f.write(f"  Medyan IoU: {class_data['IoU_score'].median():.4f}\n")
                f.write(f"  Std: {class_data['IoU_score'].std():.4f}\n")
                f.write(f"  Min IoU: {class_data['IoU_score'].min():.4f}\n")
                f.write(f"  Max IoU: {class_data['IoU_score'].max():.4f}\n")
            
            # En düşük performanslı örnekler
            f.write(f"\n\nEN DÜŞÜK 20 IoU SKORU:\n")
            f.write("-" * 30 + "\n")
            worst_cases = df_results.nsmallest(20, 'IoU_score')
            for _, row in worst_cases.iterrows():
                f.write(f"{row['image_id']}_{row['instance_id']} ({row['class_name']}): {row['IoU_score']:.4f}\n")
        
        print(f"Detaylı rapor kaydedildi: {report_path}") 