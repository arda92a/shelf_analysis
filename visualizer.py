import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

class VisualizationHelper:
    """
    IoU analizi sonuçlarını görselleştiren yardımcı sınıf.
    """
    
    def __init__(self):
        # Dinamik renk üretimi 
        self.hsv_colors = []
        self._generate_hsv_colors(100)  # 100 farklı renk 
    
    def _generate_hsv_colors(self, num_colors: int):
        """HSV renk uzayında dinamik renk üretir"""
        import colorsys
        
        for i in range(num_colors):
            
            hue = i / num_colors  
            saturation = 0.8  
            value = 0.9  
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # RGBA formatına çevir (alpha=0.5)
            rgba = [rgb[0], rgb[1], rgb[2], 0.5]
            self.hsv_colors.append(rgba)
    
    def get_mask_color(self, instance_id: str, alpha: float = 0.5) -> list:
        """Instance ID'ye göre mask rengi döndürür - her instance için farklı renk"""
        # Instance ID'den hash değeri üret
        hash_value = hash(instance_id) % len(self.hsv_colors)
        color = self.hsv_colors[hash_value].copy()
        color[3] = alpha  
        return color
    
    def get_obb_color(self, instance_id: str) -> str:
        """Instance ID'ye göre OBB rengi döndürür - her instance için farklı renk"""
        # Instance ID'den hash değeri üret
        hash_value = hash(instance_id) % len(self.hsv_colors)
        import matplotlib.colors as mcolors
        rgb = self.hsv_colors[hash_value][:3]  
        return mcolors.rgb2hex(rgb)  
    
    def visualize_comparison(self, 
                           image: np.ndarray, 
                           mask: np.ndarray, 
                           obb_corners: np.ndarray, 
                           iou_score: float,
                           image_id: str, 
                           instance_id: str,
                           class_name: str,
                           save_path: Optional[str] = None) -> None:
        """
        Görüntü, segmentation mask ve OBB'yi karşılaştırmalı olarak görselleştirir.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Orijinal görüntü
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask 
        axes[1].imshow(image)
        colored_mask = np.zeros((*mask.shape, 4))
        
        # Class'a göre renk seç
        if class_name == 'shelf':
            colored_mask[mask == 1] = [1, 0, 0, 0.6]  # Kırmızı
        elif class_name == 'shelf-space':
            colored_mask[mask == 1] = [0, 1, 0, 0.6]  # Yeşil
        else:
            colored_mask[mask == 1] = [0, 0, 1, 0.6]  # Mavi
            
        axes[1].imshow(colored_mask)
        axes[1].set_title(f'Segmentation Mask\nClass: {class_name}')
        axes[1].axis('off')
        
        # OBB ve mask birlikte
        axes[2].imshow(image)
        axes[2].imshow(colored_mask)
        
        # OBB çiz
        obb_polygon = plt.Polygon(obb_corners, fill=False, 
                                 edgecolor='blue', linewidth=2)
        axes[2].add_patch(obb_polygon)
        axes[2].set_title(f'OBB + Mask\nIoU: {iou_score:.3f}')
        axes[2].axis('off')
        
        plt.suptitle(f'Analysis: {image_id}_{instance_id} ({class_name})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_analysis_charts(self, df_results, output_dir) -> None:
        """
        Analiz grafiklerini oluşturur ve kaydeder.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Genel IoU dağılımı
        axes[0].hist(df_results['IoU_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(df_results['IoU_score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_results["IoU_score"].mean():.3f}')
        axes[0].set_xlabel('IoU Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Overall IoU Score Distribution')
        axes[0].legend()
        
        # Class bazında IoU karşılaştırması
        class_names = df_results['class_name'].unique()
        for i, class_name in enumerate(class_names):
            class_data = df_results[df_results['class_name'] == class_name]
            axes[1].hist(class_data['IoU_score'], bins=20, alpha=0.6, 
                        label=f'{class_name} (n={len(class_data)})')
        axes[1].set_xlabel('IoU Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('IoU Distribution by Class')
        axes[1].legend()
        
        plt.tight_layout()
        save_path = output_dir / "analysis_charts.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Analiz grafikleri kaydedildi: {save_path}")
        plt.show()
    
    
    def visualize_image_summary(self, 
                              image: np.ndarray, 
                              summary_data: List[Dict],
                              image_id: str,
                              mean_iou: float,
                              save_path: Optional[str] = None) -> None:
        """
        Bir görselin tüm instance'larını 3 adet görselleştirmede gösterir:
        1. Orijinal görüntü
        2. Tüm maskler overlay
        3. Tüm OBB + maskler overlay (her instance'ın IoU skoru ile)
        """
        import matplotlib.patches as patches
        
        # 3 sütunlu subplot oluştur
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Orijinal görüntü
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 2. Tüm maskler overlay
        axes[1].imshow(image)
        
        # Her mask'i ayrı ayrı çiz (üst üste binmeyi önlemek için)
        for idx, data in enumerate(summary_data):
            # Gerçek mask verisini kullan
            mask = data['mask']
            colored_mask = np.zeros((*mask.shape, 4))
            
            # Instance ID oluştur (class_name + idx)
            instance_id = f"{data['class_name']}_{idx}"
            
            # Dinamik renk sistemi kullanarak
            color_rgba = self.get_mask_color(instance_id, alpha=0.5)
            colored_mask[mask == 1] = color_rgba
            axes[1].imshow(colored_mask)
        
        axes[1].set_title('All Segmentation Masks')
        axes[1].axis('off')
        
        # 3. Tüm OBB + maskler overlay
        axes[2].imshow(image)
        
        # Tüm maskleri overlay et
        for idx, data in enumerate(summary_data):
            # Gerçek mask verisini kullan
            mask = data['mask']
            colored_mask = np.zeros((*mask.shape, 4))
            
            # Instance ID oluştur (class_name + idx)
            instance_id = f"{data['class_name']}_{idx}"
            
            # Mask rengi (daha açık ton) 
            mask_rgba = self.get_mask_color(instance_id, alpha=0.4)
            colored_mask[mask == 1] = mask_rgba
            axes[2].imshow(colored_mask)
            
            # Gerçek OBB çiz (koyu ton)
            obb_corners = data['obb_corners']
            obb_color = self.get_obb_color(instance_id)
            
            obb_polygon = patches.Polygon(obb_corners, fill=False, edgecolor=obb_color, linewidth=3)
            axes[2].add_patch(obb_polygon)
            
            # Individual IoU skorunu yazdır
            center = np.mean(obb_corners, axis=0)
            text_color = self.get_obb_color(instance_id)
            axes[2].text(center[0], center[1], f'IoU: {data["iou_score"]:.3f}', 
                        color=text_color, fontsize=5, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        axes[2].set_title(f'OBB + Masks | Mean IoU: {mean_iou:.3f}')
        axes[2].axis('off')
        
        plt.suptitle(f'Image Summary: {image_id} | Instances: {len(summary_data)}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 