"""
Shelf Dataset IoU Analizi Paketi

Bu paket, YOLO format OBB ve segmentation polygon'ları arasındaki IoU skorunu
hesaplayan ve görselleştiren araçları içerir.

Modüller:
- iou_calculator: Temel IoU hesaplama fonksiyonları
- visualizer: Görselleştirme yardımcıları
- analyzer: Ana analiz sınıfı
- main: Çalıştırma dosyası
"""

from .iou_calculator import IoUCalculator
from .visualizer import VisualizationHelper
from .analyzer import ShelfDatasetAnalyzer

__version__ = "1.0.0"
__author__ = "Shelf Analysis Team"

__all__ = [
    'IoUCalculator',
    'VisualizationHelper', 
    'ShelfDatasetAnalyzer'
] 