"""
Shelf Dataset IoU Analizi - Ana Çalıştırma Dosyası

Bu dosya, YOLO format OBB ve segmentation polygon'ları arasındaki IoU skorunu
hesaplayan ve görselleştiren ana programı çalıştırır.

Kullanım:
    python main.py
"""

from analyzer import ShelfDatasetAnalyzer

def main():
    """
    Ana analiz fonksiyonu
    """
    print("=" * 60)
    print("SHELF DATASET OBB-SEGMENTATION IoU ANALİZİ")
    print("=" * 60)
    print()
    
    # Analyzer'ı başlat
    analyzer = ShelfDatasetAnalyzer(
        yolo_format_dir="../yolo_format",
        segmentation_labels_dir="../segmentation_labels"
    )
    
    print("Shelf Dataset analizi başlatılıyor...")
    print()
    
    # Veri kümesini işle (ilk 2 görüntü ile test, max images girilmezse tüm görüntüler işlenir)
    results_df = analyzer.process_dataset(max_images=2)
    
    # Sonuçları analiz et
    if len(results_df) > 0:
        analyzer.analyze_results(results_df)
        print(f"\nTüm sonuçlar '{analyzer.output_dir}' dizininde kaydedildi.")
        print("- iou_scores.json: IoU skorları")
        print("- visualizations/: Tüm görselleştirmeler") 
        print("- low_iou_cases/: Düşük IoU skorlu örnekler")
        print("- analysis_charts.png: Analiz grafikleri")
        print("- analysis_reports/: Detaylı raporlar")
    else:
        print("Analiz edilecek veri bulunamadı!")
    
    print("\n" + "=" * 60)
    print("ANALİZ TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main() 