from flask import Flask, request, jsonify, send_file, render_template
import os
import tempfile
import json
import cv2
import numpy as np
from numba import jit
import logging
from pathlib import Path
import sys
import base64
import zipfile
import uuid
import shutil
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Mevcut scriptleri import et
from mask_to_trapezoid_fit import OBBGenerator
from trapezoid_to_obb_optimized import OptimizedTrapezoidToOBBConverter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = None

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global değişkenler
current_data = None
batch_jobs = {}

# Mevcut scriptlerden sınıfları oluştur
obb_generator = OBBGenerator(visualize=False)
trapezoid_converter = OptimizedTrapezoidToOBBConverter(visualize=False)

@jit(nopython=True)
def fast_polygon_area(points):
    """Numba ile hızlandırılmış poligon alanı hesaplama"""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def create_mask_from_polygon(polygon_points, img_height, img_width):
    """Poligon koordinatlarından mask oluştur"""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    polygon_points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_points], 255)
    return mask

def parse_yolo_obb(yolo_line, img_width, img_height):
    """YOLO formatındaki OBB'yi parse et - sadece label=1 olanları al"""
    try:
        parts = yolo_line.strip().split()
        logger.info(f"YOLO satırı parse ediliyor: {yolo_line} - Değer sayısı: {len(parts)}")
        
        if len(parts) >= 9:  # En az 9 değer olmalı (class_id + 8 koordinat)
            class_id = int(parts[0])
            logger.info(f"Class ID: {class_id}")
            
            if class_id == 1:  # Sadece shelf'leri al
                # YOLO formatı: class_id x1 y1 x2 y2 x3 y3 x4 y4
                x1 = float(parts[1]) * img_width
                y1 = float(parts[2]) * img_height
                x2 = float(parts[3]) * img_width
                y2 = float(parts[4]) * img_height
                x3 = float(parts[5]) * img_width
                y3 = float(parts[6]) * img_height
                x4 = float(parts[7]) * img_width
                y4 = float(parts[8]) * img_height
                
                logger.info(f"Koordinatlar: ({x1:.1f}, {y1:.1f}), ({x2:.1f}, {y2:.1f}), ({x3:.1f}, {y3:.1f}), ({x4:.1f}, {y4:.1f})")
                
                return {
                    'class_id': class_id,
                    'coordinates': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                }
            else:
                logger.info(f"Class ID {class_id} shelf değil, atlanıyor")
        else:
            logger.warning(f"YOLO satırında yeterli değer yok: {yolo_line} (beklenen: 9, bulunan: {len(parts)})")
    except Exception as e:
        logger.error(f"YOLO parse hatası: {e} - Satır: {yolo_line}")
    return None

def calculate_iou_from_polygons(polygon1, polygon2, img_height, img_width):
    """İki poligon arasında IoU hesapla"""
    try:
        mask1 = create_mask_from_polygon(polygon1, img_height, img_width)
        mask2 = create_mask_from_polygon(polygon2, img_height, img_width)
        
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    except Exception as e:
        logger.error(f"IoU hesaplama hatası: {e}")
        return 0.0

def draw_polygon_on_image(image, points, color, thickness=2):
    """Görsel üzerine poligon çiz"""
    if not points or len(points) < 3:
        return image
    
    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], True, color, thickness)
    return image

def draw_filled_polygon_on_image(image, points, color_bgr, alpha=0.35, outline_thickness=2):
    """Görsel üzerine içi boyalı poligon çiz (alfa karışımlı)"""
    if not points or len(points) < 3:
        return image
    overlay = image.copy()
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color_bgr)
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    # İsteğe bağlı: kontur çiz
    cv2.polylines(blended, [pts], True, color_bgr, outline_thickness)
    return blended

def draw_text_on_image(image, text, position, color=(255, 255, 255), font_scale=0.6, thickness=2):
    """Görsel üzerine metin çiz"""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return image

def image_to_data_url(image_bgr):
    success, buf = cv2.imencode('.jpg', image_bgr)
    if not success:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"

def fig_to_data_url(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{b64}"

def create_step_visualizations(image_path, mask_data, yolo_data, output_dir):
    """Adım adım görselleştirmeler oluştur"""
    logger.info(f"Görsel yükleniyor: {image_path}")
    
    # Görseli yükle
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Görsel yüklenemedi!")
        return None
    
    img_height, img_width = image.shape[:2]
    logger.info(f"Görsel boyutları: {img_width}x{img_height}")
    
    # JSON'dan poligon noktalarını al
    logger.info("JSON verisi parse ediliyor...")
    instances = mask_data.get('instances', [])
    metadata = mask_data.get('metadata', {})
    logger.info(f"Toplam instance sayısı: {len(instances)}")
    
    shelf_instances = []
    
    for i, instance in enumerate(instances):
        instance_type = instance.get('type')
        class_name = instance.get('className')
        logger.info(f"Instance {i+1}: type={instance_type}, className={class_name}")
        
        if instance.get('type') == 'polygon' and instance.get('className') == 'shelf':
            points = instance.get('points', [])
            logger.info(f"Shelf instance {i+1}: {len(points)} nokta")
            
            if len(points) >= 6:  # En az 3 nokta (6 koordinat)
                # Noktaları [x,y] formatına çevir
                polygon_points = []
                for j in range(0, len(points), 2):
                    if j + 1 < len(points):
                        polygon_points.append([points[j], points[j+1]])
                
                if len(polygon_points) >= 3:
                    shelf_instances.append(polygon_points)
                    logger.info(f"Shelf {len(shelf_instances)} eklendi: {len(polygon_points)} nokta")
                else:
                    logger.warning(f"Shelf {i+1}: Yeterli nokta yok ({len(polygon_points)})")
            else:
                logger.warning(f"Shelf {i+1}: Yeterli koordinat yok ({len(points)})")
    
    logger.info(f"Toplam {len(shelf_instances)} shelf instance bulundu")
    
    # YOLO verilerini parse et
    yolo_obbs = []
    if yolo_data:
        logger.info(f"YOLO verisi yükleniyor... Toplam satır sayısı: {len(yolo_data.strip().split('\n'))}")
        yolo_lines = yolo_data.strip().split('\n')
        for i, line in enumerate(yolo_lines):
            if line.strip():
                logger.info(f"Satır {i+1} işleniyor: {line.strip()}")
                try:
                    yolo_obb = parse_yolo_obb(line, img_width, img_height)
                    if yolo_obb:
                        yolo_obbs.append(yolo_obb)
                        logger.info(f"Satır {i+1} başarıyla parse edildi")
                    else:
                        logger.info(f"Satır {i+1} parse edildi ama sonuç None")
                except Exception as e:
                    logger.warning(f"Satır {i+1} parse edilemedi: {line.strip()} - Hata: {e}")
                    continue
        
        logger.info(f"Toplam {len(yolo_obbs)} YOLO OBB parse edildi")
    else:
        logger.info("YOLO verisi bulunamadı")
    
    # 1. ADIM: Orijinal görsel + Mask poligonları + YOLO OBB'leri
    step1_image = image.copy()
    
    # Mask poligonlarını çiz (yeşil) ve içini boya
    # Colors in BGR
    GREEN = (0, 255, 0)        # #00ff00
    YOLO_PURPLE = (128, 0, 128) # Koyu mor (#800080) BGR yaklaşık
    RED = (0, 0, 255)          # #ff0000 (RGB) -> BGR
    BLUE = (255, 0, 0)         # #0000ff (RGB) -> BGR

    for polygon_points in shelf_instances:
        step1_image = draw_filled_polygon_on_image(step1_image, polygon_points, GREEN, alpha=0.35, outline_thickness=2)
    
    # YOLO OBB'lerini çiz (koyu mor)
    for yolo_obb in yolo_obbs:
        step1_image = draw_polygon_on_image(step1_image, yolo_obb['coordinates'], YOLO_PURPLE, 2)

    # Step1: Her shelf için (varsa) eski (YOLO) IoU değerini yaz (sadece sayı)
    for idx, polygon_points in enumerate(shelf_instances):
        if idx < len(yolo_obbs):
            old_iou_val = calculate_iou_from_polygons(polygon_points, yolo_obbs[idx]['coordinates'], img_height, img_width)
            cx = int(np.mean([p[0] for p in polygon_points])); cy = int(np.mean([p[1] for p in polygon_points]))
            step1_image = draw_text_on_image(step1_image, f"IoU={old_iou_val:.3f}", (cx - 35, cy - 8), color=(255,255,255), font_scale=0.8, thickness=2)

    step1_path = os.path.join(output_dir, 'step1_original.jpg')
    cv2.imwrite(step1_path, step1_image)
    
    # 2. ADIM: Trapezoid fit edilmiş görsel
    step2_image = image.copy()
    trapezoid_results = []
    
    for i, polygon_points in enumerate(shelf_instances):
        # Mask'tan trapezoid fit et
        polygon_mask = create_mask_from_polygon(polygon_points, img_height, img_width)
        trapezoid_result = obb_generator.mask_to_trapezoid(polygon_mask)
        
        if trapezoid_result:
            trapezoid_mask, trapezoid_corners = trapezoid_result
            trapezoid_results.append((polygon_points, trapezoid_corners))
            
            # Trapezoid'i çiz (kırmızı)
            step2_image = draw_polygon_on_image(step2_image, trapezoid_corners, RED, 3)
    
    step2_path = os.path.join(output_dir, 'step2_trapezoid.jpg')
    cv2.imwrite(step2_path, step2_image)
    
    # 3. ADIM: Yeni OBB'ler + IoU karşılaştırması
    step3_image = image.copy()
    # Mask dolgusunu Step3'te de uygula (arka plan)
    for polygon_points in shelf_instances:
        step3_image = draw_filled_polygon_on_image(step3_image, polygon_points, GREEN, alpha=0.25, outline_thickness=2)
    iou_results = []
    new_obb_entries = []
    
    for i, (polygon_points, trapezoid_corners) in enumerate(trapezoid_results):
        # Trapezoid'den OBB oluştur
        # NOT: trapezoid_corners piksel cinsinden. Converter normalize (0-1) bekliyor.
        norm_trapezoid = [(float(x) / img_width, float(y) / img_height) for x, y in trapezoid_corners]
        obb_result = trapezoid_converter.trapezoid_to_obb_optimized(norm_trapezoid, (img_height, img_width))
        
        if obb_result:
            # cv2.minAreaRect returns (center, size, angle) where center is (x, y) and size is (width, height)
            center, size, angle = obb_result
            center_x, center_y = center
            width, height = size
            
            # Convert to our expected format: (center_x, center_y, width, height, angle)
            obb_tuple = (center_x, center_y, width, height, angle)
            obb_coordinates = trapezoid_converter.obb_to_polygon_fast(obb_tuple)
            
            if obb_coordinates:
                # Yeni OBB'yi çiz (mavi)
                step3_image = draw_polygon_on_image(step3_image, obb_coordinates, BLUE, 3)
                
                # Yeni OBB IoU'sunu hesapla
                new_iou = calculate_iou_from_polygons(polygon_points, obb_coordinates, img_height, img_width)
                
                # Eski YOLO OBB IoU'sunu hesapla (varsa)
                old_iou = 0.0
                if i < len(yolo_obbs):
                    old_iou = calculate_iou_from_polygons(polygon_points, yolo_obbs[i]['coordinates'], img_height, img_width)
                
                iou_results.append({
                    'shelf_id': i + 1,
                    'old_iou': old_iou,
                    'new_iou': new_iou,
                    'improvement': new_iou - old_iou
                })
                
                # Sadece ilgili shelf merkezine yeni IoU değerini yaz
                center_x = int(np.mean([p[0] for p in polygon_points]))
                center_y = int(np.mean([p[1] for p in polygon_points]))
                step3_image = draw_text_on_image(step3_image, f"IoU={new_iou:.3f}", (center_x - 35, center_y - 8), color=(255,255,255), font_scale=0.8, thickness=2)

                # OBB kaydı hazırla
                new_obb_entries.append({
                    'shelf_id': i + 1,
                    'mask_polygon': polygon_points,
                    'new_obb_coordinates': [list(map(float, p)) for p in obb_coordinates],
                    'old_iou': float(old_iou),
                    'new_iou': float(new_iou)
                })
    
    # Mean IoU'ları hesapla ve JSON çıktısı yaz
    mean_old_iou = 0.0
    mean_new_iou = 0.0
    if iou_results:
        mean_old_vals = [r['old_iou'] for r in iou_results if r['old_iou'] > 0]
        mean_old_iou = float(np.mean(mean_old_vals)) if len(mean_old_vals) > 0 else 0.0
        mean_new_iou = float(np.mean([r['new_iou'] for r in iou_results]))

    # Yeni OBB JSON'u kaydet
    obb_json = {
        'image_name': metadata.get('name', os.path.basename(image_path)),
        'image_width': int(img_width),
        'image_height': int(img_height),
        'mean_old_iou': mean_old_iou,
        'mean_new_iou': mean_new_iou,
        'instances': new_obb_entries,
    }
    obb_json_path = os.path.join(output_dir, 'step3_new_obb.json')
    try:
        with open(obb_json_path, 'w', encoding='utf-8') as f:
            json.dump(obb_json, f, ensure_ascii=False, indent=2)
        logger.info(f"Yeni OBB JSON kaydedildi: {obb_json_path}")
    except Exception as e:
        logger.error(f"Yeni OBB JSON yazılamadı: {e}")
    
    step3_path = os.path.join(output_dir, 'step3_obb_comparison.jpg')
    cv2.imwrite(step3_path, step3_image)
    
    logger.info(f"Görselleştirme dosyaları oluşturuldu:")
    logger.info(f"Step 1: {step1_path} - Exists: {os.path.exists(step1_path)}")
    logger.info(f"Step 2: {step2_path} - Exists: {os.path.exists(step2_path)}")
    logger.info(f"Step 3: {step3_path} - Exists: {os.path.exists(step3_path)}")
    
    return {
        'step1': step1_path,
        'step2': step2_path,
        'step3': step3_path,
        'iou_results': iou_results,
        'obb_json_path': obb_json_path,
        'instances': new_obb_entries
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/viewer')
def viewer():
    # İkinci sayfa için template
    return render_template('viewer.html')

@app.route('/viewer_preview', methods=['POST'])
def viewer_preview():
    try:
        image_file = request.files.get('image_file')
        mask_file = request.files.get('mask_file')
        obb_file = request.files.get('obb_file')
        if not image_file or not mask_file or not obb_file:
            return jsonify({'success': False, 'error': 'Gerekli dosyalar eksik'}), 400

        tmp_dir = tempfile.mkdtemp()
        img_path = os.path.join(tmp_dir, 'img.jpg')
        mask_path = os.path.join(tmp_dir, 'mask.json')
        obb_path = os.path.join(tmp_dir, 'obb.json')
        image_file.save(img_path)
        mask_file.save(mask_path)
        obb_file.save(obb_path)

        image = cv2.imread(img_path)
        if image is None:
            return jsonify({'success': False, 'error': 'Görsel yüklenemedi'}), 400
        h, w = image.shape[:2]

        with open(mask_path, 'r', encoding='utf-8') as f:
            mask_json = json.load(f)
        with open(obb_path, 'r', encoding='utf-8') as f:
            obb_json = json.load(f)

        # Mask polygonları (shelf)
        shelf_instances = [inst.get('points', []) for inst in mask_json.get('instances', []) if inst.get('type')=='polygon' and inst.get('className')=='shelf']

        # Sol görüntü: mask dolgulu
        left = image.copy()
        for poly in shelf_instances:
            left = draw_filled_polygon_on_image(left, [[poly[i], poly[i+1]] for i in range(0,len(poly),2)], (0,255,0), 0.35, 2)

        # Sağ görüntü: mask dolgulu + yeni OBB çizimi ve IoU yazımı
        right = image.copy()
        for poly in shelf_instances:
            right = draw_filled_polygon_on_image(right, [[poly[i], poly[i+1]] for i in range(0,len(poly),2)], (0,255,0), 0.25, 2)

        instances = obb_json.get('instances', [])
        per_shelf = []
        for inst in instances:
            obb_coords = inst.get('new_obb_coordinates', [])
            if len(obb_coords) >= 3:
                right = draw_polygon_on_image(right, obb_coords, (255,0,0), 3)
        
        # IoU hesapları mask polygonları ile obb_json içindeki koordinatlar arasında
        for idx, inst in enumerate(instances):
            if idx < len(shelf_instances):
                poly_pts = [[shelf_instances[idx][i], shelf_instances[idx][i+1]] for i in range(0,len(shelf_instances[idx]),2)]
                obb_coords = inst.get('new_obb_coordinates', [])
                iou_val = 0.0
                if len(obb_coords) >= 3:
                    iou_val = calculate_iou_from_polygons(poly_pts, obb_coords, h, w)
                    cx = int(np.mean([p[0] for p in poly_pts])); cy = int(np.mean([p[1] for p in poly_pts]))
                    right = draw_text_on_image(right, f"IoU={iou_val:.3f}", (cx-35, cy-8), (255,255,255), 0.8, 2)
                per_shelf.append({'iou': iou_val})

        mean_iou = float(np.mean([s['iou'] for s in per_shelf])) if len(per_shelf)>0 else 0.0

        return jsonify({
            'success': True,
            'left': image_to_data_url(left),
            'right': image_to_data_url(right),
            'mean_iou': mean_iou,
            'instances': per_shelf
        })
    except Exception as e:
        logger.error(f"viewer_preview error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/batch_process', methods=['POST'])
def batch_process():
    try:
        zip_file = request.files.get('dataset_zip')
        if not zip_file:
            return jsonify({'success': False, 'error': 'ZIP dosyası gerekli'}), 400

        token = str(uuid.uuid4())
        tmp_dir = tempfile.mkdtemp()
        extract_dir = os.path.join(tmp_dir, 'input')
        output_root = os.path.join(tmp_dir, 'output')
        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(output_root, exist_ok=True)

        zip_path = os.path.join(tmp_dir, 'dataset.zip')
        zip_file.save(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Haritalama: isim -> tam yol
        all_files = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                all_files.append(os.path.join(root, f))
        images_by_name = {os.path.basename(p): p for p in all_files if os.path.splitext(p)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp'} }
        jsons = [p for p in all_files if p.lower().endswith('.json')]
        yolo_by_stem = {}
        for p in all_files:
            if p.lower().endswith('.txt'):
                yolo_by_stem[os.path.splitext(os.path.basename(p))[0]] = p

        all_old = []
        all_new = []
        improvements = []
        per_image_stats = []
        new_json_paths = []

        # Görselleştirme örnekleri için en iyi/kötü listeleri
        improved_examples = []  # dicts: {'imp','image','mean_old','mean_new','path'}
        degraded_examples = []  # same

        for mask_json_path in jsons:
            try:
                with open(mask_json_path, 'r', encoding='utf-8') as f:
                    mask_data = json.load(f)
                # Image adı metadata'dan veya aynı stem
                image_name = mask_data.get('metadata', {}).get('name')
                if not image_name:
                    image_name = os.path.splitext(os.path.basename(mask_json_path))[0] + '.jpg'
                image_path = images_by_name.get(image_name)
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Görsel bulunamadı: {image_name}")
                    continue
                stem = os.path.splitext(os.path.basename(image_name))[0]
                yolo_txt_path = yolo_by_stem.get(stem)
                yolo_data = None
                if yolo_txt_path and os.path.exists(yolo_txt_path):
                    with open(yolo_txt_path, 'r') as yf:
                        yolo_data = yf.read()

                per_output = os.path.join(output_root, stem)
                os.makedirs(per_output, exist_ok=True)
                res = create_step_visualizations(image_path, mask_data, yolo_data, per_output)
                if not res:
                    continue
                # IoU toplama
                if res['iou_results']:
                    img_old = [r['old_iou'] for r in res['iou_results'] if r['old_iou'] > 0]
                    img_new = [r['new_iou'] for r in res['iou_results']]
                    if img_old:
                        all_old.extend(img_old)
                    all_new.extend(img_new)
                    improvements.extend([r['improvement'] for r in res['iou_results']])
                    mean_old_i = float(np.mean(img_old)) if len(img_old) > 0 else 0.0
                    mean_new_i = float(np.mean(img_new)) if len(img_new) > 0 else 0.0
                    imp_val = float(mean_new_i - mean_old_i)
                    per_image_stats.append({
                        'image': image_name,
                        'mean_old': mean_old_i,
                        'mean_new': mean_new_i,
                        'count': len(img_new)
                    })
                    # Görselleştirme yolu
                    step3_vis = os.path.join(per_output, 'step3_obb_comparison.jpg')
                    if os.path.exists(step3_vis):
                        entry = {'imp': imp_val, 'image': image_name, 'mean_old': mean_old_i, 'mean_new': mean_new_i, 'path': step3_vis}
                        if imp_val >= 0:
                            improved_examples.append(entry)
                        else:
                            degraded_examples.append(entry)
                # Yeni OBB JSON'u topla ve yeniden adlandır
                obb_json_src = res.get('obb_json_path')
                if obb_json_src and os.path.exists(obb_json_src):
                    target = os.path.join(output_root, f"{stem}_new_obb.json")
                    shutil.copyfile(obb_json_src, target)
                    new_json_paths.append(target)
            except Exception as e:
                logger.error(f"Batch item hata: {e}")
                continue

        mean_old = float(np.mean(all_old)) if len(all_old) > 0 else 0.0
        mean_new = float(np.mean(all_new)) if len(all_new) > 0 else 0.0
        mean_imp = float(mean_new - mean_old)
        improved_count = int(sum(1 for v in improvements if v > 0))
        degraded_count = int(sum(1 for v in improvements if v < 0))
        unchanged_count = int(sum(1 for v in improvements if v == 0))

        # Histogram grafiklerini üret
        charts = {}
        if len(all_old) > 0 and len(all_new) > 0:
            bins = np.linspace(0, 1, 21)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(all_old, bins=bins, alpha=0.6, label='Old IoU', color='#8e44ad')
            ax.hist(all_new, bins=bins, alpha=0.6, label='New IoU', color='#2980b9')
            ax.set_xlabel('IoU'); ax.set_ylabel('Count'); ax.legend(); ax.set_title('IoU Distribution')
            charts['iou_distribution'] = fig_to_data_url(fig)

            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.hist(improvements, bins=21, color='#27ae60', alpha=0.8)
            ax2.set_xlabel('Improvement (New-Old)'); ax2.set_ylabel('Count'); ax2.set_title('Improvement Distribution')
            charts['improvement_distribution'] = fig_to_data_url(fig2)

        # En iyi/kötü örneklerden ilk 3'ü seç
        improved_examples.sort(key=lambda x: x['imp'], reverse=True)
        degraded_examples.sort(key=lambda x: x['imp'])  # en negatif başta
        top_improved = improved_examples[:3]
        top_degraded = degraded_examples[:3]

        # Yeni dataset zip hazırla
        zip_out_path = os.path.join(tmp_dir, 'new_obb_dataset.zip')
        with zipfile.ZipFile(zip_out_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in new_json_paths:
                arcname = os.path.relpath(p, output_root)
                z.write(p, arcname)

        batch_jobs[token] = {
            'tmp_dir': tmp_dir,
            'zip_path': zip_out_path
        }

        # Örnek görselleri data URL olarak döndür
        example_images = {
            'improved': [
                {
                    'src': image_to_data_url(cv2.imread(e['path'])),
                    'image': e['image'],
                    'mean_old': e['mean_old'],
                    'mean_new': e['mean_new'],
                    'imp': e['imp']
                }
                for e in top_improved if os.path.exists(e['path'])
            ],
            'degraded': [
                {
                    'src': image_to_data_url(cv2.imread(e['path'])),
                    'image': e['image'],
                    'mean_old': e['mean_old'],
                    'mean_new': e['mean_new'],
                    'imp': e['imp']
                }
                for e in top_degraded if os.path.exists(e['path'])
            ]
        }

        return jsonify({
            'success': True,
            'token': token,
            'summary': {
                'mean_old': mean_old,
                'mean_new': mean_new,
                'mean_improvement': mean_imp,
                'improved': improved_count,
                'degraded': degraded_count,
                'unchanged': unchanged_count,
                'images_processed': len(per_image_stats)
            },
            'charts': charts,
            'download_url': f"/batch_download/{token}",
            'per_image': per_image_stats[:100],
            'examples': example_images
        })
    except Exception as e:
        logger.error(f"batch_process error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/batch_download/<token>')
def batch_download(token):
    job = batch_jobs.get(token)
    if not job:
        return jsonify({'error': 'İş bulunamadı'}), 404
    path = job.get('zip_path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Dosya bulunamadı'}), 404
    return send_file(path, mimetype='application/zip', as_attachment=True, download_name='new_obb_dataset.zip')

@app.route('/upload_files', methods=['POST'])
def upload_files():
    global current_data
    
    try:
        # Dosyaları al
        mask_file = request.files.get('mask_file')
        yolo_file = request.files.get('yolo_file')
        image_file = request.files.get('image_file')
        
        if not mask_file or not image_file:
            return jsonify({'error': 'Mask ve görsel dosyaları gerekli'}), 400
        
        # Geçici dizin oluştur
        temp_dir = tempfile.mkdtemp()
        
        # Dosyaları kaydet
        mask_path = os.path.join(temp_dir, 'mask.json')
        image_path = os.path.join(temp_dir, 'image.jpg')
        yolo_path = os.path.join(temp_dir, 'yolo.txt') if yolo_file else None
        
        mask_file.save(mask_path)
        image_file.save(image_path)
        if yolo_file:
            yolo_file.save(yolo_path)
        
        # JSON verisini yükle
        with open(mask_path, 'r', encoding='utf-8') as f:
            mask_data = json.load(f)
        
        # YOLO verisini yükle (varsa)
        yolo_data = None
        if yolo_path and os.path.exists(yolo_path):
            with open(yolo_path, 'r') as f:
                yolo_data = f.read()
        
        # Adım adım görselleştirmeler oluştur
        logger.info("Görselleştirme başlatılıyor...")
        try:
            results = create_step_visualizations(image_path, mask_data, yolo_data, temp_dir)
            logger.info("Görselleştirme tamamlandı")
        except Exception as e:
            logger.error(f"Görselleştirme hatası: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        if results:
            current_data = {
                'temp_dir': temp_dir,
                'mask_data': mask_data,
                'yolo_data': yolo_data,
                'image_path': image_path,
                'results': results
            }
            
            logger.info("JSON response hazırlanıyor...")
            response_data = {
                'success': True,
                'message': 'Dosyalar başarıyla yüklendi ve işlendi',
                'step1': '/get_visualization/step1',
                'step2': '/get_visualization/step2',
                'step3': '/get_visualization/step3',
                'iou_results': results['iou_results'],
                'obb_json': '/get_visualization/obb_json'
            }
            logger.info(f"Response data: {response_data}")
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Görselleştirme oluşturulamadı'}), 500
            
    except Exception as e:
        logger.error(f"Dosya yükleme hatası: {e}")
        return jsonify({'error': f'Dosya yükleme hatası: {str(e)}'}), 500

@app.route('/get_visualization/<step>')
def get_visualization(step):
    global current_data
    
    logger.info(f"Görselleştirme isteği: step={step}")
    logger.info(f"current_data keys: {list(current_data.keys()) if current_data else 'None'}")
    
    if not current_data:
        logger.error("current_data bulunamadı")
        return jsonify({'error': 'Görselleştirme bulunamadı'}), 404
    
    if 'results' not in current_data:
        logger.error("results bulunamadı")
        return jsonify({'error': 'Görselleştirme bulunamadı'}), 404
    
    logger.info(f"results keys: {list(current_data['results'].keys())}")
    
    # Frontend 'step1', 'step2', 'step3' gönderiyor. Anahtarları doğrudan kullan.
    step_key = step
    
    # Eğer sadece '1', '2', '3' gelirse, 'step' prefix ekle
    if step_key not in current_data['results']:
        candidate = f'step{step}'
        if candidate in current_data['results']:
            step_key = candidate
    
    if step_key == 'obb_json':
        # JSON dosyasını servis et
        obb_path = current_data['results'].get('obb_json_path')
        if not obb_path or not os.path.exists(obb_path):
            return jsonify({'error': 'OBB JSON bulunamadı'}), 404
        return send_file(obb_path, mimetype='application/json', as_attachment=True, download_name='new_obb.json')

    step_path = current_data['results'].get(step_key)
    logger.info(f"Resolved step key: {step_key} -> path: {step_path}")
    
    if not step_path:
        logger.error(f"Step {step} path bulunamadı")
        return jsonify({'error': f'Adım {step} görselleştirmesi bulunamadı'}), 404
    
    if not os.path.exists(step_path):
        logger.error(f"Step {step} dosyası bulunamadı: {step_path}")
        return jsonify({'error': f'Adım {step} dosyası bulunamadı'}), 404
    
    logger.info(f"Görselleştirme dosyası gönderiliyor: {step_path}")
    return send_file(step_path, mimetype='image/jpeg')

@app.route('/get_process_info')
def get_process_info():
    global current_data
    
    if not current_data:
        return jsonify({'error': 'Veri bulunamadı'}), 404
    
    try:
        mask_data = current_data['mask_data']
        instances = mask_data.get('instances', [])
        
        # İstatistikler
        total_instances = len(instances)
        shelf_instances = len([i for i in instances if i.get('className') == 'shelf'])
        shelf_space_instances = len([i for i in instances if i.get('className') == 'shelf-space'])
        
        # Metadata
        metadata = mask_data.get('metadata', {})
        image_name = metadata.get('name', 'Unknown')
        image_width = metadata.get('width', 0)
        image_height = metadata.get('height', 0)
        
        # IoU sonuçları
        iou_results = current_data.get('results', {}).get('iou_results', [])
        
        return jsonify({
            'image_name': image_name,
            'image_width': image_width,
            'image_height': image_height,
            'total_instances': total_instances,
            'shelf_instances': shelf_instances,
            'shelf_space_instances': shelf_space_instances,
            'has_yolo_data': current_data['yolo_data'] is not None,
            'iou_results': iou_results
        })
        
    except Exception as e:
        logger.error(f"Bilgi alma hatası: {e}")
        return jsonify({'error': f'Bilgi alma hatası: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
