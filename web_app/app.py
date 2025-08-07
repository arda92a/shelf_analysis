from flask import Flask, request, jsonify, send_file, render_template
import os
import tempfile
import zipfile
import shutil
from modules.dataset_loader import DatasetLoader
from modules.visualizer import Visualizer
from modules.image_processor import ImageProcessor
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = None  # Sınırsız dosya boyutu

# Global değişkenler
current_dataset = None
images_data = []
current_image_index = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global current_dataset, images_data, current_image_index
    if 'dataset' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'Sadece ZIP dosyası yükleyebilirsiniz'}), 400
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)
        
        extract_dir = os.path.join(temp_dir, 'extracted')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        dataset_loader = DatasetLoader(extract_dir)
        images_data = dataset_loader.load_all_images()
        current_dataset = extract_dir
        current_image_index = 0
        
        return jsonify({
            'success': True,
            'total_images': len(images_data),
            'message': f'{len(images_data)} görsel yüklendi'
        })
    except Exception as e:
        return jsonify({'error': f'Dataset yükleme hatası: {str(e)}'}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.route('/get_image_list')
def get_image_list():
    global images_data
    images = []
    for i, img_data in enumerate(images_data):
        instances = len(img_data.get('instances', []))
        images.append({
            'index': i,
            'name': img_data.get('image_name', f'Image {i}'),
            'instances': instances
        })
    return jsonify({'images': images})

@app.route('/get_image/<int:image_index>')
def get_image(image_index):
    global images_data, current_image_index
    if image_index >= len(images_data):
        return jsonify({'error': 'Geçersiz görsel indeksi'}), 400
    
    current_image_index = image_index
    img_data = images_data[image_index]
    
    try:
        # Visualizer ve ImageProcessor oluştur
        visualizer = Visualizer()
        image_processor = ImageProcessor()
        
        # Görselleştirmeleri oluştur
        left_image = visualizer.create_segmentation_visualization(img_data)
        right_image = visualizer.create_new_obb_visualization(img_data)
        
        # Görselleri kaydet
        left_image_path = image_processor.save_image(left_image, f'left_{image_index}.png')
        right_image_path = image_processor.save_image(right_image, f'right_{image_index}.png')
        
        # Mean IoU değerlerini hesapla (sadece yeni OBB için)
        instances = img_data.get('instances', [])
        new_ious = []
        
        for instance in instances:
            if 'new_obb' in instance and 'iou_with_gt' in instance['new_obb']:
                new_ious.append(instance['new_obb']['iou_with_gt'])
        
        new_mean_iou = np.mean(new_ious) if new_ious else 0.0
        
        return jsonify({
            'current_index': image_index,
            'total_images': len(images_data),
            'image_name': img_data.get('image_name', f'Image {image_index}'),
            'left_image': f'/get_image_file/{image_index}/left',
            'right_image': f'/get_image_file/{image_index}/right',
            'new_mean_iou': new_mean_iou
        })
    except Exception as e:
        return jsonify({'error': f'Görsel oluşturma hatası: {str(e)}'}), 500

@app.route('/get_image_file/<int:image_index>/<side>')
def get_image_file(image_index, side):
    global images_data
    if image_index >= len(images_data):
        return jsonify({'error': 'Geçersiz görsel indeksi'}), 400
    
    img_data = images_data[image_index]
    
    try:
        visualizer = Visualizer()
        image_processor = ImageProcessor()
        
        if side == 'left':
            image = visualizer.create_segmentation_visualization(img_data)
            image_path = image_processor.save_image(image, f'left_{image_index}.png')
        elif side == 'right':
            image = visualizer.create_new_obb_visualization(img_data)
            image_path = image_processor.save_image(image, f'right_{image_index}.png')
        else:
            return jsonify({'error': 'Geçersiz taraf'}), 400
        
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': f'Görsel dosyası hatası: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 