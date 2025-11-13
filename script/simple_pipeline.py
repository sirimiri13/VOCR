"""
Pipeline đơn giản: Xử lý ảnh → Detect bbox → Align với TXT → Crop dataset

INPUT:
- data/raw/images/page_001.png, page_002.png, ...
- data/raw/gt/page_001.txt, page_002.txt, ...

OUTPUT:
- data/processed/dataset/train/
- data/processed/dataset/val/
- data/processed/dataset/test/
"""

import os
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
from tqdm import tqdm
import json
import random
import re

# ===================== CONFIG =====================
class Config:
    # Input
    RAW_IMAGES_DIR = Path('/Users/huonglam/Library/Mobile Documents/com~apple~CloudDocs/Documents/Master/Tốt nghiệp/VOCR/data/convert/image')
    RAW_GT_DIR = Path('/Users/huonglam/Library/Mobile Documents/com~apple~CloudDocs/Documents/Master/Tốt nghiệp/VOCR/data/processed/text')
    
    # Output
    OUTPUT_DIR = Path('/Users/huonglam/Library/Mobile Documents/com~apple~CloudDocs/Documents/Master/Tốt nghiệp/VOCR/data/processed')
    PREPROCESSED_DIR = OUTPUT_DIR / 'preprocessed'
    DETECTION_DIR = OUTPUT_DIR / 'detection'
    DATASET_DIR = OUTPUT_DIR / 'dataset'
    
    # Preprocessing
    DPI = 300
    REMOVE_BORDERS = True
    ENHANCE_CONTRAST = True
    DENOISE = True
    
    # OCR
    OCR_LANG = 'vi'
    USE_GPU = True
    
    # Alignment
    SIMILARITY_THRESHOLD = 0.6
    
    # Dataset split
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    
    # Test mode
    TEST_MODE = True
    TEST_PAGES = 2


# ===================== BƯỚC 1: PREPROCESS =====================
def preprocess_image(image_path, output_path=None):
    """Xử lý 1 ảnh: xóa viền, nhiễu, tăng tương phản"""
    
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️ Cannot read image: {image_path}")
            return None
        
        # 1. Xóa viền
        if Config.REMOVE_BORDERS:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(cv2.bitwise_not(gray))
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2*padding)
                h = min(img.shape[0] - y, h + 2*padding)
                img = img[y:y+h, x:x+w]
        
        # 2. Xóa nhiễu
        if Config.DENOISE:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 3. Tăng độ tương phản
        if Config.ENHANCE_CONTRAST:
            # Nếu ảnh đã là grayscale từ bước 1, không cần convert lại
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(gray)
        
        # Kiểm tra ảnh có hợp lệ không
        if img is None or img.size == 0:
            print(f"⚠️ Processed image is empty: {image_path}")
            return None
        
        if output_path:
            success = cv2.imwrite(str(output_path), img)
            if not success:
                print(f"⚠️ Failed to save image: {output_path}")
                return None
        
        return img
        
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return None


def preprocess_all_images():
    """Xử lý tất cả ảnh"""
    
    print("\n🎨 BƯỚC 1: Preprocess Images")
    print("="*60)
    
    # Tạo thư mục output
    Config.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra input directory
    if not Config.RAW_IMAGES_DIR.exists():
        print(f"❌ Input directory not found: {Config.RAW_IMAGES_DIR}")
        return []
    
    # Tìm tất cả ảnh
    image_files = sorted(list(Config.RAW_IMAGES_DIR.glob('*.png')) + 
                        list(Config.RAW_IMAGES_DIR.glob('*.jpg')) +
                        list(Config.RAW_IMAGES_DIR.glob('*.jpeg')))
    
    if Config.TEST_MODE:
        image_files = image_files[:Config.TEST_PAGES]
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("❌ No image files found!")
        print(f"   Check directory: {Config.RAW_IMAGES_DIR}")
        return []
    
    processed = []
    failed = []
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            output_path = Config.PREPROCESSED_DIR / img_path.name
            
            result = preprocess_image(img_path, output_path)
            if result is not None:
                processed.append(output_path)
            else:
                failed.append(img_path.name)
                
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
            failed.append(img_path.name)
    
    print(f"✅ Processed {len(processed)} images")
    if failed:
        print(f"❌ Failed to process {len(failed)} images: {failed}")
    
    return processed


# ===================== BƯỚC 2: DETECT BBOX =====================
def detect_boxes_paddleocr(image_paths):
    """Dùng PaddleOCR detect bounding boxes"""
    
    print("\n🔍 BƯỚC 2: Detect Bounding Boxes")
    print("="*60)
    
    Config.DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Khởi tạo OCR
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=Config.OCR_LANG,
    )
    
    all_detections = {}
    
    for img_path in tqdm(image_paths, desc="Detecting"):
        # Detect
        try:
            result = ocr.ocr(str(img_path))
        except Exception as e:
            print(f"⚠️ OCR failed for {img_path.name}: {e}")
            continue
        
        if not result or not result[0]:
            print(f"⚠️ No text detected in {img_path.name}")
            continue
        
        boxes = []
        for line in result[0]:
            try:
                # Kiểm tra cấu trúc dữ liệu trả về
                if len(line) < 2:
                    print(f"⚠️ Unexpected line structure in {img_path.name}: {line}")
                    continue
                
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                
                # Kiểm tra text và confidence
                if len(line[1]) >= 2:
                    text = line[1][0]
                    conf = line[1][1]
                elif len(line[1]) >= 1:
                    # Trường hợp chỉ có text, không có confidence
                    text = line[1][0]
                    conf = 1.0  # Default confidence
                else:
                    print(f"⚠️ Invalid text structure in {img_path.name}: {line[1]}")
                    continue
                
                # Kiểm tra bbox hợp lệ
                if not bbox or len(bbox) != 4:
                    print(f"⚠️ Invalid bbox in {img_path.name}: {bbox}")
                    continue
                
                boxes.append({
                    'bbox': bbox,
                    'ocr_text': text,
                    'confidence': conf
                })
                
            except Exception as e:
                print(f"⚠️ Error parsing line in {img_path.name}: {e}")
                print(f"   Line structure: {line}")
                continue
        
        if boxes:  # Chỉ thêm vào nếu có boxes
            all_detections[img_path.name] = {
                'image_path': str(img_path),
                'boxes': boxes
            }
            
            print(f"   {img_path.name}: {len(boxes)} boxes detected")
            
            # Save visualization
            try:
                visualize_boxes(img_path, boxes, 
                               Config.DETECTION_DIR / f'vis_{img_path.name}')
            except Exception as e:
                print(f"⚠️ Visualization failed for {img_path.name}: {e}")
        else:
            print(f"   {img_path.name}: No valid boxes found")
    
    # Save JSON
    try:
        with open(Config.DETECTION_DIR / 'detections.json', 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save detections.json: {e}")
    
    total = sum(len(d['boxes']) for d in all_detections.values())
    print(f"✅ Detected {total} boxes from {len(all_detections)} images")
    
    return all_detections


def visualize_boxes(image_path, boxes, output_path):
    """Vẽ boxes lên ảnh"""
    img = cv2.imread(str(image_path))
    
    for box_info in boxes:
        bbox = box_info['bbox']
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(img, [points], True, (0, 255, 0), 2)
        
        # Label
        x, y = int(bbox[0][0]), int(bbox[0][1]) - 10
        label = f"{box_info['ocr_text'][:15]}..."
        cv2.putText(img, label, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(str(output_path), img)


# ===================== BƯỚC 3: ALIGN VỚI TXT =====================
def load_ground_truth_txt(txt_path):
    """Đọc file TXT ground truth"""
    if not txt_path.exists():
        return []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Mỗi dòng là 1 text ground truth
    return [line.strip() for line in lines if line.strip()]


def calculate_similarity(text1, text2):
    """Tính độ giống nhau giữa 2 text"""
    from difflib import SequenceMatcher
    
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    return SequenceMatcher(None, text1, text2).ratio()


def align_boxes_with_txt(detections):
    """Align boxes với ground truth từ TXT files"""
    
    print("\n🔗 BƯỚC 3: Align với Ground Truth TXT")
    print("="*60)
    
    aligned_data = {}
    
    for img_name, detection in tqdm(detections.items(), desc="Aligning"):
        # Tìm file TXT tương ứng
        txt_name = Path(img_name).stem + '.txt'
        txt_path = Config.RAW_GT_DIR / txt_name
        
        gt_lines = load_ground_truth_txt(txt_path)
        
        if not gt_lines:
            print(f"⚠️ No GT found for {img_name}")
            continue
        
        boxes = detection['boxes']
        aligned_boxes = []
        used_gt_indices = set()
        
        # Match mỗi box với GT line
        for box in boxes:
            ocr_text = box['ocr_text']
            best_match = None
            best_sim = 0
            best_idx = -1
            
            # Tìm GT line giống nhất
            for i, gt_line in enumerate(gt_lines):
                if i in used_gt_indices:
                    continue
                
                sim = calculate_similarity(ocr_text, gt_line)
                
                if sim > best_sim and sim >= Config.SIMILARITY_THRESHOLD:
                    best_sim = sim
                    best_match = gt_line
                    best_idx = i
            
            if best_match:
                used_gt_indices.add(best_idx)
                aligned_boxes.append({
                    'bbox': box['bbox'],
                    'ocr_text': ocr_text,
                    'ground_truth': best_match,
                    'similarity': best_sim,
                    'confidence': box['confidence'],
                    'status': 'matched'
                })
            else:
                # Không match được
                aligned_boxes.append({
                    'bbox': box['bbox'],
                    'ocr_text': ocr_text,
                    'ground_truth': ocr_text,  # Fallback
                    'similarity': 0,
                    'confidence': box['confidence'],
                    'status': 'no_match'
                })
        
        aligned_data[img_name] = {
            'image_path': detection['image_path'],
            'boxes': aligned_boxes
        }
    
    # Stats
    total = sum(len(d['boxes']) for d in aligned_data.values())
    matched = sum(
        sum(1 for b in d['boxes'] if b['status'] == 'matched')
        for d in aligned_data.values()
    )
    
    print(f"✅ Aligned {total} boxes")
    if total > 0:
        print(f"   Matched: {matched} ({matched/total*100:.1f}%)")
        print(f"   Unmatched: {total - matched}")
    else:
        print("   No boxes found to align!")
        print("   Possible reasons:")
        print("     - No text detected in images")
        print("     - OCR detection failed")
        print("     - Check if images are valid")
    
    # Save
    with open(Config.DETECTION_DIR / 'aligned.json', 'w', encoding='utf-8') as f:
        json.dump(aligned_data, f, ensure_ascii=False, indent=2)
    
    return aligned_data


# ===================== BƯỚC 4: CREATE DATASET =====================
def crop_box_from_image(image_path, bbox, output_path):
    """Crop vùng bbox từ ảnh"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    points = np.array(bbox, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points)
    
    # Padding
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2*padding)
    h = min(img.shape[0] - y, h + 2*padding)
    
    cropped = img[y:y+h, x:x+w]
    
    if cropped.shape[0] < 5 or cropped.shape[1] < 5:
        return False
    
    cv2.imwrite(output_path, cropped)
    return True


def create_dataset(aligned_data):
    """Tạo dataset từ aligned data"""
    
    print("\n📦 BƯỚC 4: Create Dataset")
    print("="*60)
    
    # Tạo thư mục
    for split in ['train', 'val', 'test']:
        (Config.DATASET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    
    # Chuẩn bị entries
    all_entries = []
    
    for img_name, data in aligned_data.items():
        image_path = data['image_path']
        
        for box in data['boxes']:
            # Chỉ lấy matched boxes
            if box['status'] != 'matched':
                continue
            
            # Filter text quá ngắn
            if len(box['ground_truth'].strip()) < 2:
                continue
            
            all_entries.append({
                'image_path': image_path,
                'bbox': box['bbox'],
                'text': box['ground_truth']
            })
    
    # Shuffle và split
    random.shuffle(all_entries)
    
    total = len(all_entries)
    train_size = int(total * Config.TRAIN_RATIO)
    val_size = int(total * Config.VAL_RATIO)
    
    splits = {
        'train': all_entries[:train_size],
        'val': all_entries[train_size:train_size + val_size],
        'test': all_entries[train_size + val_size:]
    }
    
    # Crop và tạo labels
    for split_name, entries in splits.items():
        print(f"\nCreating {split_name} set...")
        
        labels = []
        success = 0
        
        for i, entry in enumerate(tqdm(entries, desc=f"  {split_name}")):
            # Crop
            img_name = f'{split_name}_{i:06d}.png'
            img_path = Config.DATASET_DIR / split_name / 'images' / img_name
            
            if crop_box_from_image(entry['image_path'], entry['bbox'], str(img_path)):
                labels.append(f"images/{img_name}\t{entry['text']}\n")
                success += 1
        
        # Save labels.txt
        labels_file = Config.DATASET_DIR / split_name / 'labels.txt'
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.writelines(labels)
        
        print(f"  ✅ {split_name}: {success} samples")
    
    print(f"\n✅ Dataset created at: {Config.DATASET_DIR}")
    print(f"   Train: {len(splits['train'])} samples")
    print(f"   Val:   {len(splits['val'])} samples")
    print(f"   Test:  {len(splits['test'])} samples")


# ===================== MAIN =====================
def main():
    """Chạy toàn bộ pipeline"""
    
    print("\n" + "="*60)
    print("🚀 OCR DATASET CREATION PIPELINE")
    print("="*60)
    
    # Tạo thư mục
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra input
    if not Config.RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy: {Config.RAW_IMAGES_DIR}")
    
    if not Config.RAW_GT_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy: {Config.RAW_GT_DIR}")
    
    print("\n📁 Input:")
    print(f"   Images: {Config.RAW_IMAGES_DIR}")
    print(f"   GT TXT: {Config.RAW_GT_DIR}")
    print(f"\n📁 Output: {Config.OUTPUT_DIR}")
    
    if Config.TEST_MODE:
        print(f"\n⚠️ TEST MODE: Processing only {Config.TEST_PAGES} pages")
    
    # Pipeline
    try:
        # 1. Preprocess
        processed_images = preprocess_all_images()
        
        # 2. Detect bbox
        detections = detect_boxes_paddleocr(processed_images)
        
        # 3. Align với TXT
        aligned_data = align_boxes_with_txt(detections)
        
        # 4. Create dataset
        create_dataset(aligned_data)
        
        print("\n" + "="*60)
        print("🎉 HOÀN THÀNH!")
        print("="*60)
        print(f"\n📁 Dataset tại: {Config.DATASET_DIR}")
        print("\n👉 Bước tiếp theo:")
        print("   1. Kiểm tra visualization trong 'detection/'")
        print("   2. Kiểm tra samples trong 'dataset/'")
        print("   3. Nếu OK, tắt TEST_MODE và chạy lại full")
        print("   4. Train PaddleOCR với dataset này!")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()