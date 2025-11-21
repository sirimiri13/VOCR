import json


def similarity(text1, text2):
    """
    Tính độ giống nhau giữa 2 chuỗi (0-1)
    """
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def load_cache(cache_path):
    """
    Đọc file cache
    
    Returns:
        dict: {filename: [{transcription, points, difficult}]}
    """
    cache_data = {}
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t', 1)
            if len(parts) == 2:
                filename = parts[0]
                ocr_data = json.loads(parts[1])
                cache_data[filename] = ocr_data
    
    return cache_data


def load_ground_truth_txt(txt_path):
    """
    Đọc file TXT ground truth
    
    Returns:
        list: Danh sách các dòng text
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def align_gt_with_cache(ground_truth_lines, cache_boxes, similarity_threshold=0.3):
    """
    Dóng hàng ground truth với cache bbox bằng thuật toán similarity matching
    
    Args:
        ground_truth_lines: List text ground truth
        cache_boxes: List bbox từ cache
        similarity_threshold: Ngưỡng tương đồng tối thiểu (0-1)
    
    Returns:
        list: Cache mới với text ground truth đã được dóng hàng
    """
    new_cache = []
    used_gt_indices = set()
    
    num_gt = len(ground_truth_lines)
    num_boxes = len(cache_boxes)
    
    print(f"🔍 BẮT ĐẦU ALIGNMENT:")
    print(f"   - Ground truth: {num_gt} dòng")
    print(f"   - Cache bbox:   {num_boxes} boxes")
    print(f"   - Ngưỡng similarity: {similarity_threshold}\n")
    
    # Duyệt qua từng bbox trong cache
    for box_idx, cache_box in enumerate(cache_boxes):
        ocr_text = cache_box['transcription']
        best_match_idx = -1
        best_similarity = 0
        
        # Tìm ground truth giống nhất với OCR text
        for gt_idx, gt_text in enumerate(ground_truth_lines):
            if gt_idx in used_gt_indices:
                continue
            
            sim = similarity(ocr_text, gt_text)
            
            if sim > best_similarity:
                best_similarity = sim
                best_match_idx = gt_idx
        
        # Quyết định match hay giữ nguyên
        if best_match_idx != -1 and best_similarity >= similarity_threshold:
            matched_text = ground_truth_lines[best_match_idx]
            used_gt_indices.add(best_match_idx)
            
            new_cache.append({
                'transcription': matched_text,
                'points': cache_box['points'],
                'difficult': cache_box.get('difficult', False)
            })
            
            if best_similarity < 0.9:  # Chỉ hiển thị nếu không match hoàn toàn
                print(f"   [{box_idx+1}] Similarity={best_similarity:.2f}")
                print(f"       OCR: {ocr_text[:50]}...")
                print(f"       GT:  {matched_text[:50]}...\n")
        else:
            # Không tìm thấy match đủ tốt, giữ nguyên OCR text
            new_cache.append(cache_box)
            print(f"   [{box_idx+1}] ⚠️  Không tìm thấy match (best={best_similarity:.2f})")
            print(f"       Giữ nguyên OCR: {ocr_text[:50]}...\n")
    
    # Cảnh báo nếu có GT chưa được dùng
    unused_gt = num_gt - len(used_gt_indices)
    if unused_gt > 0:
        print(f"\n⚠️  Có {unused_gt} dòng ground truth CHƯA được match!")
    
    print(f"✅ Đã align {len(new_cache)} boxes")
    
    return new_cache


def save_new_cache(cache_data, output_path):
    """
    Lưu cache mới
    Format: filename\t[{transcription, points, difficult}]
    """
    import os
    
    # Tạo thư mục nếu chưa tồn tại
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Đã tạo thư mục: {output_dir}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for filename, boxes in cache_data.items():
            json_str = json.dumps(boxes, ensure_ascii=False)
            f.write(f"{filename}\t{json_str}\n")
    
    print(f"✅ Đã lưu cache mới: {output_path}")


def compare_old_new(old_boxes, new_boxes, num_show=5):
    """
    So sánh cache cũ vs mới
    """
    print("=" * 80)
    print("SO SÁNH TEXT CŨ (OCR) vs MỚI (GROUND TRUTH)")
    print("=" * 80)
    
    for i in range(min(num_show, len(old_boxes), len(new_boxes))):
        old_text = old_boxes[i]['transcription']
        new_text = new_boxes[i]['transcription']
        
        if old_text != new_text:
            print(f"\n[Dòng {i+1}]")
            print(f"  Cũ: {old_text}")
            print(f"  Mới: {new_text}")
            print(f"  BBox giữ nguyên: {new_boxes[i]['points'][0][:2]}...")


def process_single_page(cache_path, gt_txt_path, page_name, output_path):
    """
    Xử lý 1 trang: thay text trong cache bằng ground truth
    
    Args:
        cache_path: Đường dẫn file cache gốc
        gt_txt_path: Đường dẫn file TXT ground truth
        page_name: Tên trang trong cache (vd: 'test-png/page_130.png')
        output_path: Đường dẫn file cache mới
    """
    print(f"📁 Đang xử lý trang: {page_name}\n")
    
    # 1. Load cache
    print("1️⃣  Load cache gốc...")
    cache_data = load_cache(cache_path)
    
    if page_name not in cache_data:
        print(f"❌ Không tìm thấy {page_name} trong cache!")
        print(f"   Các trang có sẵn:")
        for name in cache_data.keys():
            print(f"   - {name}")
        return
    
    old_boxes = cache_data[page_name]
    print(f"   ✅ Tìm thấy {len(old_boxes)} boxes\n")
    
    # 2. Load ground truth
    print("2️⃣  Load ground truth TXT...")
    gt_lines = load_ground_truth_txt(gt_txt_path)
    print(f"   ✅ Đọc được {len(gt_lines)} dòng\n")
    
    # 3. Dóng hàng
    print("3️⃣  Dóng hàng ground truth với bbox...")
    new_boxes = align_gt_with_cache(gt_lines, old_boxes)
    print(f"   ✅ Đã map {len(new_boxes)} dòng\n")
    
    # 4. So sánh
    compare_old_new(old_boxes, new_boxes, num_show=5)
    
    # 5. Lưu cache mới
    print(f"\n4️⃣  Lưu cache mới...")
    new_cache_data = cache_data.copy()
    new_cache_data[page_name] = new_boxes
    save_new_cache(new_cache_data, output_path)
    
    print(f"\n🎉 Hoàn thành! Đã thay {len(new_boxes)} text boxes")


def process_multiple_pages(cache_path, gt_folder, output_path):
    """
    Xử lý nhiều trang cùng lúc
    
    Args:
        cache_path: File cache gốc
        gt_folder: Thư mục chứa các file TXT ground truth
        output_path: File cache mới
        
    Quy ước đặt tên file TXT:
        Cache: 'test-png/page_130.png'
        TXT:   'gt_folder/page_130.txt'
    """
    import os
    
    cache_data = load_cache(cache_path)
    new_cache_data = cache_data.copy()
    
    for page_name in cache_data.keys():
        # Tạo tên file TXT tương ứng
        base_name = os.path.splitext(os.path.basename(page_name))[0]
        txt_file = os.path.join(gt_folder, f"{base_name}.txt")
        
        if not os.path.exists(txt_file):
            print(f"⚠️  Bỏ qua {page_name}: không tìm thấy {txt_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Xử lý: {page_name}")
        print(f"{'='*80}")
        
        gt_lines = load_ground_truth_txt(txt_file)
        old_boxes = cache_data[page_name]
        new_boxes = align_gt_with_cache(gt_lines, old_boxes)
        new_cache_data[page_name] = new_boxes
        
        print(f"✅ Đã map {len(new_boxes)} dòng cho {page_name}")
    
    save_new_cache(new_cache_data, output_path)
    print(f"\n🎉 Hoàn thành tất cả!")


if __name__ == "__main__":
    # ===== CÁCH 1: Xử lý 1 trang =====
    # Đọc cache cũ, sửa text, ghi đè lại vào chính file cache đó
    cache_path = '/Users/huonglam/Library/Mobile Documents/com~apple~CloudDocs/Documents/Master/Tốt nghiệp/VOCR/data/Cache.cach'
    cache_new = 'data/algin/cache.cach'
    process_single_page(
        cache_path=cache_path,
        gt_txt_path='/Users/huonglam/Library/Mobile Documents/com~apple~CloudDocs/Documents/Master/Tốt nghiệp/VOCR/data/raw/daivietsuky-text.txt',
        page_name='test-png/daivietsuky-image_page_130.png',
        output_path=  cache_new
    )