# ⚡ Hướng dẫn sửa lỗi "string index out of range" trong Google Colab

## 🚨 Lỗi phổ biến và cách khắc phục

### Lỗi: "string index out of range"

**Nguyên nhân:**
- PaddleOCR không detect được text trong ảnh
- Ảnh quá mờ, độ phân giải thấp
- Format kết quả OCR không đúng

### 💡 Cell cải thiện để test OCR (thay thế Cell 9)

```python
# Test OCR với error handling và fallback strategies
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

def safe_ocr_test(bbox_generator, test_image):
    """Test OCR với error handling toàn diện"""
    
    print(f"🔍 Testing OCR với: {os.path.basename(test_image)}")
    
    # 1. Kiểm tra file
    if not os.path.exists(test_image):
        print(f"❌ File không tồn tại: {test_image}")
        return False
    
    # 2. Kiểm tra ảnh có đọc được không
    try:
        img_cv = cv2.imread(test_image)
        if img_cv is None:
            print(f"❌ Không thể đọc ảnh bằng OpenCV")
            return False
        
        print(f"✅ Kích thước ảnh: {img_cv.shape}")
        
        # 3. Kiểm tra chất lượng ảnh
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Tính độ rõ nét (variance of Laplacian)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"📊 Độ rõ nét: {variance:.2f} {'✅ OK' if variance > 50 else '⚠️ Mờ'}")
        
        # 4. Hiển thị ảnh trước khi OCR
        img_pil = Image.open(test_image)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_pil)
        plt.title(f"Ảnh test: {os.path.basename(test_image)}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"❌ Lỗi kiểm tra ảnh: {e}")
        return False
    
    # 5. Test OCR với nhiều strategy
    strategies = [
        ("Mặc định", {}),
        ("Không detect góc", {"cls": False}),
        ("Chi tiết hơn", {"det": True, "rec": True, "cls": True})
    ]
    
    for strategy_name, ocr_params in strategies:
        try:
            print(f"\n🔄 Thử strategy: {strategy_name}")
            
            # Gọi OCR
            if ocr_params:
                result = bbox_generator.ocr.ocr(test_image, **ocr_params)
            else:
                result = bbox_generator.ocr.ocr(test_image)
            
            # Kiểm tra kết quả chi tiết
            print(f"📋 Raw result type: {type(result)}")
            print(f"📋 Raw result length: {len(result) if result else 0}")
            
            if not result:
                print(f"⚠️ {strategy_name}: Không có kết quả")
                continue
                
            if not result[0]:
                print(f"⚠️ {strategy_name}: result[0] rỗng")
                continue
            
            # Parse kết quả an toàn
            text_regions = []
            for idx, line in enumerate(result[0]):
                try:
                    if not line or len(line) < 2:
                        continue
                    
                    bbox = line[0]
                    text_info = line[1]
                    
                    if not text_info or len(text_info) < 2:
                        continue
                    
                    text = str(text_info[0]) if text_info[0] else ""
                    confidence = float(text_info[1]) if text_info[1] else 0.0
                    
                    if text.strip() and confidence > 0.1:  # Threshold confidence
                        text_regions.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        
                except Exception as parse_error:
                    print(f"⚠️ Lỗi parse line {idx}: {parse_error}")
                    continue
            
            if text_regions:
                print(f"✅ {strategy_name}: Detect được {len(text_regions)} text!")
                
                # Hiển thị top results
                sorted_regions = sorted(text_regions, key=lambda x: x['confidence'], reverse=True)
                print(f"\n📝 Top {min(5, len(sorted_regions))} text có confidence cao nhất:")
                
                for i, region in enumerate(sorted_regions[:5]):
                    text_preview = region['text'][:50] + "..." if len(region['text']) > 50 else region['text']
                    print(f"  {i+1}. [{region['confidence']:.3f}] {text_preview}")
                
                return True
            else:
                print(f"⚠️ {strategy_name}: Không có text hợp lệ")
                
        except Exception as strategy_error:
            print(f"❌ Lỗi {strategy_name}: {strategy_error}")
            continue
    
    print(f"\n❌ Tất cả strategies đều thất bại")
    return False

# Chạy test
processed_dir = "data/processed/image"

if not os.path.exists(processed_dir):
    print(f"❌ Thư mục {processed_dir} không tồn tại")
    print("💡 Hãy chạy lại cell preprocessing ảnh trước")
else:
    image_files = [f for f in os.listdir(processed_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ Không tìm thấy ảnh trong thư mục processed")
        print("💡 Hãy chạy lại cell preprocessing ảnh")
    else:
        print(f"📁 Tìm thấy {len(image_files)} ảnh")
        
        # Test với ảnh đầu tiên
        test_image = os.path.join(processed_dir, image_files[0])
        success = safe_ocr_test(bbox_generator, test_image)
        
        if not success and len(image_files) > 1:
            print(f"\n🔄 Thử với ảnh thứ 2...")
            test_image2 = os.path.join(processed_dir, image_files[1])
            safe_ocr_test(bbox_generator, test_image2)
```

### 💡 Cell khởi tạo PaddleOCR an toàn hơn (thay thế Cell 8)

```python
# Khởi tạo PaddleOCR với fallback strategies
import os

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

def safe_paddleocr_init():
    """Khởi tạo PaddleOCR với nhiều fallback options"""
    
    init_configs = [
        {
            "name": "GPU + Vietnamese",
            "params": {"lang": 'vi', "use_gpu": True, "use_angle_cls": True, "show_log": False}
        },
        {
            "name": "CPU + Vietnamese", 
            "params": {"lang": 'vi', "use_gpu": False, "use_angle_cls": True, "show_log": False}
        },
        {
            "name": "CPU + English",
            "params": {"lang": 'en', "use_gpu": False, "use_angle_cls": True, "show_log": False}
        },
        {
            "name": "Basic CPU",
            "params": {"lang": 'en', "use_gpu": False, "use_angle_cls": False, "show_log": False}
        }
    ]
    
    for config in init_configs:
        try:
            print(f"🔄 Thử khởi tạo: {config['name']}")
            
            from script.text_extractor import BBoxGenerator
            bbox_generator = BBoxGenerator(**config['params'])
            
            print(f"✅ Thành công với: {config['name']}")
            
            # Test nhanh
            print("🧪 Test khởi tạo...")
            test_result = bbox_generator.ocr.ocr("data/processed/image/page_0130.jpg") if \
                         os.path.exists("data/processed/image/page_0130.jpg") else None
            
            if test_result is not None:
                print("✅ OCR engine hoạt động tốt!")
            else:
                print("⚠️ OCR engine khởi tạo nhưng chưa test được")
            
            return bbox_generator
            
        except Exception as e:
            print(f"❌ Thất bại {config['name']}: {str(e)[:100]}...")
            continue
    
    print("❌ Không thể khởi tạo PaddleOCR với bất kỳ config nào")
    return None

# Khởi tạo
bbox_generator = safe_paddleocr_init()

if bbox_generator:
    print("\n🎉 PaddleOCR sẵn sàng!")
else:
    print("\n💡 Các giải pháp thay thế:")
    print("  1. Restart runtime và chạy lại")
    print("  2. Kiểm tra kết nối mạng") 
    print("  3. Thử runtime khác (GPU/CPU)")
```

## 🔧 Các lỗi khác và cách khắc phục

### Lỗi: "CUDA out of memory"
```python
# Chuyển sang CPU
bbox_generator = BBoxGenerator(lang='vi', use_gpu=False)
```

### Lỗi: "Cannot download model"
```python
# Restart runtime và chạy lại, hoặc kiểm tra mạng
```

### Lỗi: "Image not found"
```python
# Kiểm tra đường dẫn file
import os
print("Files in processed dir:", os.listdir("data/processed/image"))
```

Sử dụng các cell cải thiện này thay cho các cell gốc để có trải nghiệm ổn định hơn!
