# VOCR - Vietnamese OCR Pipeline

Pipeline xử lý OCR tiếng Việt sử dụng PaddleOCR với khả năng chạy trên cả môi trường local và Google Colab.

### Mở Colab

1. Truy cập [Google Colab](https://colab.research.google.com)
2. Upload file `colab_pipeline.ipynb` hoặc
3. Clone repository trực tiếp trong Colab

### Chạy pipeline

- Chạy từng cell theo thứ tự trong `colab_pipeline.ipynb`
- Pipeline sẽ tự động cài đặt dependencies
- Sử dụng GPU miễn phí của Colab cho tốc độ xử lý nhanh hơn

## 💻 Chạy trên Local (macOS/Linux)

### Prerequisites

```bash
# Cài đặt poppler (cho pdf2image)
# macOS:
brew install poppler

# Ubuntu/Debian:
sudo apt-get install poppler-utils

# Windows:
# Download từ https://poppler.freedesktop.org/
```

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
# Import modules
from script.image_process import ImagePreprocessor
from script.text_extractor import PDFTextExtractor, BBoxGenerator

# Khởi tạo
preprocessor = ImagePreprocessor()
text_extractor = PDFTextExtractor()
bbox_generator = BBoxGenerator(lang='vi', use_gpu=False)

# Chạy pipeline
# ... (xem pipeline.ipynb để biết chi tiết)
```

## 📁 Cấu trúc thư mục

```
VOCR/
├── data/
│   ├── raw/
│   │   ├── image/          # PDF ảnh input
│   │   └── gt/             # PDF text ground truth
│   ├── convert/            # Ảnh được convert từ PDF
│   ├── processed/          # Ảnh sau preprocessing
│   └── bbox/               # Kết quả OCR với bounding boxes
├── script/
│   ├── image_process.py    # Xử lý ảnh
│   └── text_extractor.py   # Trích xuất text và OCR
├── pipeline.ipynb         # Notebook cho local
├── colab_pipeline.ipynb   # Notebook cho Google Colab
└── requirements.txt       # Dependencies
```

## ⚡ Lợi ích của Google Colab

1. **GPU miễn phí**: Tăng tốc PaddleOCR đáng kể
2. **Không cần cài đặt**: Môi trường đã được setup sẵn
3. **Ổn định**: Tránh các lỗi compatibility trên local
4. **Chia sẻ dễ dàng**: Có thể chia sẻ notebook với team

## 🛠️ Troubleshooting

### Local Issues

- **Kernel crash**: Chuyển sang sử dụng Colab
- **poppler error**: Cài đặt poppler theo hướng dẫn ở trên
- **Memory issues**: Giảm batch size hoặc dùng Colab

### Colab Issues

- **Runtime disconnect**: Save checkpoint thường xuyên
- **Storage limit**: Download kết quả và xóa file tạm

### PaddleOCR Issues

- **"string index out of range"**: 
  - Ảnh có thể không có text hoặc quá mờ
  - Thử resize ảnh lớn hơn
  - Kiểm tra định dạng ảnh (JPG/PNG)
  
- **GPU memory error**:
  ```python
  # Thử với CPU thay vì GPU
  bbox_generator = BBoxGenerator(lang='vi', use_gpu=False)
  ```

- **PaddleOCR initialization failed**:
  ```python
  # Thử với ngôn ngữ English trước
  bbox_generator = BBoxGenerator(lang='en', use_gpu=False)
  ```

### Cách fix lỗi "string index out of range" trong Colab:

**Cell sửa lỗi (chạy thay cho Cell 9):**

```python
# Test OCR với error handling tốt hơn
import os
from google.colab import files
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Tìm ảnh đầu tiên để test
processed_dir = "data/processed/image"
if not os.path.exists(processed_dir):
    print(f"❌ Thư mục {processed_dir} không tồn tại")
else:
    image_files = [f for f in os.listdir(processed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        test_image = os.path.join(processed_dir, image_files[0])
        print(f"🔍 Test OCR với ảnh: {test_image}")
        
        # Kiểm tra ảnh trước khi OCR
        try:
            img_check = cv2.imread(test_image)
            if img_check is None:
                print(f"❌ Không thể đọc ảnh: {test_image}")
            else:
                print(f"✅ Ảnh hợp lệ: {img_check.shape}")
                
                # OCR với error handling
                try:
                    text_regions = bbox_generator.detect_text_regions(test_image)
                    
                    if not text_regions:
                        print("⚠️ Không detect được text nào")
                        print("💡 Thử với ảnh khác hoặc kiểm tra chất lượng ảnh")
                    else:
                        print(f"✅ Phát hiện {len(text_regions)} vùng text!")
                        
                        # Hiển thị ảnh
                        img = Image.open(test_image)
                        plt.figure(figsize=(10, 8))
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(f"Ảnh test: {image_files[0]}")
                        plt.show()
                        
                        # Hiển thị text với error handling
                        print("\n📝 Text đã detect:")
                        for i, region in enumerate(text_regions[:5]):
                            try:
                                confidence = region.get('confidence', 0.0)
                                text = region.get('text', '[No text]')
                                print(f"  {i+1}. [{confidence:.2f}] {text}")
                            except Exception as e:
                                print(f"  {i+1}. [Error reading region: {e}]")
                        
                        if len(text_regions) > 5:
                            print(f"  ... và {len(text_regions) - 5} text khác")
                
                except Exception as ocr_error:
                    print(f"❌ Lỗi OCR: {ocr_error}")
                    print("💡 Các giải pháp:")
                    print("  - Thử khởi tạo lại bbox_generator")
                    print("  - Kiểm tra ảnh có text rõ ràng không")
                    print("  - Thử với ảnh khác")
                    
        except Exception as img_error:
            print(f"❌ Lỗi đọc ảnh: {img_error}")
    else:
        print("❌ Không tìm thấy ảnh đã processed để test")
        print("💡 Hãy chạy lại cell preprocessing ảnh")
```
