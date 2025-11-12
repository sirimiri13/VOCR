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
