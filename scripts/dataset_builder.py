"""
PaddleOCR Vietnamese Dataset Builder
=====================================
Quy trình xây dựng dataset cho huấn luyện PaddleOCR tiếng Việt

Quy trình:
1. Tiền xử lý ảnh (preprocessing)
2. Trích xuất text từ PDF text
3. Phát hiện text với PaddleOCR (bounding boxes)
4. Gióng hàng ảnh và text ground truth
5. Xuất dataset theo format PaddleOCR
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from pathlib import Path
import shutil
from typing import List, Dict, Tuple
import re

class DatasetBuilder:
    def __init__(self, work_dir: str = "./paddle_dataset"):
        """
        Khởi tạo Dataset Builder
        
        Args:
            work_dir: Thư mục làm việc chính
        """
        self.work_dir = Path(work_dir)
        self.setup_directories()
        
        # Khởi tạo PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='vi',
        )
        
    def setup_directories(self):
        """Tạo cấu trúc thư mục"""
        self.dirs = {
            'raw_images': self.work_dir / 'raw_images',
            'raw_pdfs': self.work_dir / 'raw_pdfs',
            'preprocessed': self.work_dir / 'preprocessed',
            'ground_truth': self.work_dir / 'ground_truth',
            'dataset': self.work_dir / 'dataset',
            'logs': self.work_dir / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # BƯỚC 1: TIỀN XỬ LÝ ẢNH
    # ============================================================
    
    def preprocess_image(self, image_path: str, output_path: str = None) -> str:
        """
        Tiền xử lý ảnh: chỉnh góc, khử nhiễu, tăng độ tương phản
        
        Args:
            image_path: Đường dẫn ảnh đầu vào
            output_path: Đường dẫn ảnh đầu ra
            
        Returns:
            Đường dẫn ảnh đã xử lý
        """
        print(f"Processing image: {image_path}")
        
        # Đọc ảnh
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # 1. Chuyển sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Khử nhiễu (denoise)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 3. Tăng độ tương phản (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 4. Làm rõ chữ (sharpening)
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 5. Adaptive thresholding (tùy chọn)
        # binary = cv2.adaptiveThreshold(sharpened, 255, 
        #                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY, 11, 2)
        
        # 6. Deskew (chỉnh góc nghiêng)
        coords = np.column_stack(np.where(sharpened > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:  # Chỉ xoay nếu góc > 0.5 độ
                (h, w) = sharpened.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                sharpened = cv2.warpAffine(sharpened, M, (w, h),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=255)
        
        # 7. Resize nếu quá lớn
        max_size = 2000
        h, w = sharpened.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            sharpened = cv2.resize(sharpened, (new_w, new_h), 
                                  interpolation=cv2.INTER_AREA)
        
        # 8. Xóa viền trắng
        sharpened = self._remove_borders(sharpened)
        
        # Lưu ảnh
        if output_path is None:
            output_path = self.dirs['preprocessed'] / Path(image_path).name
        
        cv2.imwrite(str(output_path), sharpened)
        print(f"Saved preprocessed image: {output_path}")
        
        return str(output_path)
    
    def _remove_borders(self, img: np.ndarray, threshold: int = 250) -> np.ndarray:
        """Xóa viền trắng xung quanh ảnh"""
        # Tìm vùng có nội dung
        mask = img < threshold
        coords = np.argwhere(mask)
        
        if len(coords) == 0:
            return img
        
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        # Thêm padding nhỏ
        padding = 10
        y0 = max(0, y0 - padding)
        x0 = max(0, x0 - padding)
        y1 = min(img.shape[0], y1 + padding)
        x1 = min(img.shape[1], x1 + padding)
        
        return img[y0:y1, x0:x1]
    
    # ============================================================
    # BƯỚC 2: TRÍCH XUẤT TEXT TỪ PDF
    # ============================================================
    
    def extract_text_from_pdf(self, pdf_path: str, page_num: int = None, output_path: str = None) -> str:
        """
        Trích xuất text từ PDF text (không phải PDF ảnh)
        
        Args:
            pdf_path: Đường dẫn file PDF
            page_num: Số trang cần trích xuất (None = tất cả trang)
            output_path: Đường dẫn file text đầu ra
            
        Returns:
            Đường dẫn file text
        """
        print(f"Extracting text from PDF: {pdf_path}")
        if page_num is not None:
            print(f"Extracting from page: {page_num}")
        
        pdf_file = fitz.open(pdf_path)
        
        all_text = []
        page_texts = []
        
        # Nếu chỉ định page_num, chỉ xử lý trang đó
        if page_num is not None:
            page_range = [page_num]
        else:
            page_range = range(len(pdf_file))
        
        for p_num in page_range:
            page = pdf_file.load_page(p_num)
            
            # Lấy text theo từng block
            blocks = page.get_text("dict")["blocks"]
            page_content = []
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        
                        line_text = line_text.strip()
                        if line_text:
                            page_content.append(line_text)
            
            page_texts.append({
                'page': p_num + 1,
                'lines': page_content
            })
            all_text.extend(page_content)
        
        # Lưu text
        if output_path is None:
            pdf_name = Path(pdf_path).stem
            output_path = self.dirs['ground_truth'] / f"{pdf_name}.txt"
            json_path = self.dirs['ground_truth'] / f"{pdf_name}.json"
        else:
            output_path = Path(output_path)
            json_path = output_path.with_suffix('.json')
        
        # Lưu text thuần
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        # Lưu JSON có cấu trúc
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(page_texts, f, ensure_ascii=False, indent=2)
        
        print(f"Extracted {len(all_text)} lines to: {output_path}")
        
        return str(output_path)
    
    # ============================================================
    # BƯỚC 3: PHÁT HIỆN TEXT VỚI PADDLEOCR
    # ============================================================
    
    def detect_text_regions(self, image_path: str) -> List[Dict]:
        """
        Phát hiện vùng text trong ảnh bằng PaddleOCR
        
        Args:
            image_path: Đường dẫn ảnh
            
        Returns:
            List các bounding box và text nhận dạng được
        """
        print(f"Detecting text regions in: {image_path}")
        
        # Chạy OCR
        try:
            result = self.ocr.ocr(image_path)
            
            print(f"OCR result type: {type(result)}")
            print(f"OCR result length: {len(result) if result else 0}")
            print(f"OCR result[0] type: {type(result[0]) if result and len(result) > 0 else 'N/A'}")
            
            if result is None or len(result) == 0:
                print("No text detected")
                return []
            
            # Kiểm tra cấu trúc kết quả
            if result[0] is None:
                print("No text detected in image")
                return []
            
            detections = []
            for i, line in enumerate(result[0]):  # result[0] vì chỉ có 1 ảnh
                try:
                    print(f"Processing line {i}: {type(line)} - {line}")
                    
                    # Kiểm tra kiểu dữ liệu của line
                    if line is None:
                        print(f"Line {i} is None, skipping")
                        continue
                        
                    # Nếu line là string, bỏ qua
                    if isinstance(line, str):
                        print(f"Line {i} is string: {line}")
                        continue
                        
                    # Nếu line không phải là list/tuple hoặc không đủ phần tử
                    if not isinstance(line, (list, tuple)):
                        print(f"Line {i} is not list/tuple: {type(line)}")
                        continue
                        
                    if len(line) < 2:
                        print(f"Line {i} has insufficient elements: {len(line)}")
                        continue
                        
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]  # (text, confidence) hoặc chỉ text
                    
                    print(f"Line {i} - bbox: {bbox}, text_info: {text_info}")
                    
                    # Xử lý các format khác nhau của text_info
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                    elif isinstance(text_info, (list, tuple)) and len(text_info) == 1:
                        text = text_info[0]
                        confidence = 1.0  # Default confidence
                    elif isinstance(text_info, str):
                        text = text_info
                        confidence = 1.0  # Default confidence
                    else:
                        print(f"Unexpected text_info format: {text_info}")
                        continue
                    
                    detection = {
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    }
                    detections.append(detection)
                    print(f"Successfully processed line {i}")
                    
                except Exception as e:
                    print(f"Error processing line {i}: {e}")
                    print(f"Line content: {line}")
                    continue
                    
        except Exception as e:
            print(f"Error in OCR detection: {e}")
            return []
        
        print(f"Found {len(detections)} text regions")
        return detections
    
    # ============================================================
    # BƯỚC 4: GIÓNG HÀNG ẢNH VÀ TEXT GROUND TRUTH
    # ============================================================
    
    def align_ocr_with_groundtruth(self, 
                                   detections: List[Dict], 
                                   ground_truth_lines: List[str]) -> List[Dict]:
        """
        Gióng hàng kết quả OCR với ground truth text
        
        Args:
            detections: List các detection từ PaddleOCR
            ground_truth_lines: List các dòng text ground truth
            
        Returns:
            List các cặp (detection, ground_truth) đã gióng hàng
        """
        print("Aligning OCR results with ground truth...")
        
        aligned_data = []
        
        # Làm sạch ground truth
        gt_cleaned = [self._clean_text(line) for line in ground_truth_lines]
        
        for det in detections:
            ocr_text = self._clean_text(det['text'])
            
            # Tìm ground truth khớp nhất
            best_match = None
            best_score = 0
            
            for i, gt_line in enumerate(gt_cleaned):
                score = self._calculate_similarity(ocr_text, gt_line)
                if score > best_score:
                    best_score = score
                    best_match = i
            
            if best_match is not None and best_score > 0.6:  # Threshold
                aligned_data.append({
                    'bbox': det['bbox'],
                    'ocr_text': det['text'],
                    'ground_truth': ground_truth_lines[best_match],
                    'confidence': det['confidence'],
                    'match_score': best_score
                })
        
        print(f"Aligned {len(aligned_data)} text regions")
        return aligned_data
    
    def _clean_text(self, text: str) -> str:
        """Làm sạch text để so sánh"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Xóa dấu câu
        text = re.sub(r'\s+', ' ', text)  # Chuẩn hóa khoảng trắng
        return text.strip()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Tính độ tương đồng giữa 2 chuỗi (simple Levenshtein)"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap ratio
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    # ============================================================
    # BƯỚC 5: XUẤT DATASET
    # ============================================================
    
    def export_dataset(self, 
                      image_path: str,
                      aligned_data: List[Dict],
                      output_name: str = None):
        """
        Xuất dataset theo format PaddleOCR
        
        Format:
        - Mỗi dòng trong label file: image_path\t[bbox, text]
        """
        if output_name is None:
            output_name = Path(image_path).stem
        
        # Đọc ảnh gốc
        img = cv2.imread(image_path)
        
        # Tạo label file
        label_file = self.dirs['dataset'] / 'labels.txt'
        
        with open(label_file, 'a', encoding='utf-8') as f:
            for i, item in enumerate(aligned_data):
                # Crop ảnh theo bbox
                bbox = np.array(item['bbox'], dtype=np.int32)
                x_min = int(bbox[:, 0].min())
                y_min = int(bbox[:, 1].min())
                x_max = int(bbox[:, 0].max())
                y_max = int(bbox[:, 1].max())
                
                cropped = img[y_min:y_max, x_min:x_max]
                
                # Lưu ảnh crop
                crop_name = f"{output_name}_crop_{i}.jpg"
                crop_path = self.dirs['dataset'] / crop_name
                cv2.imwrite(str(crop_path), cropped)
                
                # Ghi label
                label_data = {
                    'transcription': item['ground_truth'],
                    'points': item['bbox']
                }
                
                f.write(f"{crop_name}\t{json.dumps(label_data, ensure_ascii=False)}\n")
        
        print(f"Exported dataset to: {self.dirs['dataset']}")
    
    # ============================================================
    # PIPELINE CHÍNH
    # ============================================================
    
    def process_document_pair(self, 
                             image_pdf_path: str,
                             text_pdf_path: str,
                             image_page_num: int = 0,
                             text_page_num: int = None):
        """
        Xử lý một cặp document: PDF ảnh + PDF text
        
        Args:
            image_pdf_path: PDF dạng ảnh
            text_pdf_path: PDF dạng text (ground truth)
            image_page_num: Số trang cần lấy ảnh (0 = trang đầu)
            text_page_num: Số trang cần lấy text (None = dùng image_page_num)
        """
        print(f"\n{'='*60}")
        print(f"Processing document pair: {Path(image_pdf_path).name}")
        print(f"Image page: {image_page_num}, Text page: {text_page_num if text_page_num is not None else image_page_num}")
        print(f"{'='*60}\n")
        
        # Nếu không chỉ định text_page_num, dùng image_page_num
        if text_page_num is None:
            text_page_num = image_page_num
        
        # 1. Extract ảnh từ PDF ảnh
        doc = fitz.open(image_pdf_path)
        page = doc.load_page(image_page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
        
        img_name = f"{Path(image_pdf_path).stem}_page{image_page_num}.png"
        img_path = self.dirs['raw_images'] / img_name
        pix.save(str(img_path))
        
        # 2. Tiền xử lý ảnh
        preprocessed_path = self.preprocess_image(str(img_path))
        
        # 3. Extract ground truth text từ trang cụ thể
        gt_path = self.extract_text_from_pdf(text_pdf_path, page_num=text_page_num)
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_lines = [line.strip() for line in f if line.strip()]
        
        # 4. Detect text với PaddleOCR
        detections = self.detect_text_regions(preprocessed_path)
        
        # 5. Align với ground truth
        aligned_data = self.align_ocr_with_groundtruth(detections, gt_lines)
        
        # 6. Export dataset
        self.export_dataset(preprocessed_path, aligned_data, 
                          output_name=f"{Path(image_pdf_path).stem}_img_p{image_page_num}_txt_p{text_page_num}")
        
        print(f"\n{'='*60}")
        print(f"Completed processing: {img_name}")
        print(f"Total aligned regions: {len(aligned_data)}")
        print(f"{'='*60}\n")


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Khởi tạo builder
    builder = DatasetBuilder(work_dir="./paddle_dataset")
    
    # Xử lý một cặp document
    # builder.process_document_pair(
    #     image_pdf_path="path/to/scanned.pdf",
    #     text_pdf_path="path/to/text_version.pdf",
    #     image_page_num=130,  # Lấy ảnh từ trang 131 (index 130)
    #     text_page_num=2      # Lấy text từ trang 3 (index 2)
    # )
    
    print("Dataset Builder initialized successfully!")
    print(f"Working directory: {builder.work_dir}")
    print("\nNext steps:")
    print("1. Place your PDF files in raw_pdfs/")
    print("2. Call builder.process_document_pair() for each document pair")
    print("3. Check dataset/ folder for output")