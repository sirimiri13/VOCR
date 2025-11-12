"""
Bước 2 & 3: Trích xuất text từ PDF và tạo bounding boxes bằng PaddleOCR
"""

import PyPDF2
from paddleocr import PaddleOCR
import cv2
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import re
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextExtractor:
    """Trích xuất text từ PDF text"""
    
    def __init__(self):
        pass
    
    def extract_from_pdf(self, pdf_path: str, skip_pages: int = 2) -> Dict[int, str]:
        """
        Trích xuất text từ PDF, bỏ qua các trang đầu
        Args:
            pdf_path: đường dẫn PDF
            skip_pages: số trang cần bỏ qua (mặc định 2 trang bìa)
        Returns:
            Dict {page_number: text_content}
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        pages_text = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            logger.info(f"Total pages: {total_pages}, skipping first {skip_pages} pages")
            
            for page_num in range(skip_pages, total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Chuẩn hóa text: xóa whitespace dư thừa
                text = self.normalize_text(text)
                
                # Page number trong output (bắt đầu từ 0 sau khi skip)
                pages_text[page_num - skip_pages] = text
                
                logger.debug(f"Page {page_num - skip_pages}: {len(text)} characters")
        
        logger.info(f"Extracted {len(pages_text)} pages")
        return pages_text
    
    def normalize_text(self, text: str) -> str:
        """Chuẩn hóa text"""
        # Xóa multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Xóa space ở đầu/cuối dòng
        lines = [line.strip() for line in text.split('\n')]
        # Giữ lại cấu trúc đoạn văn
        text = '\n'.join(lines)
        return text.strip()
    
    def save_to_file(self, pages_text: Dict[int, str], output_dir: str):
        """Lưu text theo từng trang"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for page_num, text in pages_text.items():
            output_file = output_path / f"page_{page_num:04d}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.debug(f"Saved: {output_file}")
        
        logger.info(f"Saved {len(pages_text)} text files to {output_dir}")


# Alias for backward compatibility
PDFTextExtractor = TextExtractor


class BBoxGenerator:
    """Tạo bounding boxes bằng PaddleOCR"""
    
    def __init__(self, lang='vi', use_angle_cls=True, use_gpu=False):
        """
        Khởi tạo PaddleOCR
        Args:
            lang: ngôn ngữ ('vi' cho tiếng Việt)
            use_angle_cls: detect góc xoay text
            use_gpu: sử dụng GPU
        """
        logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            # use_gpu=use_gpu,
            # show_log=False
        )
        logger.info("PaddleOCR initialized successfully")
    
    def detect_text_regions(self, image_path: str) -> List[Dict]:
        """
        Detect text regions trong ảnh
        Returns:
            List of dict containing bbox coordinates and recognized text
        """
        logger.info(f"Detecting text in: {image_path}")
        
        result = self.ocr.ocr(image_path)
        
        if not result or not result[0]:
            logger.warning(f"No text detected in {image_path}")
            return []
        
        text_regions = []
        for idx, line in enumerate(result[0]):
            bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_info = line[1]  # (text, confidence)
            
            text_regions.append({
                'id': idx,
                'bbox': bbox,
                'text': text_info[0],
                'confidence': text_info[1],
                # Tính toán bounding box rectangle (x, y, w, h)
                'rect': self._bbox_to_rect(bbox)
            })
        
        logger.info(f"Detected {len(text_regions)} text regions")
        return text_regions
    
    def _bbox_to_rect(self, bbox: List[List[float]]) -> Dict[str, int]:
        """Convert bbox 4 điểm sang rectangle (x, y, width, height)"""
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        return {
            'x': x_min,
            'y': y_min,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
    def visualize_bboxes(self, image_path: str, text_regions: List[Dict], 
                         output_path: str = None):
        """Vẽ bounding boxes lên ảnh để kiểm tra"""
        image = cv2.imread(image_path)
        
        for region in text_regions:
            bbox = region['bbox']
            # Chuyển bbox sang integer
            points = [[int(x), int(y)] for x, y in bbox]
            
            # Vẽ polygon
            cv2.polylines(image, [np.array(points)], True, (0, 255, 0), 2)
            
            # Vẽ text label (nếu ảnh đủ lớn)
            if len(region['text']) < 50:  # Chỉ hiển thị text ngắn
                cv2.putText(
                    image, 
                    region['text'][:20], 
                    (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1
                )
        
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Visualization saved to: {output_path}")
        
        return image
    
    def process_directory(self, image_dir: str, output_dir: str,
                         visualize: bool = False) -> Dict[str, List[Dict]]:
        """
        Xử lý tất cả ảnh trong thư mục
        Returns:
            Dict {image_filename: text_regions}
        """
        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if visualize:
            viz_path = output_path / 'visualizations'
            viz_path.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = sorted([f for f in image_path.iterdir() 
                            if f.suffix.lower() in image_extensions])
        
        all_results = {}
        
        for img_file in image_files:
            logger.info(f"Processing {img_file.name}...")
            
            try:
                # Detect text regions
                text_regions = self.detect_text_regions(str(img_file))
                all_results[img_file.name] = text_regions
                
                # Lưu kết quả JSON
                json_file = output_path / f"{img_file.stem}_bbox.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(text_regions, f, ensure_ascii=False, indent=2)
                
                # Visualization nếu cần
                if visualize and text_regions:
                    viz_file = viz_path / f"{img_file.stem}_viz.jpg"
                    self.visualize_bboxes(str(img_file), text_regions, str(viz_file))
                
            except Exception as e:
                logger.error(f"Error processing {img_file.name}: {e}")
        
        # Lưu tổng hợp
        summary_file = output_path / 'all_bboxes.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {len(all_results)} images. Results saved to {output_dir}")
        return all_results


# # Ví dụ sử dụng
# if __name__ == "__main__":
#     import numpy as np
    
#     # Bước 1: Trích xuất text từ PDF text
#     text_extractor = TextExtractor()
#     pages_text = text_extractor.extract_from_pdf(
#         pdf_path="data/raw/gt/book.pdf",
#         skip_pages=2  # Bỏ 2 trang bìa
#     )
    
#     # Lưu text
#     text_extractor.save_to_file(
#         pages_text,
#         output_dir="data/processed/gt"
#     )
    
#     # Bước 2: Tạo bounding boxes bằng PaddleOCR
#     bbox_generator = BBoxGenerator(lang='vi', use_gpu=False)
    
#     # Xử lý thư mục ảnh
#     bbox_results = bbox_generator.process_directory(
#         image_dir="data/processed/image",
#         output_dir="data/processed/bbox",
#         visualize=True  # Tạo ảnh visualization để kiểm tra
#     )
    
#     print(f"Đã xử lý {len(bbox_results)} ảnh")
#     print(f"Trích xuất {len(pages_text)} trang text")