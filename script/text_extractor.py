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


class PDFTextExtractor(TextExtractor):
    """
    Alias cho TextExtractor - để tương thích với code cũ
    Có thể dùng cả PDFTextExtractor hoặc TextExtractor
    """
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


class BBoxGenerator:
    """Tạo bounding boxes bằng PaddleOCR - FIXED VERSION"""
    
    def __init__(self, lang='vi', use_angle_cls=True):
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
            show_log=False,
            # Thêm params để tăng khả năng detect và giảm lỗi
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
        )
        logger.info("PaddleOCR initialized successfully")
    
    def detect_text_regions(self, image_path: str) -> List[Dict]:
        """
        Detect text regions trong ảnh - FIXED VERSION với full error handling
        Returns:
            List of dict containing bbox coordinates and recognized text
        """
        logger.info(f"Detecting text in: {image_path}")
        
        # Validate image file
        if not Path(image_path).exists():
            logger.error(f"Image file not found: {image_path}")
            return []
        
        # Check if image is readable
        test_img = cv2.imread(image_path)
        if test_img is None:
            logger.error(f"Cannot read image: {image_path}")
            return []
        
        try:
            result = self.ocr.ocr(image_path)
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return []
        
        # VALIDATION CHAIN - kiểm tra từng bước
        if result is None:
            logger.warning(f"OCR returned None for {image_path}")
            return []
        
        if not isinstance(result, list):
            logger.warning(f"OCR returned unexpected type: {type(result)}")
            return []
        
        if len(result) == 0:
            logger.warning(f"OCR returned empty list for {image_path}")
            return []
        
        if result[0] is None:
            logger.warning(f"No text detected in {image_path}")
            return []
        
        if not isinstance(result[0], list):
            logger.warning(f"OCR result[0] has unexpected type: {type(result[0])}")
            return []
        
        if len(result[0]) == 0:
            logger.warning(f"No text lines in {image_path}")
            return []
        
        # Parse results với error handling từng line
        text_regions = []
        skipped_count = 0
        
        for idx, line in enumerate(result[0]):
            try:
                # Validate line structure
                if line is None:
                    skipped_count += 1
                    continue
                
                if not isinstance(line, (list, tuple)):
                    logger.debug(f"Line {idx}: invalid type {type(line)}")
                    skipped_count += 1
                    continue
                
                if len(line) < 2:
                    logger.debug(f"Line {idx}: invalid length {len(line)}")
                    skipped_count += 1
                    continue
                
                bbox = line[0]
                text_info = line[1]
                
                # Validate bbox
                if bbox is None or not isinstance(bbox, (list, tuple)):
                    logger.debug(f"Line {idx}: invalid bbox")
                    skipped_count += 1
                    continue
                
                if len(bbox) != 4:
                    logger.debug(f"Line {idx}: bbox length != 4")
                    skipped_count += 1
                    continue
                
                # Validate text_info
                if text_info is None or not isinstance(text_info, (list, tuple)):
                    logger.debug(f"Line {idx}: invalid text_info")
                    skipped_count += 1
                    continue
                
                if len(text_info) < 1:
                    logger.debug(f"Line {idx}: text_info too short")
                    skipped_count += 1
                    continue
                
                # SAFE EXTRACTION
                text = ""
                confidence = 0.0
                
                # Extract text (index 0)
                if len(text_info) >= 1 and text_info[0] is not None:
                    text = str(text_info[0])
                
                # Extract confidence (index 1)
                if len(text_info) >= 2 and text_info[1] is not None:
                    try:
                        confidence = float(text_info[1])
                    except (ValueError, TypeError):
                        confidence = 0.0
                
                # Calculate rect safely
                try:
                    rect = self._bbox_to_rect(bbox)
                except Exception as e:
                    logger.debug(f"Line {idx}: failed to calculate rect: {e}")
                    rect = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                
                text_regions.append({
                    'id': idx,
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence,
                    'rect': rect
                })
                
            except Exception as e:
                logger.error(f"Error processing line {idx}: {e}")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid lines")
        
        logger.info(f"Detected {len(text_regions)} text regions")
        return text_regions
    
    def _bbox_to_rect(self, bbox: List[List[float]]) -> Dict[str, int]:
        """Convert bbox 4 điểm sang rectangle (x, y, width, height)"""
        try:
            x_coords = [float(point[0]) for point in bbox]
            y_coords = [float(point[1]) for point in bbox]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            return {
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            }
        except Exception as e:
            logger.error(f"Failed to convert bbox to rect: {e}")
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
    
    def visualize_bboxes(self, image_path: str, text_regions: List[Dict], 
                         output_path: str = None):
        """Vẽ bounding boxes lên ảnh để kiểm tra"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot read image for visualization: {image_path}")
                return None
            
            for region in text_regions:
                bbox = region['bbox']
                # Chuyển bbox sang integer
                points = [[int(x), int(y)] for x, y in bbox]
                
                # Vẽ polygon
                cv2.polylines(image, [np.array(points)], True, (0, 255, 0), 2)
                
                # Vẽ text label (nếu ảnh đủ lớn)
                text = region['text']
                if len(text) < 50:  # Chỉ hiển thị text ngắn
                    cv2.putText(
                        image, 
                        text[:20], 
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
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None
    
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
                # Tiếp tục với ảnh tiếp theo thay vì crash
                all_results[img_file.name] = []
        
        # Lưu tổng hợp
        summary_file = output_path / 'all_bboxes.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {len(all_results)} images. Results saved to {output_dir}")
        return all_results

