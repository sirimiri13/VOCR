"""
Bước 1: Tiền xử lý ảnh PDF
Xử lý ảnh trước khi đưa vào PaddleOCR để tạo bbox
"""

import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
import json
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Tiền xử lý ảnh từ PDF để chuẩn bị cho OCR"""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'dpi': 300,  # DPI cao cho chất lượng tốt
            'target_width': 2480,  # Chuẩn hóa width
            'denoise_strength': 10,
            'contrast_alpha': 1.3,  # Tăng contrast
            'brightness_beta': 10,
            'margin_threshold': 50,  # Pixel để detect margin
        }
    
    def pdf_to_images(self, pdf_path: str, output_dir: str, 
                      start_page: int = 1) -> List[str]:
        """
        Convert PDF sang ảnh
        Args:
            pdf_path: đường dẫn file PDF
            output_dir: thư mục lưu ảnh
            start_page: trang bắt đầu (mặc định 1, tính từ 0)
        Returns:
            List đường dẫn ảnh đã lưu
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting PDF: {pdf_path}")
        images = convert_from_path(
            pdf_path, 
            dpi=self.config['dpi'],
            first_page=start_page + 1  # pdf2image đếm từ 1
        )
        
        saved_paths = []
        for i, image in enumerate(images):
            # Tên file: book_name_page_XXX.jpg
            img_array = np.array(image)
            output_file = output_path / f"page_{start_page + i:04d}.jpg"
            cv2.imwrite(str(output_file), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            saved_paths.append(str(output_file))
            logger.info(f"Saved: {output_file}")
        
        return saved_paths
    
    def remove_margins(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Tự động phát hiện và xóa viền/margin
        Returns: ảnh đã crop và thông tin crop
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold để tìm vùng có nội dung
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image, {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # Tìm bounding box của tất cả nội dung
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        padding = 220

        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        cropped = image[y:y+h, x:x+w]
        crop_info = {'top': y, 'left': x, 'width': w, 'height': h}
        
        return cropped, crop_info
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Khử nhiễu ảnh"""
        return cv2.fastNlMeansDenoisingColored(
            image, None, 
            self.config['denoise_strength'], 
            self.config['denoise_strength'], 
            7, 21
        )
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Tăng độ tương phản"""
        return cv2.convertScaleAbs(
            image, 
            alpha=self.config['contrast_alpha'], 
            beta=self.config['brightness_beta']
        )
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Chỉnh góc nghiêng của ảnh"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Chỉ chỉnh nếu góc nghiêng đáng kể
        if abs(angle) < 0.5:
            return image
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Chuẩn hóa kích thước theo width"""
        h, w = image.shape[:2]
        target_w = self.config['target_width']
        
        if w != target_w:
            ratio = target_w / w
            target_h = int(h * ratio)
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            return resized
        
        return image
    
    def detect_and_remove_headers_footers(self, image: np.ndarray, 
                                          header_ratio: float = 0.08,
                                          footer_ratio: float = 0.08) -> np.ndarray:
        """
        Xóa header/footer dựa trên tỷ lệ chiều cao
        Args:
            header_ratio: tỷ lệ phần trên cần xóa (0.08 = 8%)
            footer_ratio: tỷ lệ phần dưới cần xóa
        """
        h, w = image.shape[:2]
        header_h = int(h * header_ratio)
        footer_h = int(h * footer_ratio)
        
        # Crop bỏ header và footer
        cropped = image[header_h:h-footer_h, :]
        return cropped
    
    def process_single_image(self, image_path: str, output_path: str = None,
                            remove_header_footer: bool = True) -> dict:
        """
        Xử lý một ảnh hoàn chỉnh
        Returns: dict chứa thông tin xử lý
        """
        logger.info(f"Processing: {image_path}")
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        original_shape = image.shape
        
        # Pipeline xử lý
        # 1. Chỉnh góc nghiêng
        # image = self.deskew_image(image)
        
        # 2. Xóa margin
        image, crop_info = self.remove_margins(image)
        
        # 3. Xóa header/footer nếu cần
        if remove_header_footer:
            image = self.detect_and_remove_headers_footers(image)
        
        # 4. Khử nhiễu
        image = self.denoise_image(image)
        
        # 5. Tăng contrast
        image = self.enhance_contrast(image)
        
        # 6. Chuẩn hóa kích thước
        image = self.resize_image(image)
        
        # Lưu ảnh
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            logger.info(f"Saved processed image: {output_path}")
        
        return {
            'original_shape': original_shape,
            'processed_shape': image.shape,
            'crop_info': crop_info,
            'output_path': output_path
        }
    
    def process_directory(self, input_dir: str, output_dir: str,
                         remove_header_footer: bool = True) -> List[dict]:
        """
        Xử lý tất cả ảnh trong thư mục
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        results = []
        for img_file in sorted(image_files):
            output_file = output_path / img_file.name
            try:
                result = self.process_single_image(
                    str(img_file), 
                    str(output_file),
                    remove_header_footer
                )
                result['input_file'] = str(img_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
        
        # Lưu metadata
        metadata_file = output_path / 'preprocessing_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {len(results)} images. Metadata saved to {metadata_file}")
        return results


