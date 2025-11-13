#!/usr/bin/env python3
"""
ocr_image_pdf.py
- Render PDF pages to images (PyMuPDF)
- Run PaddleOCR on each image page
- Save OCR result as JSON with bbox coords (x0,y0,x1,y1) in page coordinates (origin top-left)

Usage:
    python ocr_image_pdf.py input.pdf --pages 1 2 --out scanned_ocr.json --lang vi --use-gpu
Notes:
- Do NOT pass use_gpu into PaddleOCR constructor (some paddleocr versions don't accept it).
- To use GPU: install paddlepaddle-gpu wheel appropriate for your CUDA, then run with --use-gpu.
"""
import json
import argparse
import fitz  # pymupdf
from PIL import Image
import io
import paddle
from paddleocr import PaddleOCR
from tqdm import tqdm
import sys

def render_page_to_pil(page, zoom=2):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes()))
    return img, pix.width, pix.height, zoom

def setup_device(use_gpu: bool):
    if use_gpu:
        # Check if paddle was built with CUDA support
        try:
            has_cuda = paddle.is_compiled_with_cuda()
        except Exception:
            has_cuda = False
        if not has_cuda:
            print("WARNING: --use-gpu specified but installed paddle does NOT have CUDA support.", file=sys.stderr)
            print("Install paddlepaddle-gpu compatible with your CUDA or run without --use-gpu.", file=sys.stderr)
            # We will continue but set to CPU to avoid crash
            paddle.set_device('cpu')
        else:
            paddle.set_device('gpu')
            print("Using GPU for Paddle (paddle.set_device('gpu'))")
    else:
        paddle.set_device('cpu')
        print("Using CPU for Paddle (paddle.set_device('cpu'))")

def ocr_pdf_with_paddle(pdf_path, pages=None, lang='vi', use_gpu=False, zoom=2):
    # Setup device (do NOT pass use_gpu into PaddleOCR constructor)
    setup_device(use_gpu)

    # Initialize PaddleOCR (do not pass use_gpu param)
    # Note: 'lang' option depends on your paddleocr version; common values: 'en', 'ch', 'vi' (if supported)
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    except TypeError:
        # fallback if 'lang' not supported in this version
        ocr = PaddleOCR(use_angle_cls=True)

    doc = fitz.open(pdf_path)
    results = {"file": pdf_path, "pages": []}
    page_indices = pages if pages else list(range(1, len(doc)+1))
    for pnum in tqdm(page_indices, desc="Pages"):
        page = doc[pnum-1]
        img, w, h, zoom_used = render_page_to_pil(page, zoom=zoom)
        # PaddleOCR accepts numpy array or pil image
        ocr_res = ocr.ocr(img, cls=True)
        boxes = []
        # ocr_res: list of lists -> [ [[x1,y1],[x2,y2],...], (text, conf) ]
        for i, line in enumerate(ocr_res):
            try:
                bbox = line[0]  # list of points
                text = line[1][0]
                conf = float(line[1][1])
            except Exception:
                # If structure differs, skip
                continue
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            boxes.append({
                "id": f"p{pnum}_b{i}",
                "text": text,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "conf": conf
            })
        results["pages"].append({"page_num": pnum, "width": w, "height": h, "zoom": zoom_used, "boxes": boxes})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Input scanned PDF")
    parser.add_argument("--pages", nargs="+", type=int, help="Page numbers to process (1-based). If omitted, process all pages.")
    parser.add_argument("--out", default=None, help="Output JSON filename")
    parser.add_argument("--lang", default="vi", help="PaddleOCR language (e.g. 'vi' or 'en')")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for Paddle (requires paddlepaddle-gpu installed)")
    parser.add_argument("--zoom", type=int, default=2, help="Render zoom (scale) for higher resolution")
    args = parser.parse_args()

    res = ocr_pdf_with_paddle(args.pdf, pages=args.pages, lang=args.lang, use_gpu=args.use_gpu, zoom=args.zoom)
    out_file = args.out or args.pdf.replace('.pdf','_ocr.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Saved OCR JSON to", out_file)

if __name__ == "__main__":
    main()