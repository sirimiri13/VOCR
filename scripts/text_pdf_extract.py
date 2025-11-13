#!/usr/bin/env python3
"""
text_pdf_extract.py
- Extract text blocks/words from PDF (text-based) with bbox using PyMuPDF (fitz)
- Save JSON with lines/blocks and bounding boxes
Usage:
    python text_pdf_extract.py text.pdf --pages 1 2 --out text_layout.json
"""
import json
import argparse
import fitz
from collections import defaultdict
from rapidfuzz import fuzz

def group_words_to_lines(words, y_tol=3):
    # words: list of (x0,y0,x1,y1, "word") from page.get_text("words")
    # Group words into lines by similar y coordinate (top)
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))  # sort by top, then left
    lines = []
    current_line = []
    current_y = None
    for w in words_sorted:
        x0,y0,x1,y1,txt = w
        if current_y is None:
            current_y = y0
            current_line = [w]
        elif abs(y0 - current_y) <= y_tol:
            current_line.append(w)
        else:
            # flush
            line_txt = " ".join([wi[4] for wi in current_line])
            lx0 = min(wi[0] for wi in current_line)
            ly0 = min(wi[1] for wi in current_line)
            lx1 = max(wi[2] for wi in current_line)
            ly1 = max(wi[3] for wi in current_line)
            lines.append({"line_id": f"l{len(lines)}","text": line_txt, "x0": lx0,"y0":ly0,"x1":lx1,"y1":ly1})
            current_line = [w]
            current_y = y0
    # flush last
    if current_line:
        line_txt = " ".join([wi[4] for wi in current_line])
        lx0 = min(wi[0] for wi in current_line)
        ly0 = min(wi[1] for wi in current_line)
        lx1 = max(wi[2] for wi in current_line)
        ly1 = max(wi[3] for wi in current_line)
        lines.append({"line_id": f"l{len(lines)}","text": line_txt, "x0": lx0,"y0":ly0,"x1":lx1,"y1":ly1})
    return lines

def extract_text_pdf(pdf_path, pages=None):
    doc = fitz.open(pdf_path)
    results = {"file": pdf_path, "pages": []}
    page_indices = pages if pages else list(range(1, len(doc)+1))
    for pnum in page_indices:
        page = doc[pnum-1]
        words = page.get_text("words")  # list of tuples (x0,y0,x1,y1, "word", block_no, line_no, word_no)
        # normalize to (x0,y0,x1,y1, text)
        words_norm = [(w[0], w[1], w[2], w[3], w[4]) for w in words]
        lines = group_words_to_lines(words_norm, y_tol=3)
        results["pages"].append({"page_num": pnum, "lines": lines})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Input text PDF")
    parser.add_argument("--pages", nargs="+", type=int, help="Pages to extract")
    parser.add_argument("--out", default=None, help="Output JSON")
    args = parser.parse_args()
    res = extract_text_pdf(args.pdf, pages=args.pages)
    out_file = args.out or args.pdf.replace('.pdf','_text.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Saved text-layout JSON to", out_file)

if __name__ == "__main__":
    main()