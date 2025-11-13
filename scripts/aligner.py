#!/usr/bin/env python3
"""
aligner.py
- Align OCR boxes (from scanned PDF) to text lines (from text-PDF)
- Uses simple spatial filtering (y-center proximity / overlap) + textual similarity (rapidfuzz)
Usage:
    python aligner.py --ocr scanned_ocr.json --text text_layout.json --out aligned.json
"""
import json
import argparse
from rapidfuzz import fuzz
import csv

def y_center(box):
    return (box['y0'] + box['y1']) / 2

def overlap_ratio(boxA, boxB):
    # compute vertical overlap ratio relative to min height
    yA0, yA1 = boxA['y0'], boxA['y1']
    yB0, yB1 = boxB['y0'], boxB['y1']
    inter0 = max(yA0, yB0)
    inter1 = min(yA1, yB1)
    inter = max(0, inter1 - inter0)
    hmin = min(yA1 - yA0, yB1 - yB0)
    if hmin <= 0:
        return 0.0
    return inter / hmin

def align_page(ocr_page, text_page, y_tol=30, min_score=30):
    ocr_boxes = ocr_page.get('boxes', [])
    text_lines = text_page.get('lines', [])
    # For convenience ensure line ids
    for i, t in enumerate(text_lines):
        if 'line_id' not in t:
            t['line_id'] = f"l{i}"
    alignments = []
    for o in ocr_boxes:
        oc = y_center(o)
        # spatial candidates: those with center y within y_tol OR overlap ratio > 0.1
        candidates = []
        for t in text_lines:
            tc = (t['y0'] + t['y1']) / 2
            if abs(tc - oc) <= y_tol or overlap_ratio(o, t) > 0.1:
                candidates.append(t)
        # score by text similarity
        best = None
        for t in candidates:
            score = fuzz.token_set_ratio(o['text'], t['text'])
            if best is None or score > best['score']:
                best = {"line_id": t['line_id'], "text": t['text'], "score": score}
        if best is None or best['score'] < min_score:
            alignments.append({"ocr_id": o['id'], "ocr_text": o['text'], "match": None, "score": best['score'] if best else 0})
        else:
            alignments.append({"ocr_id": o['id'], "ocr_text": o['text'], "match": best['line_id'], "text": best['text'], "score": best['score']})
    return alignments

def align_documents(ocr_doc, text_doc, y_tol=30, min_score=30):
    # assume pages correspond by page_num (1-to-1). If pages mismatch, more heuristics needed.
    out = {"file_ocr": ocr_doc.get('file'), "file_text": text_doc.get('file'), "pages": []}
    text_pages_map = {p['page_num']: p for p in text_doc.get('pages', [])}
    for o_p in ocr_doc.get('pages', []):
        pnum = o_p['page_num']
        t_p = text_pages_map.get(pnum, {"page_num": pnum, "lines": []})
        aligns = align_page(o_p, t_p, y_tol=y_tol, min_score=min_score)
        out['pages'].append({"page_num": pnum, "alignments": aligns})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", required=True, help="OCR JSON from ocr_image_pdf.py")
    parser.add_argument("--text", required=True, help="Text-layout JSON from text_pdf_extract.py")
    parser.add_argument("--out", default="aligned.json", help="Output JSON")
    parser.add_argument("--csv", default=None, help="Optionally output CSV")
    parser.add_argument("--y-tol", type=int, default=30, help="Y tolerance in pixels")
    parser.add_argument("--min-score", type=int, default=30, help="Minimum similarity score")
    args = parser.parse_args()

    with open(args.ocr, 'r', encoding='utf-8') as f:
        ocr_doc = json.load(f)
    with open(args.text, 'r', encoding='utf-8') as f:
        text_doc = json.load(f)
    aligned = align_documents(ocr_doc, text_doc, y_tol=args.y_tol, min_score=args.min_score)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)
    print("Saved alignment JSON to", args.out)

    if args.csv:
        rows = []
        for p in aligned['pages']:
            for a in p['alignments']:
                rows.append({
                    "page": p['page_num'],
                    "ocr_id": a.get('ocr_id'),
                    "ocr_text": a.get('ocr_text'),
                    "match_line_id": a.get('match') or "",
                    "text_line": a.get('text') or "",
                    "score": a.get('score') or 0
                })
        with open(args.csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()) if rows else ["page","ocr_id","ocr_text","match_line_id","text_line","score"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print("Saved alignment CSV to", args.csv)

if __name__ == "__main__":
    main()