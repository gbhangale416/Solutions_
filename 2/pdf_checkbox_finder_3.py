#!/usr/bin/env python3
"""
pdf_checkbox_finder.py
----------------------
Given (1) a PDF form, (2) the page number, and (3) a screenshot of one question
from that page, report which checkbox option(s) are ticked.

Pipeline
  1. Render the page to an image (PyMuPDF).
  2. Multi-scale template matching finds where the screenshot sits on the page (OpenCV).
  3. Crop that region, detect checkboxes and decide checked / unchecked (OpenCV).
  4. Get the words (question text + option labels such as Yes / No / Unknown) from ONE of:
       pdf        - the PDF's own text layer via PyMuPDF   (no OCR, nothing extra to install)
       rapidocr   - pip install rapidocr-onnxruntime       (OCR, pip only)
       easyocr    - pip install easyocr                    (OCR, pip only, heavy: needs torch)
       tesseract  - pip install pytesseract + Tesseract engine (OCR)
     --text-source auto (default) uses `pdf` when the page has a text layer,
     otherwise the first installed OCR engine.

Install (minimum)
  pip install pymupdf opencv-python numpy

Usage
  python pdf_checkbox_finder.py --pdf form.pdf --page 23 --question question.png
  python pdf_checkbox_finder.py --pdf form.pdf --page 23 --question q.png --text-source rapidocr
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

try:
    import pymupdf as fitz  # new package name
except ImportError:
    import fitz  # older PyMuPDF


# --------------------------------------------------------------------------
# 1. PDF -> image
# --------------------------------------------------------------------------
def render_page(doc, page_no, dpi):
    pix = doc[page_no - 1].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------
# 2. Locate the question screenshot on the page
# --------------------------------------------------------------------------
def _match_at_scale(page, shot, s):
    t = cv2.resize(shot, None, fx=s, fy=s,
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    th, tw = t.shape[:2]
    if th < 15 or tw < 15 or th >= page.shape[0] or tw >= page.shape[1]:
        return None
    res = cv2.matchTemplate(page, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return (float(score), loc[0], loc[1], tw, th)


def locate(page_bgr, shot_bgr):
    """Coarse-to-fine multi-scale template match. Returns (score, x, y, w, h) in page pixels."""
    page = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    shot = cv2.cvtColor(shot_bgr, cv2.COLOR_BGR2GRAY)

    best, best_s = (-1.0, 0, 0, 0, 0), 1.0
    for s in np.geomspace(0.3, 4.0, 60):               # coarse pass
        m = _match_at_scale(page, shot, s)
        if m and m[0] > best[0]:
            best, best_s = m, s
    for s in np.linspace(best_s * 0.94, best_s * 1.06, 25):   # fine pass around the best scale
        m = _match_at_scale(page, shot, s)
        if m and m[0] > best[0]:
            best = m
    return best


# --------------------------------------------------------------------------
# 3. Checkbox detection
# --------------------------------------------------------------------------
def find_checkboxes(bw, min_px=14, max_px=140):
    """Return list of (x, y, w, h) for square, solid-outline boxes."""
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_px <= w <= max_px and min_px <= h <= max_px):
            continue
        if not (0.8 <= w / h <= 1.25):
            continue
        if cv2.contourArea(c) / float(w * h) < 0.80:
            continue
        boxes.append((x, y, w, h))

    # drop duplicates (outer + inner contour of the same border)
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    unique = []
    for b in boxes:
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        if all(abs(cx - (u[0] + u[2] / 2)) > b[2] / 2 or
               abs(cy - (u[1] + u[3] / 2)) > b[3] / 2 for u in unique):
            unique.append(b)
    return unique


def is_checked(bw, box, threshold=0.06):
    """Ink ratio inside the box (border excluded). Above threshold => ticked."""
    x, y, w, h = box
    m = max(2, int(0.2 * w))
    inner = bw[y + m:y + h - m, x + m:x + w - m]
    if inner.size == 0:
        return False, 0.0
    ratio = cv2.countNonZero(inner) / float(inner.size)
    return ratio > threshold, ratio


# --------------------------------------------------------------------------
# 4. Text sources.  Every source returns words as
#    {"text", "x", "y", "w", "h"} in the pixel space of the SCALED crop.
# --------------------------------------------------------------------------
def split_line_into_words(poly, text):
    """OCR engines that return whole lines: split a line box into words by character count."""
    pts = np.array(poly, dtype=float).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    text = text.strip()
    if not text:
        return []
    n = len(text)
    out, pos = [], 0
    for tok in text.split():
        start = text.index(tok, pos)
        pos = start + len(tok)
        out.append({"text": tok,
                    "x": x0 + (x1 - x0) * start / n,
                    "y": y0,
                    "w": (x1 - x0) * len(tok) / n,
                    "h": y1 - y0})
    return out


def words_from_pdf(doc, page_no, dpi, origin, scale, size):
    """Words from the PDF's own text layer, mapped into the scaled-crop pixel space."""
    f = dpi / 72.0
    ox, oy = origin
    W, H = size
    out = []
    for x0, y0, x1, y1, t, *_ in doc[page_no - 1].get_text("words"):
        X0, X1 = (x0 * f - ox) * scale, (x1 * f - ox) * scale
        Y0, Y1 = (y0 * f - oy) * scale, (y1 * f - oy) * scale
        cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
        if 0 <= cx <= W and 0 <= cy <= H:
            out.append({"text": t, "x": X0, "y": Y0, "w": X1 - X0, "h": Y1 - Y0})
    return out


def words_from_rapidocr(img_bgr):
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        from rapidocr import RapidOCR
    out = RapidOCR()(img_bgr)
    if isinstance(out, tuple):                       # rapidocr-onnxruntime
        items = [(b, t) for b, t, *_ in (out[0] or [])]
    else:                                            # newer rapidocr
        boxes = getattr(out, "boxes", None)
        txts = getattr(out, "txts", None)
        items = list(zip(boxes, txts)) if boxes is not None and txts is not None else []
    words = []
    for poly, text in items:
        words += split_line_into_words(poly, text)
    return words


def words_from_easyocr(img_bgr):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    words = []
    for poly, text, _conf in reader.readtext(img_bgr):
        words += split_line_into_words(poly, text)
    return words


def words_from_tesseract(img_bgr, cmd=None):
    import pytesseract
    candidates = [
        cmd, shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            pytesseract.pytesseract.tesseract_cmd = c
            break
    else:
        raise SystemExit("Tesseract engine not found. Install it, pass --tesseract-cmd, "
                         "or use --text-source pdf / rapidocr.")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    return [{"text": t.strip(), "x": d["left"][i], "y": d["top"][i],
             "w": d["width"][i], "h": d["height"][i]}
            for i, t in enumerate(d["text"]) if t.strip() and float(d["conf"][i]) > 0]


def module_available(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def choose_text_source(requested, doc, page_no):
    if requested != "auto":
        return requested
    page = doc[page_no - 1]
    if page.rotation == 0 and len(page.get_text("words")) > 20:
        return "pdf"
    for name, mod in (("rapidocr", "rapidocr_onnxruntime"), ("rapidocr", "rapidocr"),
                      ("easyocr", "easyocr"), ("tesseract", "pytesseract")):
        if module_available(mod):
            return name
    raise SystemExit("This page has no text layer and no OCR library is installed.\n"
                     "Install one:  pip install rapidocr-onnxruntime")


# --------------------------------------------------------------------------
# Layout logic: rows -> question groups -> options
# --------------------------------------------------------------------------
def same_line(word, cy, tol):
    return abs((word["y"] + word["h"] / 2) - cy) < tol


def group_rows(boxes):
    rows = []
    for b in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
        cy = b[1] + b[3] / 2
        for r in rows:
            if abs(r["cy"] - cy) < b[3]:
                r["boxes"].append(b)
                break
        else:
            rows.append({"cy": cy, "boxes": [b]})
    for r in rows:
        r["boxes"].sort(key=lambda b: b[0])
    return rows


def option_label(box, words):
    """First word to the right of the box on the same line."""
    x, y, w, h = box
    cy = y + h / 2
    cands = [wd for wd in words if wd["x"] >= x + w - 5 and same_line(wd, cy, h)]
    return min(cands, key=lambda wd: wd["x"]) if cands else None


def read_checkboxes(crop_bgr, get_words, scale=2, debug_out=None):
    img = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    words = get_words(img)
    boxes = find_checkboxes(bw)
    results = []

    for row in group_rows(boxes):
        row_boxes = row["boxes"]
        tol = row_boxes[0][3]
        prev_end_x = 0

        # a question group ends when extra words (another question) sit between boxes
        groups, current = [], []
        for i, b in enumerate(row_boxes):
            current.append(b)
            lab = option_label(b, words)
            lab_end = (lab["x"] + lab["w"]) if lab else (b[0] + b[2])
            nxt = row_boxes[i + 1] if i + 1 < len(row_boxes) else None
            if nxt is None:
                groups.append(current)
                break
            between = [wd for wd in words
                       if same_line(wd, row["cy"], tol)
                       and lab_end - 2 < wd["x"] < nxt[0] - 2]
            if between:
                groups.append(current)
                current = []

        for g in groups:
            first = g[0]
            q_words = sorted((wd for wd in words
                              if same_line(wd, row["cy"], tol)
                              and prev_end_x - 2 <= wd["x"] < first[0] - 2),
                             key=lambda wd: wd["x"])
            question = " ".join(wd["text"] for wd in q_words)

            options = []
            for b in g:
                checked, ratio = is_checked(bw, b)
                lab = option_label(b, words)
                options.append({"label": lab["text"] if lab else "?",
                                "checked": bool(checked),
                                "fill": round(ratio, 3)})
                cv2.rectangle(img, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]),
                              (0, 200, 0) if checked else (0, 0, 255), 2)
            last = option_label(g[-1], words)
            prev_end_x = (last["x"] + last["w"]) if last else (g[-1][0] + g[-1][2])

            results.append({"question": question, "options": options,
                            "selected": [o["label"] for o in options if o["checked"]]})

    if debug_out:
        cv2.imwrite(debug_out, img)
    return results


# --------------------------------------------------------------------------
# Optional: fillable-form fields (exact values)
# --------------------------------------------------------------------------
def read_widgets(doc, page_no):
    out = []
    try:
        for w in doc[page_no - 1].widgets() or []:
            if w.field_type_string == "CheckBox":
                v = str(w.field_value).lower() if w.field_value else "off"
                out.append({"field": w.field_name, "checked": v not in ("off", "false", "0", "")})
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def find_answer(pdf_path, question_path, page, dpi=200, min_score=0.5, pad=8,
                text_source="auto", tesseract_cmd=None, scale=2, debug_dir="."):
    shot = cv2.imread(question_path)
    if shot is None:
        raise FileNotFoundError(f"Cannot read question image: {question_path}")

    doc = fitz.open(pdf_path)
    if not (1 <= page <= len(doc)):
        raise ValueError(f"--page {page} is out of range (PDF has {len(doc)} pages)")

    page_img = render_page(doc, page, dpi)
    score, x, y, w, h = locate(page_img, shot)
    print(f"page {page}: match score {score:.3f}", file=sys.stderr)
    if score < min_score:
        raise RuntimeError(f"Question not found on page {page} (score {score:.2f}). "
                           "Check the page number, lower --min-score, or use a cleaner screenshot.")

    H, W = page_img.shape[:2]
    ox, oy = max(0, x - pad), max(0, y - pad)
    crop = page_img[oy:min(H, y + h + pad), ox:min(W, x + w + pad)]
    cv2.imwrite(f"{debug_dir}/matched_crop.png", crop)
    marked = page_img.copy()
    cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite(f"{debug_dir}/matched_page.png", marked)

    source = choose_text_source(text_source, doc, page)
    print(f"text source: {source}", file=sys.stderr)
    size = (crop.shape[1] * scale, crop.shape[0] * scale)

    if source == "pdf":
        get_words = lambda img: words_from_pdf(doc, page, dpi, (ox, oy), scale, size)
    elif source == "rapidocr":
        get_words = words_from_rapidocr
    elif source == "easyocr":
        get_words = words_from_easyocr
    else:
        get_words = lambda img: words_from_tesseract(img, tesseract_cmd)

    results = read_checkboxes(crop, get_words, scale=scale,
                              debug_out=f"{debug_dir}/debug_checkboxes.png")
    return {"page": page, "match_score": round(score, 3), "text_source": source,
            "results": results, "form_fields": read_widgets(doc, page)}


def main():
    ap = argparse.ArgumentParser(description="Read the checked options of one question in a PDF page.")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, required=True, help="1-based PDF page number, e.g. 23")
    ap.add_argument("--question", required=True, help="screenshot of the question")
    ap.add_argument("--text-source", default="auto",
                    choices=["auto", "pdf", "rapidocr", "easyocr", "tesseract"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tesseract-cmd", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    out = find_answer(a.pdf, a.question, a.page, a.dpi, a.min_score,
                      text_source=a.text_source, tesseract_cmd=a.tesseract_cmd,
                      debug_dir=a.out_dir)
    if a.json:
        print(json.dumps(out, indent=2))
        return

    print(f"\nFound on page {out['page']} (match score {out['match_score']}, text: {out['text_source']})\n")
    for r in out["results"]:
        print(f"Q: {r['question'] or '(question text not read)'}")
        for o in r["options"]:
            print(f"   {'[X]' if o['checked'] else '[ ]'} {o['label']:<10} (fill={o['fill']})")
        print(f"   -> Selected: {', '.join(r['selected']) or 'none'}\n")
    if out["form_fields"]:
        print("Fillable checkbox fields on this page:")
        for f in out["form_fields"]:
            print(f"   {f['field']}: {'checked' if f['checked'] else 'unchecked'}")
    print("Debug images: matched_crop.png, matched_page.png, debug_checkboxes.png")


if __name__ == "__main__":
    main()
