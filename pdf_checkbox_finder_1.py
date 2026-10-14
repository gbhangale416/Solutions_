#!/usr/bin/env python3
"""
pdf_checkbox_finder.py
----------------------
Given (1) a PDF form and (2) a screenshot of one question from that form,
find the question in the PDF and report which checkbox option(s) are ticked.

Pipeline
  1. Render every PDF page to an image (PyMuPDF).
  2. Multi-scale template matching to locate the screenshot on a page (OpenCV).
  3. Crop that region, detect checkboxes, decide checked / unchecked (OpenCV).
  4. OCR (Tesseract) to read each option label (Yes / No / Unknown ...) and the question text.

Install
  pip install pymupdf opencv-python numpy pytesseract
  + the Tesseract engine:  sudo apt install tesseract-ocr   (Linux)
                           brew install tesseract           (macOS)
                           https://github.com/UB-Mannheim/tesseract/wiki  (Windows)

Usage
  python pdf_checkbox_finder.py --pdf form.pdf --page 23 --question question.png
  python pdf_checkbox_finder.py --pdf form.pdf --page 23 --question q.png --dpi 250 --json
"""

import argparse
import json
import sys

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract

# On Windows you may need:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --------------------------------------------------------------------------
# 1. PDF -> images
# --------------------------------------------------------------------------
def parse_pages(spec, total):
    """'3' -> [3]; '20-25' -> [20..25]; '1,4,7-9' -> [1,4,7,8,9]; None -> all."""
    if not spec:
        return list(range(1, total + 1))
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= total)


def render_page(doc, page_no, dpi):
    pix = doc[page_no - 1].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------
# 2. Locate the question screenshot on a page
# --------------------------------------------------------------------------
def locate(page_bgr, shot_bgr, scales):
    """Multi-scale template match. Returns (score, x, y, w, h) in page pixels."""
    page = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    shot = cv2.cvtColor(shot_bgr, cv2.COLOR_BGR2GRAY)
    best = (-1.0, 0, 0, 0, 0)
    for s in scales:
        t = cv2.resize(shot, None, fx=s, fy=s,
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        th, tw = t.shape[:2]
        if th < 15 or tw < 15:
            continue
        if th >= page.shape[0] or tw >= page.shape[1]:
            break
        res = cv2.matchTemplate(page, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (float(score), loc[0], loc[1], tw, th)
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
# OCR helpers
# --------------------------------------------------------------------------
def get_words(gray):
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t and float(d["conf"][i]) > 0:
            words.append({"text": t, "x": d["left"][i], "y": d["top"][i],
                          "w": d["width"][i], "h": d["height"][i]})
    return words


def same_line(word, cy, tol):
    return abs((word["y"] + word["h"] / 2) - cy) < tol


def group_rows(boxes):
    """Cluster boxes into rows by vertical centre; each row sorted left->right."""
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
    """The first word to the right of the box on the same line."""
    x, y, w, h = box
    cy = y + h / 2
    cands = [wd for wd in words if wd["x"] >= x + w - 5 and same_line(wd, cy, h)]
    return min(cands, key=lambda wd: wd["x"]) if cands else None


def read_checkboxes(crop_bgr, scale=2, debug_out=None):
    img = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    words = get_words(gray)
    boxes = find_checkboxes(bw)
    results = []

    for row in group_rows(boxes):
        row_boxes = row["boxes"]
        tol = row_boxes[0][3]
        prev_end_x = 0  # right edge of the previous question's last option label

        # split the row into question groups: a group ends when extra words
        # (another question) appear between one option and the next box
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
            q_words = [wd for wd in words
                       if same_line(wd, row["cy"], tol)
                       and prev_end_x - 2 <= wd["x"] < first[0] - 2]
            q_words.sort(key=lambda wd: wd["x"])
            question = " ".join(wd["text"] for wd in q_words)

            options = []
            for b in g:
                checked, ratio = is_checked(bw, b)
                lab = option_label(b, words)
                options.append({
                    "label": lab["text"] if lab else "?",
                    "checked": bool(checked),
                    "fill": round(ratio, 3),
                })
                cv2.rectangle(img, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]),
                              (0, 200, 0) if checked else (0, 0, 255), 2)
            last_lab = option_label(g[-1], words)
            prev_end_x = (last_lab["x"] + last_lab["w"]) if last_lab else (g[-1][0] + g[-1][2])

            results.append({
                "question": question,
                "options": options,
                "selected": [o["label"] for o in options if o["checked"]],
            })

    if debug_out:
        cv2.imwrite(debug_out, img)
    return results


# --------------------------------------------------------------------------
# Optional: fillable-form fields (exact, no image analysis needed)
# --------------------------------------------------------------------------
def read_widgets(doc, page_no):
    out = []
    try:
        for w in doc[page_no - 1].widgets() or []:
            if w.field_type_string == "CheckBox":
                out.append({"field": w.field_name, "checked": bool(w.field_value
                            and str(w.field_value).lower() not in ("off", "false", "0"))})
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def find_answer(pdf_path, question_path, page, dpi=200, min_score=0.5,
                pad=8, debug_dir="."):
    shot = cv2.imread(question_path)
    if shot is None:
        raise FileNotFoundError(f"Cannot read question image: {question_path}")

    doc = fitz.open(pdf_path)
    if not (1 <= page <= len(doc)):
        raise ValueError(f"--page {page} is out of range (PDF has {len(doc)} pages)")
    page_list = [page]  # only look on the page the user gave
    scales = np.geomspace(0.3, 4.0, 45)

    best = None  # (score, page_no, x, y, w, h, page_img)
    for p in page_list:
        page_img = render_page(doc, p, dpi)
        score, x, y, w, h = locate(page_img, shot, scales)
        print(f"  page {p:>3}: match score {score:.3f}", file=sys.stderr)
        if best is None or score > best[0]:
            best = (score, p, x, y, w, h, page_img)

    score, page_no, x, y, w, h, page_img = best
    if score < min_score:
        raise RuntimeError(
            f"Question not found (best score {score:.2f} on page {page_no}). "
            f"Check the page number, lower --min-score, or use a cleaner screenshot.")

    H, W = page_img.shape[:2]
    crop = page_img[max(0, y - pad):min(H, y + h + pad),
                    max(0, x - pad):min(W, x + w + pad)]
    cv2.imwrite(f"{debug_dir}/matched_crop.png", crop)

    # also mark the match on the full page for easy visual verification
    marked = page_img.copy()
    cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite(f"{debug_dir}/matched_page.png", marked)

    results = read_checkboxes(crop, debug_out=f"{debug_dir}/debug_checkboxes.png")
    return {
        "page": page_no,
        "match_score": round(score, 3),
        "results": results,
        "form_fields": read_widgets(doc, page_no),
    }


def main():
    ap = argparse.ArgumentParser(description="Find a question in a PDF and read its checked options.")
    ap.add_argument("--pdf", required=True, help="path to the PDF")
    ap.add_argument("--question", required=True, help="screenshot of the question")
    ap.add_argument("--page", type=int, required=True, help="1-based PDF page number to look on, e.g. 23")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--out-dir", default=".", help="where debug images are saved")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    a = ap.parse_args()

    out = find_answer(a.pdf, a.question, a.page, a.dpi, a.min_score, debug_dir=a.out_dir)

    if a.json:
        print(json.dumps(out, indent=2))
        return

    print(f"\nFound on page {out['page']} (match score {out['match_score']})\n")
    for r in out["results"]:
        print(f"Q: {r['question'] or '(question text not read)'}")
        for o in r["options"]:
            mark = "[X]" if o["checked"] else "[ ]"
            print(f"   {mark} {o['label']:<10} (fill={o['fill']})")
        print(f"   -> Selected: {', '.join(r['selected']) or 'none'}\n")
    if out["form_fields"]:
        print("Fillable checkbox fields on this page:")
        for f in out["form_fields"]:
            print(f"   {f['field']}: {'checked' if f['checked'] else 'unchecked'}")
    print("Debug images: matched_crop.png, matched_page.png, debug_checkboxes.png")


if __name__ == "__main__":
    main()
