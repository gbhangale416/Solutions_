Here is the complete updated code. It uses PyMuPDF (`import pymupdf`) instead of `fitz`, fixes the NumPy Boolean JSON serialization error, and accepts the PDF path, page number, and question as parameters.

The code returns the checked option, its label, and a review status if the checkbox state is ambiguous.

## 1. Install dependencies

Run in your VS Code terminal:

```
python -m pip install --upgrade pymupdf opencv-python pytesseract numpy
```

Install Tesseract OCR for Windows if you have not already installed it.

## 2. End-to-end code

Save as `extract_checked_options.py`.

```

import re
import json
from difflib import SequenceMatcher

import pymupdf
import cv2
import numpy as np
import pytesseract

from pytesseract import Output


# ==========================================================
# CONFIGURATION
# ==========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DPI = 300

# Minimum OCR question-match score.
QUESTION_MATCH_THRESHOLD = 0.60

# Starting heuristic; calibrate against your PDF.
CHECKED_THRESHOLD = 0.10

SAVE_DEBUG_IMAGE = True


# ==========================================================
# TEXT UTILITIES
# ==========================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def similarity(text1, text2):
    return SequenceMatcher(
        None,
        normalize(text1),
        normalize(text2)
    ).ratio()


# ==========================================================
# RENDER PDF PAGE USING PYMUPDF
# ==========================================================

def render_page(page):
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(DPI / 72, DPI / 72),
        alpha=False
    )

    image = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ==========================================================
# OCR WORDS WITH POSITIONS
# ==========================================================

def get_ocr_words(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        output_type=Output.DICT,
        config="--psm 11"
    )

    words = []

    for i, value in enumerate(data["text"]):
        value = value.strip()

        if not value:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < 15:
            continue

        words.append({
            "text": value,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "w": int(data["width"][i]),
            "h": int(data["height"][i]),
            "block": int(data["block_num"][i]),
            "paragraph": int(data["par_num"][i]),
            "line": int(data["line_num"][i])
        })

    return words


def group_lines(words):
    grouped = {}

    for word in words:
        key = (
            word["block"],
            word["paragraph"],
            word["line"]
        )
        grouped.setdefault(key, []).append(word)

    lines = []

    for items in grouped.values():
        items.sort(key=lambda item: item["x"])

        lines.append({
            "words": items,
            "text": " ".join(w["text"] for w in items),
            "x1": min(w["x"] for w in items),
            "x2": max(w["x"] + w["w"] for w in items),
            "y1": min(w["y"] for w in items),
            "y2": max(w["y"] + w["h"] for w in items)
        })

    return sorted(lines, key=lambda line: line["y1"])


# ==========================================================
# LOCATE THE QUESTION
# ==========================================================

def locate_question(page, image, question):
    """
    Try native PDF text first, then OCR.

    Coordinates returned are in rendered-image pixels.
    """

    scale = DPI / 72

    matches = page.search_for(question)

    if matches:
        rect = matches[0]

        return {
            "x1": int(rect.x0 * scale),
            "y1": int(rect.y0 * scale),
            "x2": int(rect.x1 * scale),
            "y2": int(rect.y1 * scale),
            "method": "pdf_text"
        }

    words = get_ocr_words(image)
    lines = group_lines(words)

    best_line = None
    best_score = 0.0

    for line in lines:
        score = similarity(question, line["text"])

        if score > best_score:
            best_score = score
            best_line = line

    if (
        best_line is None
        or best_score < QUESTION_MATCH_THRESHOLD
    ):
        raise ValueError(
            f"Could not locate question confidently. "
            f"Best OCR match score: {best_score:.3f}. "
            f"Question: {question}"
        )

    return {
        "x1": best_line["x1"],
        "y1": best_line["y1"],
        "x2": best_line["x2"],
        "y2": best_line["y2"],
        "method": "ocr",
        "match_score": float(best_score)
    }


# ==========================================================
# DETECT CHECKBOXES
# ==========================================================

def detect_checkboxes(image, question_box):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    question_y = (
        question_box["y1"] + question_box["y2"]
    ) / 2

    question_height = max(
        question_box["y2"] - question_box["y1"],
        20
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Approximate checkbox dimensions at 300 DPI.
        if not (10 <= w <= 65 and 10 <= h <= 65):
            continue

        ratio = w / float(h)

        if not 0.75 <= ratio <= 1.25:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.06 * perimeter,
            True
        )

        if not 4 <= len(approx) <= 8:
            continue

        center_y = y + h / 2

        # Must be near the question's vertical position.
        if abs(center_y - question_y) > max(
            25, question_height * 0.9
        ):
            continue

        # Checkbox should be to the right of the question start.
        if x < question_box["x1"]:
            continue

        # Exclude checkboxes too far below the question.
        if y > question_box["y2"] + question_height:
            continue

        margin_x = max(2, int(w * 0.25))
        margin_y = max(2, int(h * 0.25))

        interior = binary[
            y + margin_y:y + h - margin_y,
            x + margin_x:x + w - margin_x
        ]

        if interior.size == 0:
            continue

        dark_ratio = (
            np.count_nonzero(interior) / interior.size
        )

        # Convert NumPy/Python numeric values explicitly.
        dark_ratio = float(dark_ratio)
        checked = bool(dark_ratio > CHECKED_THRESHOLD)

        candidates.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "checked": checked,
            "dark_ratio": dark_ratio
        })

    # Remove duplicate contour detections.
    candidates.sort(key=lambda box: (box["x"], box["y"]))

    unique = []

    for box in candidates:
        duplicate = any(
            abs(box["x"] - old["x"]) < 4
            and abs(box["y"] - old["y"]) < 4
            and abs(box["w"] - old["w"]) < 4
            and abs(box["h"] - old["h"]) < 4
            for old in unique
        )

        if not duplicate:
            unique.append(box)

    return sorted(unique, key=lambda box: box["x"])


# ==========================================================
# READ LABEL NEXT TO CHECKBOX
# ==========================================================

def read_option_label(image, box, next_box=None):
    height, width = image.shape[:2]

    start_x = box["x"] + box["w"] + 1

    if next_box:
        end_x = next_box["x"]
    else:
        end_x = min(width, start_x + 160)

    pad_y = max(8, int(box["h"] * 0.6))

    start_y = max(0, box["y"] - pad_y)
    end_y = min(
        height,
        box["y"] + box["h"] + pad_y
    )

    if end_x <= start_x:
        return ""

    crop = image[start_y:end_y, start_x:end_x]

    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale small labels for OCR.
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    text = pytesseract.image_to_string(
        gray,
        config="--psm 7"
    )

    text = re.sub(r"[\r\n]+", " ", text).strip()
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)

    match = re.match(
        r"(Unknown|Yes|No|N/?A|True|False)\b",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1)

    # Fallback for other labels.
    return text.split()[0] if text else ""


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def get_checked_option(pdf_path, page_number, question):
    """
    Parameters:
        pdf_path: Full path to the PDF.
        page_number: 1-based page number.
        question: Question text to locate.

    Returns:
        JSON-serializable Python dictionary.
    """

    with pymupdf.open(pdf_path) as pdf:

        if not 1 <= page_number <= len(pdf):
            raise ValueError(
                f"Invalid page {page_number}. "
                f"PDF has {len(pdf)} pages."
            )

        page = pdf[page_number - 1]
        image = render_page(page)

        question_box = locate_question(
            page,
            image,
            question
        )

        boxes = detect_checkboxes(
            image,
            question_box
        )

        if not boxes:
            return {
                "page_number": int(page_number),
                "question": question,
                "checked_option": None,
                "status": "Review required: no checkboxes detected",
                "options": []
            }

        options = []

        for index, box in enumerate(boxes):
            next_box = (
                boxes[index + 1]
                if index + 1 < len(boxes)
                else None
            )

            label = read_option_label(
                image,
                box,
                next_box
            )

            options.append({
                "label": str(label),
                "checked": bool(box["checked"]),
                "dark_ratio": float(box["dark_ratio"]),
                "position": {
                    "x": int(box["x"]),
                    "y": int(box["y"])
                }
            })

        selected = [
            option for option in options
            if option["checked"]
        ]

        if (
            len(selected) == 1
            and selected[0]["label"]
        ):
            checked_option = selected[0]["label"]
            status = "OK"

        elif len(selected) == 0:
            checked_option = None
            status = (
                "Review required: no checked option detected"
            )

        else:
            checked_option = None
            status = (
                "Review required: multiple checked boxes "
                "or missing option labels"
            )

        # Optional diagnostic image:
        if SAVE_DEBUG_IMAGE:
            debug = image.copy()

            for box in boxes:
                color = (
                    (0, 255, 0)
                    if box["checked"]
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    debug,
                    (box["x"], box["y"]),
                    (
                        box["x"] + box["w"],
                        box["y"] + box["h"]
                    ),
                    color,
                    2
                )

            cv2.imwrite(
                f"debug_page_{page_number}.png",
                debug
            )

        return {
            "page_number": int(page_number),
            "question": question,
            "checked_option": checked_option,
            "status": status,
            "question_match_method": question_box["method"],
            "options": options
        }


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    PDF_PATH = r"C:\Girisb\Pdf_extraction\Tests\Input.pdf"
    PAGE_NUMBER = 3

    QUESTION = (
        "Do you experience excessive bloating after eating?"
    )

    try:
        result = get_checked_option(
            pdf_path=PDF_PATH,
            page_number=PAGE_NUMBER,
            question=QUESTION
        )

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )
        )

    except Exception as exc:
        print(f"Extraction failed: {exc}")

```

## 3. Run the script

Update `PDF_PATH`, `PAGE_NUMBER`, and `QUESTION` at the bottom, then execute:

```
python extract_checked_options.py
```

The output will include `checked_option`, `status`, and diagnostic information about the detected checkboxes. The `bool()` and `float()` conversions prevent the JSON serialization error you encountered.

Important: checkbox detection is heuristic. For your form, validate the `debug_page_3.png` image and the extracted labels before using the result in an automated workflow. Green outlines indicate boxes classified as checked; red outlines indicate unchecked boxes. A wrong classification can still occur if the checkbox is tiny or the tick is faint.
