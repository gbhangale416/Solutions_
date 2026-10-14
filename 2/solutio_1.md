Below is an end-to-end Python solution where you provide only:

- The PDF file path
- The PDF page number
- The question text you want to find

The code locates the question, identifies its checkbox options, detects the checked option, and returns the selected label. It supports both text-based PDFs and scanned PDFs.

The approach uses PyMuPDF, OpenCV, and Tesseract OCR. It uses the question's position to associate checkboxes with the correct question, which is important when multiple questions appear on the same line.

## 1. Install dependencies

```
pip install pymupdf opencv-python pytesseract numpy
```

Install Tesseract OCR for Windows separately. The default executable path is usually:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## 2. Complete Python code

Save this as `pdf_checked_option.py`.

```
import re
import json
import fitz
import cv2
import numpy as np
import pytesseract

from difflib import SequenceMatcher
from pytesseract import Output


# ==========================================================
# CONFIGURATION
# ==========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DPI = 300


# ==========================================================
# TEXT UTILITIES
# ==========================================================

def normalize(text):
    """Normalize text for approximate question matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def similarity(a, b):
    return SequenceMatcher(
        None, normalize(a), normalize(b)
    ).ratio()


# ==========================================================
# RENDER PDF PAGE
# ==========================================================

def render_page(page):
    pix = page.get_pixmap(
        matrix=fitz.Matrix(DPI / 72, DPI / 72),
        alpha=False
    )

    image = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ==========================================================
# OCR WORDS AND POSITIONS
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
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
            "block": data["block_num"][i],
            "paragraph": data["par_num"][i],
            "line": data["line_num"][i]
        })

    return words


def group_lines(words):
    lines = {}

    for word in words:
        key = (
            word["block"],
            word["paragraph"],
            word["line"]
        )
        lines.setdefault(key, []).append(word)

    result = []

    for items in lines.values():
        items.sort(key=lambda item: item["x"])

        result.append({
            "words": items,
            "text": " ".join(w["text"] for w in items),
            "x1": min(w["x"] for w in items),
            "x2": max(w["x"] + w["w"] for w in items),
            "y1": min(w["y"] for w in items),
            "y2": max(w["y"] + w["h"] for w in items)
        })

    return sorted(result, key=lambda line: line["y1"])


# ==========================================================
# FIND QUESTION ON THE SPECIFIED PAGE
# ==========================================================

def locate_question(page, image, question):
    """
    First use native PDF text coordinates.
    If unavailable, locate the question using OCR.
    """

    scale = DPI / 72

    # Method 1: Native PDF text.
    matches = page.search_for(question, quads=False)

    if matches:
        rect = matches[0]

        return {
            "x1": int(rect.x0 * scale),
            "y1": int(rect.y0 * scale),
            "x2": int(rect.x1 * scale),
            "y2": int(rect.y1 * scale),
            "method": "pdf_text"
        }

    # Method 2: OCR line matching.
    words = get_ocr_words(image)
    lines = group_lines(words)

    best_line = None
    best_score = 0

    for line in lines:
        score = similarity(question, line["text"])

        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None or best_score < 0.60:
        raise ValueError(
            f"Could not confidently locate question: {question}"
        )

    return {
        "x1": best_line["x1"],
        "y1": best_line["y1"],
        "x2": best_line["x2"],
        "y2": best_line["y2"],
        "method": "ocr",
        "match_score": round(best_score, 3)
    }


# ==========================================================
# DETECT CHECKBOXES NEAR THE QUESTION
# ==========================================================

def detect_checkboxes(image, question_box):
    """
    Detect square checkbox outlines in the question's row.

    Detection uses contours and approximate dimensions.
    The thresholds may need calibration for a particular PDF.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    question_y = (question_box["y1"] + question_box["y2"]) // 2
    question_height = max(
        question_box["y2"] - question_box["y1"], 20
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Typical checkbox dimensions at 300 DPI.
        if not (10 <= w <= 65 and 10 <= h <= 65):
            continue

        ratio = w / float(h)

        if not 0.75 <= ratio <= 1.25:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour, 0.06 * perimeter, True
        )

        if not 4 <= len(approx) <= 8:
            continue

        center_y = y + h / 2

        # Checkboxes must be vertically aligned with the question.
        if abs(center_y - question_y) > max(
            25, question_height * 0.9
        ):
            continue

        # Only checkboxes to the right of the question start.
        if x < question_box["x1"]:
            continue

        # Exclude boxes too far below or above the question.
        if y > question_box["y2"] + question_height:
            continue

        # Remove the outline and inspect the checkbox interior.
        margin_x = max(2, int(w * 0.25))
        margin_y = max(2, int(h * 0.25))

        x1 = x + margin_x
        y1 = y + margin_y
        x2 = x + w - margin_x
        y2 = y + h - margin_y

        interior = binary[y1:y2, x1:x2]

        if interior.size == 0:
            continue

        dark_ratio = (
            np.count_nonzero(interior) / interior.size
        )

        # A starting heuristic. Validate using debug images.
        checked = dark_ratio > 0.10

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "checked": checked,
            "dark_ratio": round(dark_ratio, 3)
        })

    # Deduplicate nested contour detections.
    candidates.sort(key=lambda b: (b["x"], b["y"]))
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

    return sorted(unique, key=lambda b: b["x"])


# ==========================================================
# READ OPTION LABELS TO THE RIGHT OF CHECKBOXES
# ==========================================================

def read_option_label(image, box, next_box=None):
    """
    OCR the area immediately following a checkbox.
    The next checkbox limits the search area.
    """

    height, width = image.shape[:2]

    start_x = box["x"] + box["w"] + 1

    if next_box:
        end_x = next_box["x"]
    else:
        end_x = min(width, start_x + 160)

    # Limit the OCR crop to the checkbox's horizontal row.
    pad_y = max(8, int(box["h"] * 0.6))

    start_y = max(0, box["y"] - pad_y)
    end_y = min(
        height, box["y"] + box["h"] + pad_y
    )

    if end_x <= start_x:
        return ""

    crop = image[start_y:end_y, start_x:end_x]

    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(
        gray,
        config="--psm 7"
    )

    # Return the first detected option token.
    text = re.sub(r"[\r\n]+", " ", text).strip()
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)

    match = re.match(
        r"(Unknown|Yes|No|N/?A|True|False)\b",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1)

    # Fallback for other option names.
    return text.split()[0] if text else ""


# ==========================================================
# MAIN PUBLIC FUNCTION
# ==========================================================

def get_checked_option(
    pdf_path,
    page_number,
    question
):
    """
    page_number is 1-based.

    Returns the checked option and supporting diagnostic data.
    """

    with fitz.open(pdf_path) as pdf:
        if page_number < 1 or page_number > len(pdf):
            raise ValueError(
                f"Page {page_number} is invalid. "
                f"PDF contains {len(pdf)} pages."
            )

        page = pdf[page_number - 1]
        image = render_page(page)

        question_box = locate_question(
            page, image, question
        )

        boxes = detect_checkboxes(
            image, question_box
        )

        if not boxes:
            return {
                "page_number": page_number,
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
                image, box, next_box
            )

            options.append({
                "label": label,
                "checked": box["checked"],
                "dark_ratio": box["dark_ratio"],
                "position": {
                    "x": box["x"],
                    "y": box["y"]
                }
            })

        selected = [
            option for option in options
            if option["checked"]
        ]

        if len(selected) == 1 and selected[0]["label"]:
            checked_option = selected[0]["label"]
            status = "OK"

        elif len(selected) == 0:
            checked_option = None
            status = "Review required: no checked option detected"

        else:
            checked_option = None
            status = (
                "Review required: multiple checked boxes "
                "or missing option labels"
            )

        return {
            "page_number": page_number,
            "question": question,
            "checked_option": checked_option,
            "status": status,
            "options": options
        }


# ==========================================================
# EXECUTION EXAMPLE
# ==========================================================

if __name__ == "__main__":

    pdf_path = r"C:\PDF\Input.pdf"
    page_number = 3

    question = (
        "Do you experience excessive bloating after eating?"
    )

    result = get_checked_option(
        pdf_path=pdf_path,
        page_number=page_number,
        question=question
    )

    print(json.dumps(result, indent=4))

```

## 3. Example output

If the `No` checkbox is detected as checked, the intended output is:

```
{
    "page_number": 3,
    "question": "Do you experience excessive bloating after eating?",
    "checked_option": "No",
    "status": "OK",
    "options": [
        {
            "label": "Yes",
            "checked": false
        },
        {
            "label": "No",
            "checked": true
        },
        {
            "label": "Unknown",
            "checked": false
        }
    ]
}
```

The actual output also includes diagnostic positions and checkbox darkness measurements.

## 4. Important adjustments for your PDF

Your sample contains small checkboxes and multiple questions on a single line. Three parts may require tuning:

- Checkbox detection: the `dark_ratio > 0.10` threshold is only a starting point. A tick may occupy very little of the box interior, while printed artifacts can look like marks.
- Question location: native PDF text search works best when the question is selectable text. OCR matching is a fallback for scanned pages.
- Option labels: the example recognizes common labels such as `Yes`, `No`, and `Unknown`. If your PDF has other labels, adjust the regular expression in `read_option_label()`.

For production use, validate the detected checkbox state against a cropped image of each checkbox and return a review status when confidence is insufficient. The code above is a starting implementation and should be calibrated against your actual PDF before relying on its results.
