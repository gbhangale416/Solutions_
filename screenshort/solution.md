You can pass a screenshot of the question instead of the question text. The code will use that screenshot to locate the matching question on the specified PDF page, then detect the checked checkbox and return its label.

For example:

```
result = get_checked_option(
    pdf_path=r"C:\PDF\Input.pdf",
    page_number=3,
    question_screenshot=r"C:\PDF\question.png"
)

print(json.dumps(result, indent=4))
```

## Updated end-to-end code

Keep the existing imports, OCR, rendering, checkbox detection, and label-reading functions from the previous script. Replace `locate_question()` and `get_checked_option()` with the following functions, and add the screenshot-matching helper.

```

import json
import re

import cv2
import numpy as np
import pymupdf
import pytesseract


# =========================================================
# CONFIGURATION
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DPI = 300

# Minimum screenshot matching score.
MATCH_THRESHOLD = 0.55

# Checkbox detection thresholds need validation on your PDF.
CHECKED_THRESHOLD = 0.10


# =========================================================
# RENDER PDF PAGE
# =========================================================

def render_page(page):
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(DPI / 72, DPI / 72),
        alpha=False
    )

    image = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# =========================================================
# LOCATE SCREENSHOT ON PDF PAGE
# =========================================================

def locate_screenshot(page_image, screenshot_path):
    template = cv2.imread(screenshot_path)

    if template is None:
        raise FileNotFoundError(
            f"Cannot read screenshot: {screenshot_path}"
        )

    page_gray = cv2.cvtColor(
        page_image, cv2.COLOR_BGR2GRAY
    )
    template_gray = cv2.cvtColor(
        template, cv2.COLOR_BGR2GRAY
    )

    page_h, page_w = page_gray.shape
    template_h, template_w = template_gray.shape

    best = None

    # Try different sizes because screenshot and PDF
    # rendering resolutions may differ.
    for scale in np.linspace(0.5, 2.0, 31):
        w = int(template_w * scale)
        h = int(template_h * scale)

        if w < 10 or h < 10 or w > page_w or h > page_h:
            continue

        resized = cv2.resize(
            template_gray,
            (w, h),
            interpolation=cv2.INTER_AREA
        )

        result = cv2.matchTemplate(
            page_gray,
            resized,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, location = cv2.minMaxLoc(result)

        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "x": int(location[0]),
                "y": int(location[1]),
                "w": w,
                "h": h
            }

    if best is None or best["score"] < MATCH_THRESHOLD:
        raise ValueError(
            "Screenshot could not be matched reliably. "
            f"Best score: {best['score'] if best else 'N/A'}"
        )

    return best


# =========================================================
# DETECT CHECKBOXES NEAR MATCHED QUESTION
# =========================================================

def detect_checkboxes(image, match):
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

    # Search the matched question row and the area to its right.
    row_y1 = max(0, match["y"] - 25)
    row_y2 = min(
        image.shape[0],
        match["y"] + match["h"] + 35
    )

    search_x1 = match["x"]
    search_x2 = image.shape[1]

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if not (10 <= w <= 65 and 10 <= h <= 65):
            continue

        if not 0.75 <= w / float(h) <= 1.25:
            continue

        if x < search_x1 or x > search_x2:
            continue

        center_y = y + h / 2

        if not row_y1 <= center_y <= row_y2:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour, 0.06 * perimeter, True
        )

        if not 4 <= len(approx) <= 8:
            continue

        # Measure dark pixels inside the box, excluding borders.
        mx = max(2, int(w * 0.25))
        my = max(2, int(h * 0.25))

        interior = binary[
            y + my:y + h - my,
            x + mx:x + w - mx
        ]

        if interior.size == 0:
            continue

        dark_ratio = float(
            np.count_nonzero(interior) / interior.size
        )

        candidates.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "dark_ratio": dark_ratio,
            "checked": bool(
                dark_ratio > CHECKED_THRESHOLD
            )
        })

    # Deduplicate nested contours.
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


# =========================================================
# READ LABEL TO THE RIGHT OF CHECKBOX
# =========================================================

def read_label(image, box, next_box=None):
    height, width = image.shape[:2]

    x1 = box["x"] + box["w"] + 1
    x2 = (
        next_box["x"]
        if next_box
        else min(width, x1 + 160)
    )

    padding = max(8, int(box["h"] * 0.6))

    y1 = max(0, box["y"] - padding)
    y2 = min(height, box["y"] + box["h"] + padding)

    if x2 <= x1:
        return ""

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

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

    text = re.sub(r"\s+", " ", text).strip()

    # Recognize common checkbox labels.
    match = re.match(
        r"^(Unknown|Yes|No|N/?A|True|False)\b",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return text.split()[0] if text else ""


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def get_checked_option(
    pdf_path,
    page_number,
    question_screenshot
):
    """
    page_number is 1-based.
    question_screenshot is a cropped screenshot of the question.
    """

    with pymupdf.open(pdf_path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError(
                f"Invalid page number {page_number}; "
                f"PDF has {len(pdf)} pages."
            )

        page = pdf[page_number - 1]
        image = render_page(page)

        match = locate_screenshot(
            image,
            question_screenshot
        )

        boxes = detect_checkboxes(image, match)

        if not boxes:
            return {
                "page_number": page_number,
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

            label = read_label(image, box, next_box)

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
            if option["checked"] and option["label"]
        ]

        if len(selected) == 1:
            checked_option = selected[0]["label"]
            status = "OK"
        elif len(selected) == 0:
            checked_option = None
            status = "Review required: no checked option detected"
        else:
            checked_option = None
            status = "Review required: multiple boxes appear checked"

        # Save a diagnostic image with detected boxes.
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
                (box["x"] + box["w"], box["y"] + box["h"]),
                color,
                2
            )

        cv2.imwrite(
            f"debug_page_{page_number}.png",
            debug
        )

        return {
            "page_number": int(page_number),
            "screenshot_match_score": float(match["score"]),
            "checked_option": checked_option,
            "status": status,
            "options": options
        }


# =========================================================
# EXECUTION
# =========================================================

if __name__ == "__main__":
    result = get_checked_option(
        pdf_path=r"C:\PDF\Input.pdf",
        page_number=3,
        question_screenshot=r"C:\PDF\question.png"
    )

    print(json.dumps(result, indent=4, ensure_ascii=False))


```

## Important notes

- The screenshot should be cropped around the question text, ideally including its checkbox options if possible.
- The screenshot and PDF page should have similar visual content. Template matching can fail if the screenshot is resized, heavily compressed, or captured at a different zoom level.
- The existing `detect_checkboxes()` function uses the matched screenshot area to find checkboxes. If your screenshot contains only the question text and excludes the options, the checkbox search region may need to extend farther to the right. For reliable results, the best next improvement is to use the screenshot to locate the question, then search the entire question row for its option checkboxes.
