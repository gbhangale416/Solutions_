Below is an end-to-end Python solution to extract each question and its checked checkbox option from a multi-page PDF and export the results to Excel.

Because your PDF contains questions, option labels, and small checkboxes, the script uses OCR plus image processing. It does not require you to share the full PDF.

## 1. Install the required libraries

Run this in your Windows command prompt:

```
pip install pymupdf opencv-python pytesseract pandas openpyxl
```

Install Tesseract OCR for Windows separately. After installation, note its executable path, usually:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## 2. Complete Python script

Save this as `extract_checked_options.py`.

The script renders each page, detects checkbox-like squares, estimates their checked state, reads text with OCR, and exports the question and selected option to Excel.

```

import re
import cv2
import fitz
import pytesseract
import pandas as pd
import numpy as np

from pathlib import Path
from pytesseract import Output

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PDF_PATH = r"C:\PDF\Input.pdf"
OUTPUT_EXCEL = r"C:\PDF\checked_answers.xlsx"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DPI = 300

# Set to True to save diagnostic page images.
SAVE_DEBUG_IMAGES = True

# --------------------------------------------------
# PDF PAGE TO IMAGE
# --------------------------------------------------

def render_page(page):
    pix = page.get_pixmap(
        matrix=fitz.Matrix(DPI / 72, DPI / 72),
        alpha=False
    )

    image = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# --------------------------------------------------
# DETECT CHECKBOXES
# --------------------------------------------------

def detect_checkboxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image.
    binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Detect square outlines.
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Adjust these dimensions if your checkboxes
        # are unusually small or large.
        if not (10 <= w <= 65 and 10 <= h <= 65):
            continue

        ratio = w / float(h)

        if not 0.75 <= ratio <= 1.25:
            continue

        area = cv2.contourArea(contour)

        if area < 40:
            continue

        # Check that the shape is approximately square.
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour, 0.08 * perimeter, True
        )

        if len(approx) < 4 or len(approx) > 8:
            continue

        # Estimate whether the checkbox interior contains
        # a tick, cross, or other dark marking.
        margin_x = max(2, int(w * 0.25))
        margin_y = max(2, int(h * 0.25))

        x1 = x + margin_x
        y1 = y + margin_y
        x2 = x + w - margin_x
        y2 = y + h - margin_y

        interior = binary[y1:y2, x1:x2]

        if interior.size == 0:
            continue

        dark_ratio = np.count_nonzero(interior) / interior.size

        # Tune this threshold using your PDF if necessary.
        checked = dark_ratio > 0.12

        boxes.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "checked": checked,
            "dark_ratio": round(dark_ratio, 3)
        })

    # Remove duplicate detections of the same checkbox.
    boxes.sort(key=lambda b: (b["y"], b["x"]))
    unique = []

    for box in boxes:
        duplicate = any(
            abs(box["x"] - old["x"]) < 4
            and abs(box["y"] - old["y"]) < 4
            and abs(box["w"] - old["w"]) < 4
            and abs(box["h"] - old["h"]) < 4
            for old in unique
        )

        if not duplicate:
            unique.append(box)

    return unique


# --------------------------------------------------
# OCR WORDS WITH THEIR POSITIONS
# --------------------------------------------------

def extract_words(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        output_type=Output.DICT,
        config="--psm 11"
    )

    words = []

    for i, text in enumerate(data["text"]):
        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < 15:
            continue

        words.append({
            "text": text,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
            "line": (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i]
            )
        })

    return words


# --------------------------------------------------
# GROUP WORDS INTO VISUAL LINES
# --------------------------------------------------

def group_words_by_line(words):
    lines = {}

    for word in words:
        key = word["line"]
        lines.setdefault(key, []).append(word)

    result = []

    for line_words in lines.values():
        line_words.sort(key=lambda w: w["x"])

        result.append({
            "y": min(w["y"] for w in line_words),
            "words": line_words,
            "text": " ".join(w["text"] for w in line_words)
        })

    return sorted(result, key=lambda line: line["y"])


# --------------------------------------------------
# MATCH CHECKBOXES TO TEXT
# --------------------------------------------------

def get_line_for_checkbox(box, lines):
    """
    Find the OCR line vertically aligned with a checkbox.
    """
    center_y = box["y"] + box["h"] / 2

    candidates = []

    for line in lines:
        words = line["words"]

        top = min(w["y"] for w in words)
        bottom = max(w["y"] + w["h"] for w in words)

        distance = 0 if top <= center_y <= bottom else min(
            abs(center_y - top),
            abs(center_y - bottom)
        )

        if distance <= max(15, box["h"]):
            candidates.append((distance, line))

    if not candidates:
        return None

    return min(candidates, key=lambda item: item[0])[1]


def get_option_after_checkbox(box, line):
    """
    In this PDF layout, the option label is assumed to
    appear immediately to the right of its checkbox.
    """
    if line is None:
        return None

    words = line["words"]
    box_center_y = box["y"] + box["h"] / 2

    candidates = []

    for word in words:
        word_center_y = word["y"] + word["h"] / 2

        if word["x"] < box["x"] + box["w"] - 2:
            continue

        vertical_distance = abs(word_center_y - box_center_y)

        if vertical_distance > max(15, box["h"]):
            continue

        horizontal_distance = word["x"] - (box["x"] + box["w"])

        candidates.append(
            (vertical_distance, horizontal_distance, word)
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))

    # Usually options are Yes, No, Unknown, etc.
    return candidates[0][2]["text"]


# --------------------------------------------------
# EXTRACT QUESTIONS AND SELECTED OPTIONS
# --------------------------------------------------

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_page(image, page_number):
    boxes = detect_checkboxes(image)
    words = extract_words(image)
    lines = group_words_by_line(words)

    if SAVE_DEBUG_IMAGES:
        debug = image.copy()

        for box in boxes:
            color = (0, 255, 0) if box["checked"] else (0, 0, 255)

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

    # Assign each checkbox to its OCR line.
    items = []

    for box in boxes:
        line = get_line_for_checkbox(box, lines)

        if line is None:
            continue

        option = get_option_after_checkbox(box, line)

        items.append({
            **box,
            "line": line,
            "option": option
        })

    items.sort(key=lambda item: (
        item["y"],
        item["x"]
    ))

    results = []

    # Group checkboxes on the same visual line.
    line_groups = {}

    for item in items:
        key = id(item["line"])
        line_groups.setdefault(key, []).append(item)

    for group in line_groups.values():
        group.sort(key=lambda item: item["x"])

        first_box_x = group[0]["x"]

        # The text before the first checkbox is treated
        # as the question.
        question_words = [
            word for word in group[0]["line"]["words"]
            if word["x"] < first_box_x
        ]

        question = clean_text(
            " ".join(word["text"] for word in question_words)
        )

        selected = [
            item["option"]
            for item in group
            if item["checked"] and item["option"]
        ]

        all_options = [
            item["option"]
            for item in group
            if item["option"]
        ]

        # Skip lines that do not look like question/option lines.
        if not question or not all_options:
            continue

        results.append({
            "Page": page_number,
            "Question": question,
            "Checked Option": ", ".join(selected),
            "All Options": ", ".join(all_options),
            "Status": (
                "No option checked" if not selected
                else "Multiple options checked" if len(selected) > 1
                else "OK"
            )
        })

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    pdf_file = Path(PDF_PATH)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    all_results = []

    with fitz.open(PDF_PATH) as pdf:
        for page_index, page in enumerate(pdf):
            print(
                f"Processing page {page_index + 1} "
                f"of {len(pdf)}..."
            )

            image = render_page(page)

            results = extract_page(
                image,
                page_index + 1
            )

            all_results.extend(results)

    if not all_results:
        print(
            "No question/option rows detected. "
            "Inspect the debug images and adjust detection."
        )
        return

    df = pd.DataFrame(all_results)

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Checked Answers"
        )

        worksheet = writer.sheets["Checked Answers"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column in worksheet.columns:
            max_length = max(
                len(str(cell.value or ""))
                for cell in column
            )
            worksheet.column_dimensions[
                column[0].column_letter
            ].width = min(max_length + 2, 70)

    print(f"Extraction complete: {OUTPUT_EXCEL}")
    print(f"Rows extracted: {len(df)}")


if __name__ == "__main__":
    main()

```

## 3. Configure and run

Update these two paths in the script:

```
PDF_PATH = r"C:\PDF\Input.pdf"
OUTPUT_EXCEL = r"C:\PDF\checked_answers.xlsx"
```

Run:

```
python extract_checked_options.py
```

The Excel file will include the page number, question, checked option, all detected options, and a status column for review.

## Important limitations

This is a starting implementation, not a guaranteed extractor for every PDF layout. Your sample has very small checkboxes and multiple questions on a line, so checkbox detection and question association may need calibration. In particular:

- The script assumes option labels appear immediately to the right of checkboxes.
- It assumes the question text is to the left of the first checkbox on the same OCR line.
- The checked-state threshold may need adjustment for light ticks, crosses, or filled boxes.

The script saves `debug_page_1.png`, `debug_page_2.png`, and so on. Green boxes are detected as checked; red boxes are detected as unchecked. Review these images before relying on the Excel output.

If you can share one cropped, high-resolution image of a question and its checkboxes, I can tune the detection logic to match your exact PDF layout.
