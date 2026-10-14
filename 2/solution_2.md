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
import reimport jsonfrom difflib import SequenceMatcherimport pymupdfimport cv2import numpy as npimport pytesseractfrom pytesseract import Output# ==========================================================# CONFIGURATION# ==========================================================pytesseract.pytesseract.tesseract_cmd = (    r"C:\Program Files\Tesseract-OCR\tesseract.exe")DPI = 300# Minimum OCR question-match score.QUESTION_MATCH_THRESHOLD = 0.60# Starting heuristic; calibrate against your PDF.CHECKED_THRESHOLD = 0.10SAVE_DEBUG_IMAGE = True# ==========================================================# TEXT UTILITIES# ==========================================================def normalize(text):    text = text.lower()    text = re.sub(r"[^a-z0-9]+", " ", text)    return " ".join(text.split())def similarity(text1, text2):    return SequenceMatcher(        None,        normalize(text1),        normalize(text2)    ).ratio()# ==========================================================# RENDER PDF PAGE USING PYMUPDF# ==========================================================def render_page(page):    pix = page.get_pixmap(        matrix=pymupdf.Matrix(DPI / 72, DPI / 72),        alpha=False    )    image = np.frombuffer(        pix.samples, dtype=np.uint8    ).reshape(pix.height, pix.width, pix.n)    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)# ==========================================================# OCR WORDS WITH POSITIONS# ==========================================================
```

## 3. Run the script

Update `PDF_PATH`, `PAGE_NUMBER`, and `QUESTION` at the bottom, then execute:

```
python extract_checked_options.py
```

The output will include `checked_option`, `status`, and diagnostic information about the detected checkboxes. The `bool()` and `float()` conversions prevent the JSON serialization error you encountered.

Important: checkbox detection is heuristic. For your form, validate the `debug_page_3.png` image and the extracted labels before using the result in an automated workflow. Green outlines indicate boxes classified as checked; red outlines indicate unchecked boxes. A wrong classification can still occur if the checkbox is tiny or the tick is faint.
