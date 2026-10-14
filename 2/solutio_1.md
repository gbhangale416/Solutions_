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
                "dark_ratio": box["dark_ratio"],                "position": {                    "x": box["x"],                    "y": box["y"]                }            })        selected = [            option for option in options            if option["checked"]        ]        if len(selected) == 1 and selected[0]["label"]:            checked_option = selected[0]["label"]            status = "OK"        elif len(selected) == 0:            checked_option = None            status = "Review required: no checked option detected"        else:            checked_option = None            status = (                "Review required: multiple checked boxes "                "or missing option labels"            )        return {            "page_number": page_number,            "question": question,            "checked_option": checked_option,            "status": status,            "options": options        }# ==========================================================# EXECUTION EXAMPLE# ==========================================================if __name__ == "__main__":    pdf_path = r"C:\PDF\Input.pdf"    page_number = 3    question = (        "Do you experience excessive bloating after eating?"    )    result = get_checked_option(        pdf_path=pdf_path,        page_number=page_number,        question=question    )    print(json.dumps(result, indent=4))
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
