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

def locate_question_from_screenshot(
    page_image,
    screenshot_path
):
    """
    Locate a question screenshot inside a rendered PDF page
    using OpenCV template matching.

    The screenshot should contain a distinctive portion of
    the question text, preferably without large margins.
    """

    screenshot = cv2.imread(screenshot_path)

    if screenshot is None:
        raise FileNotFoundError(
            f"Cannot read screenshot: {screenshot_path}"
        )

    page_gray = cv2.cvtColor(
        page_image, cv2.COLOR_BGR2GRAY
    )

    template_gray = cv2.cvtColor(
        screenshot, cv2.COLOR_BGR2GRAY
    )

    page_height, page_width = page_gray.shape
    template_height, template_width = template_gray.shape

    # Try several scales because the screenshot and PDF
    # may have different resolutions.
    best = None

    for scale in np.linspace(0.5, 2.0, 31):
        new_width = int(template_width * scale)
        new_height = int(template_height * scale)

        if (
            new_width < 10
            or new_height < 10
            or new_width > page_width
            or new_height > page_height
        ):
            continue

        resized = cv2.resize(
            template_gray,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        result = cv2.matchTemplate(
            page_gray,
            resized,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_score, _, max_location = (
            cv2.minMaxLoc(result)
        )

        if best is None or max_score > best["score"]:
            best = {
                "score": float(max_score),
                "x": int(max_location[0]),
                "y": int(max_location[1]),
                "w": int(new_width),
                "h": int(new_height)
            }

    if best is None or best["score"] < 0.55:
        raise ValueError(
            "Could not locate the screenshot on the PDF page. "
            "Use a clearer crop or a higher-resolution screenshot. "
            f"Best score: {best['score'] if best else 'N/A'}"
        )

    return {
        "x1": best["x"],
        "y1": best["y"],
        "x2": best["x"] + best["w"],
        "y2": best["y"] + best["h"],
        "method": "screenshot_template_matching",
        "match_score": best["score"]
    }


def get_checked_option(
    pdf_path,
    page_number,
    question_screenshot
):
    """
    Inputs:
        pdf_path: Path to the original PDF.
        page_number: 1-based page number.
        question_screenshot: Screenshot crop of the question.

    Returns:
        A JSON-serializable dictionary.
    """

    with pymupdf.open(pdf_path) as pdf:

        if not 1 <= page_number <= len(pdf):
            raise ValueError(
                f"Invalid page {page_number}. "
                f"PDF has {len(pdf)} pages."
            )

        page = pdf[page_number - 1]
        image = render_page(page)

        question_box = locate_question_from_screenshot(
            image,
            question_screenshot
        )

        # Find checkbox candidates associated with the
        # matched screenshot area.
        boxes = detect_checkboxes(
            image,
            question_box
        )

        if not boxes:
            return {
                "page_number": page_number,
                "question_screenshot": question_screenshot,
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
            "page_number": int(page_number),
            "question_screenshot": question_screenshot,
            "checked_option": checked_option,
            "status": status,
            "screenshot_match_score": question_box["match_score"],
            "options": options
        }


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
