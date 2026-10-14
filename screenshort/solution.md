You can pass a screenshot of the question instead of the question text. The code will use that screenshot to locate the matching question on the specified PDF page, then detect the checked checkbox and return its label.

For example:

```
result = get_checked_option(    pdf_path=r"C:\PDF\Input.pdf",    page_number=3,    question_screenshot=r"C:\PDF\question.png")print(json.dumps(result, indent=4))
```

## Updated end-to-end code

Keep the existing imports, OCR, rendering, checkbox detection, and label-reading functions from the previous script. Replace `locate_question()` and `get_checked_option()` with the following functions, and add the screenshot-matching helper.

```
def locate_question_from_screenshot(    page_image,    screenshot_path):    """    Locate a question screenshot inside a rendered PDF page    using OpenCV template matching.    The screenshot should contain a distinctive portion of    the question text, preferably without large margins.    """    screenshot = cv2.imread(screenshot_path)    if screenshot is None:        raise FileNotFoundError(            f"Cannot read screenshot: {screenshot_path}"        )    page_gray = cv2.cvtColor(        page_image, cv2.COLOR_BGR2GRAY    )    template_gray = cv2.cvtColor(        screenshot, cv2.COLOR_BGR2GRAY    )    page_height, page_width = page_gray.shape    template_height, template_width = template_gray.shape    # Try several scales because the screenshot and PDF    # may have different resolutions.    best = None    for scale in np.linspace(0.5, 2.0, 31):
```

## Important notes

- The screenshot should be cropped around the question text, ideally including its checkbox options if possible.
- The screenshot and PDF page should have similar visual content. Template matching can fail if the screenshot is resized, heavily compressed, or captured at a different zoom level.
- The existing `detect_checkboxes()` function uses the matched screenshot area to find checkboxes. If your screenshot contains only the question text and excludes the options, the checkbox search region may need to extend farther to the right. For reliable results, the best next improvement is to use the screenshot to locate the question, then search the entire question row for its option checkboxes.
