import pymupdf
import cv2
import numpy as np

def extract_answer_from_question_crop(pdf_path, page_num, question_crop_path):
    """
    pdf_path: Path to the target PDF
    page_num: 1-indexed page number (e.g., 23)
    question_crop_path: File path of the question screenshot
    """
    # 1. Render PDF page at 2.0x resolution (~144 DPI)
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_num - 1)
    zoom = 2.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    
    page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    page_img = cv2.cvtColor(page_img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
    page_gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

    # 2. Load screenshot query
    template = cv2.imread(question_crop_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return "Error: Could not load the question screenshot."

    # In case the screenshot resolution differs slightly, match at current scale
    # (or resize if screenshot was taken on a higher/lower DPI monitor)
    th, tw = template.shape[:2]
    res = cv2.matchTemplate(page_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If direct match confidence is lower than 0.65, rescale template to find best fit
    best_scale = 1.0
    if max_val < 0.65:
        for scale in np.linspace(0.6, 1.4, 17):
            resized_tpl = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            if resized_tpl.shape[0] > page_gray.shape[0] or resized_tpl.shape[1] > page_gray.shape[1]:
                continue
            r = cv2.matchTemplate(page_gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
            _, s_max_val, _, s_max_loc = cv2.minMaxLoc(r)
            if s_max_val > max_val:
                max_val = s_max_val
                max_loc = s_max_loc
                best_scale = scale

    if max_val < 0.50:
        return f"Question not matched reliably on page {page_num} (Confidence: {max_val:.2f})"

    # 3. Locate the matched region
    match_x, match_y = max_loc
    scaled_w = int(tw * best_scale)
    scaled_h = int(th * best_scale)

    # Crop the full row around the question and extending to the right margin
    crop_y1 = max(0, match_y - 10)
    crop_y2 = min(page_gray.shape[0], match_y + scaled_h + 10)
    crop_x1 = match_x
    crop_x2 = min(page_gray.shape[1], match_x + scaled_w + 300)

    row_roi = page_gray[crop_y1:crop_y2, crop_x1:crop_x2]

    # 4. Detect checkboxes within this matched row
    _, thresh = cv2.threshold(row_roi, 205, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_dim = int(10 * zoom)
    max_dim = int(28 * zoom)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        if 0.8 <= aspect_ratio <= 1.25 and min_dim <= w <= max_dim and min_dim <= h <= max_dim:
            boxes.append((x, y, w, h))

    # Deduplicate nested box boundaries
    filtered_boxes = []
    for b in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if not any(abs(b[0] - fb[0]) < 5 and abs(b[1] - fb[1]) < 5 for fb in filtered_boxes):
            filtered_boxes.append(b)

    # Sort checkboxes from left to right
    filtered_boxes.sort(key=lambda b: b[0])

    if not filtered_boxes:
        return "No checkboxes detected for this question"

    # Map options depending on total boxes present in this question row
    if len(filtered_boxes) == 2:
        labels = ["Yes", "No"]
    elif len(filtered_boxes) == 3:
        labels = ["Yes", "No", "Unknown"]
    else:
        # If the screenshot already included the options row or part of adjacent text
        labels = ["Yes", "No", "Unknown"][:len(filtered_boxes)]

    # 5. Measure ink density inside each box
    checked_answers = []
    margin = int(2.5 * zoom)

    for (bx, by, bw, bh), label in zip(filtered_boxes, labels):
        inner_roi = thresh[by + margin : by + bh - margin, bx + margin : bx + bw - margin]
        if inner_roi.size > 0:
            fill_ratio = cv2.countNonZero(inner_roi) / (inner_roi.shape[0] * inner_roi.shape[1])
        else:
            fill_ratio = 0.0

        # Checkmark/cross occupies > 12% interior pixels; empty box is near 0%
        if fill_ratio > 0.12:
            checked_answers.append(label)

    return ", ".join(checked_answers) if checked_answers else "None Selected"


# Example execution:
pdf_file = "sample.pdf"
question_screenshot = "question_crop.png"  # your snippet image

answer = extract_answer_from_question_crop(pdf_file, 23, question_screenshot)
print("Detected Answer:", answer)