import fitz  # PyMuPDF
import cv2
import numpy as np

pdf_path = "your_document.pdf"
page_number = 23  # 1-indexed

# 1. Render PDF Page 23 to an OpenCV image at high resolution (2x zoom)
doc = fitz.open(pdf_path)
page = doc.load_page(page_number - 1)

zoom = 2.0
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat)

# Convert pixmap to numpy BGR image
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

# 2. Binary thresholding for checkbox detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# 3. Detect all square checkboxes
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
# Expected square size bounds (in zoomed pixels)
min_dim, max_dim = int(10 * zoom), int(26 * zoom)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    if 0.85 <= aspect_ratio <= 1.15 and min_dim <= w <= max_dim and min_dim <= h <= max_dim:
        boxes.append((x, y, w, h))

# Remove duplicate nested contours
filtered_boxes = []
for b in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
    bx, by, bw, bh = b
    if not any(abs(bx - fx) < 4 and abs(by - fy) < 4 for fx, fy, _, _ in filtered_boxes):
        filtered_boxes.append(b)

# 4. Extract text words with bounding boxes to map options (Yes / No / Dx)
words = page.get_text("words")  # returns: (x0, y0, x1, y1, "word", block_no, line_no, word_no)

# 5. Evaluate status and associate with nearby text
results = []
output_img = img.copy()
margin = int(2.5 * zoom)

for x, y, w, h in filtered_boxes:
    # Measure pixel density inside the box (excluding border)
    roi = thresh[y + margin : y + h - margin, x + margin : x + w - margin]
    fill_ratio = (cv2.countNonZero(roi) / (roi.shape[0] * roi.shape[1])) if roi.size > 0 else 0
    is_checked = fill_ratio > 0.15

    # Map box coordinates back to PDF points
    pdf_x, pdf_y = x / zoom, y / zoom
    pdf_w, pdf_h = w / zoom, h / zoom

    # Find the nearest label to the right of the checkbox (within 80 points)
    label_candidates = [
        w[4] for w in words
        if (pdf_x + pdf_w - 2) <= w[0] <= (pdf_x + pdf_w + 80)
        and abs((w[1] + w[3]) / 2 - (pdf_y + pdf_h / 2)) < 8
    ]
    label = " ".join(label_candidates) if label_candidates else "Unknown"

    results.append({
        "bbox": (x, y, w, h),
        "label": label,
        "checked": is_checked,
        "ratio": fill_ratio,
        "y": y
    })

    # Visual overlay
    color = (0, 180, 0) if is_checked else (0, 0, 220)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(output_img, "CHK" if is_checked else "", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

# Sort results top-to-bottom, then left-to-right
results.sort(key=lambda item: (round(item["y"] / (15 * zoom)) * (15 * zoom), item["bbox"][0]))

# 6. Display results table
print(f"{'#':<4} {'Label / Option':<25} {'Status':<12} {'Fill %':<10} {'Box (x,y,w,h)'}")
print("-" * 75)
for i, item in enumerate(results, 1):
    status_str = "[X] CHECKED" if item["checked"] else "[ ] Unchecked"
    print(f"{i:<4} {item['label']:<25} {status_str:<12} {item['ratio']:.1%}      {item['bbox']}")

cv2.imwrite("page_23_detected.png", output_img)