import pymupdf  # Replaces 'import fitz' to resolve warning
import cv2
import numpy as np

pdf_path = "your_document.pdf"
page_number = 23  # 1-indexed

# 1. Load and render page at 2x resolution (~144 DPI)
doc = pymupdf.open(pdf_path)
page = doc.load_page(page_number - 1)
zoom = 2.0
pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))

# Convert to OpenCV BGR image
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

# 2. Binary threshold (invert: black marks/borders become white on black background)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

# 3. Morphological extraction: isolate horizontal and vertical box edges
# This separates box boundaries from text characters
scale_factor = int(12 * (zoom / 2.0))
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale_factor, 1))
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_factor))

h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

# Combine lines to recover complete checkbox rectangles
box_table_mask = cv2.add(h_lines, v_lines)

# 4. Find all closed box contours
contours, _ = cv2.findContours(box_table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

raw_boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Checkbox parameters (accommodates 10-35 pt checkboxes scaled by zoom)
    if 0.75 <= aspect_ratio <= 1.3 and int(12 * zoom) <= w <= int(35 * zoom) and int(12 * zoom) <= h <= int(35 * zoom):
        raw_boxes.append((x, y, w, h))

# Remove duplicate/overlapping bounding boxes
unique_boxes = []
for b in sorted(raw_boxes, key=lambda item: item[2] * item[3], reverse=True):
    bx, by, bw, bh = b
    if not any(abs(bx - fx) < 6 and abs(by - fy) < 6 for fx, fy, _, _ in unique_boxes):
        unique_boxes.append(b)

print(f"Total checkboxes detected: {len(unique_boxes)}")

# 5. Extract words from PDF for label matching
words = page.get_text("words")  # (x0, y0, x1, y1, text, block_no, line_no, word_no)

results = []
output_img = img.copy()
margin = int(2.5 * zoom)

for x, y, w, h in unique_boxes:
    # Crop checkbox interior (exclude borders)
    inner_roi = thresh[y + margin : y + h - margin, x + margin : x + w - margin]
    
    if inner_roi.size > 0:
        total_px = inner_roi.shape[0] * inner_roi.shape[1]
        fill_ratio = cv2.countNonZero(inner_roi) / total_px
    else:
        fill_ratio = 0.0

    # Fill ratio threshold: checked boxes typically have > 12% ink density inside
    is_checked = fill_ratio > 0.12
    status = "CHECKED" if is_checked else "Unchecked"

    # Map box back to PDF coordinate space for text matching
    pdf_x, pdf_y = x / zoom, y / zoom
    pdf_w, pdf_h = w / zoom, h / zoom

    # Find the nearest text directly to the right of the checkbox
    label_candidates = [
        w[4] for w in words
        if (pdf_x + pdf_w - 2) <= w[0] <= (pdf_x + pdf_w + 90)
        and abs(((w[1] + w[3]) / 2) - (pdf_y + pdf_h / 2)) < 10
    ]
    label = " ".join(label_candidates) if label_candidates else "(No label nearby)"

    results.append({
        "label": label,
        "status": status,
        "ratio": fill_ratio,
        "box": (x, y, w, h),
        "y": y
    })

    # Draw color indicators: Green for checked, Red for unchecked
    color = (0, 200, 0) if is_checked else (0, 0, 240)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(output_img, f"{'X' if is_checked else ''}", (x + 3, y + h - 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * zoom, color, 2)

# Sort reading order: row-wise (top to bottom), then column-wise (left to right)
row_band = int(14 * zoom)
results.sort(key=lambda item: (round(item["y"] / row_band) * row_band, item["box"][0]))

# 6. Print final table
print(f"\n{'#':<4} {'Label / Option':<25} {'Status':<14} {'Fill %':<10} {'Box (x, y, w, h)'}")
print("-" * 75)
for i, item in enumerate(results, 1):
    print(f"{i:<4} {item['label']:<25} {item['status']:<14} {item['ratio']:.1%}      {item['box']}")

# Save diagnostic output images
cv2.imwrite("detected_checkboxes_annotated.png", output_img)
cv2.imwrite("mask_debug.png", box_table_mask)
print("\nSaved annotated image to 'detected_checkboxes_annotated.png'.")