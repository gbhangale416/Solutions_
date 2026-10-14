import pymupdf  # PyMuPDF
import cv2
import numpy as np

pdf_path = "your_document.pdf"
page_number = 23  # 1-indexed

# 1. Load and render page at 2x resolution
doc = pymupdf.open(pdf_path)
page = doc.load_page(page_number - 1)
zoom = 2.0
pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))

img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

# 2. Thresholding and morphological extraction for checkboxes
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

scale_factor = int(10 * (zoom / 2.0))
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale_factor, 1))
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_factor))

h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
box_mask = cv2.add(h_lines, v_lines)

# 3. Detect all square checkboxes
contours, _ = cv2.findContours(box_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

raw_boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    if 0.75 <= aspect_ratio <= 1.3 and int(10 * zoom) <= w <= int(35 * zoom) and int(10 * zoom) <= h <= int(35 * zoom):
        raw_boxes.append((x, y, w, h))

# Remove duplicate nested bounding boxes
unique_boxes = []
for b in sorted(raw_boxes, key=lambda item: item[2] * item[3], reverse=True):
    bx, by, bw, bh = b
    if not any(abs(bx - fx) < 6 and abs(by - fy) < 6 for fx, fy, _, _ in unique_boxes):
        unique_boxes.append(b)

# 4. Check status of each box
margin = int(2.5 * zoom)
checked_boxes = []

for x, y, w, h in unique_boxes:
    inner_roi = thresh[y + margin : y + h - margin, x + margin : x + w - margin]
    if inner_roi.size > 0:
        fill_ratio = cv2.countNonZero(inner_roi) / (inner_roi.shape[0] * inner_roi.shape[1])
    else:
        fill_ratio = 0.0

    # Convert pixel coords back to PDF points
    pdf_rect = fitz_rect = (x / zoom, y / zoom, (x + w) / zoom, (y + h) / zoom)
    is_checked = fill_ratio > 0.12
    checked_boxes.append((pdf_rect, is_checked))

# 5. Extract text blocks / lines from PDF to map questions & options
# words: (x0, y0, x1, y1, word, block_no, line_no, word_no)
words = page.get_text("words")

# Helper function: find status (Yes, No, Unknown) for any specific question text
def get_question_answer(question_text):
    # Find word occurrences of the question to locate its vertical position
    question_clean = question_text.lower().strip()
    target_words = page.get_text("blocks")
    
    q_bbox = None
    for block in target_words:
        # block: (x0, y0, x1, y1, text, block_no, block_type)
        if question_clean in block[4].lower():
            q_bbox = block[:4]
            break

    if not q_bbox:
        return f"Question '{question_text}' not found on page."

    q_y0, q_y1 = q_bbox[1], q_bbox[3]

    # Find checkboxes on the same line or immediate line of this question
    line_boxes = [
        (rect, is_checked) for rect, is_checked in checked_boxes
        if abs(((rect[1] + rect[3]) / 2) - ((q_y0 + q_y1) / 2)) < 15
    ]

    # Find the corresponding label ('Yes', 'No', 'Unknown') directly next to each checkbox
    checked_answers = []
    all_options = []

    for rect, is_checked in line_boxes:
        bx1, by_mid = rect[2], (rect[1] + rect[3]) / 2
        
        # Look for option words within 50 points to the right of the checkbox
        option_words = [
            w[4] for w in words
            if (bx1 - 2) <= w[0] <= (bx1 + 55) and abs(((w[1] + w[3]) / 2) - by_mid) < 8
        ]
        label = " ".join(option_words).strip()
        
        if label:
            all_options.append((label, is_checked))
            if is_checked:
                checked_answers.append(label)

    if checked_answers:
        return ", ".join(checked_answers)
    elif all_options:
        return "None Selected"
    else:
        return "No options detected"

# 6. Query specific questions directly
questions_to_check = [
    "Do you experience excessive bloating after eating?",
    "Do you have pain just below the rib cage",
    "Do you have malodorous, fatty stools?",
    "Do you experience reflux symptoms?",
    "Chronic constipation requiring treatment?",
    "Have you fainted or lost consciousness?",
    "Have you ever had a stroke (CVA)?",
    "Do you have a seizure disorder?"
]

print(f"{'Question':<55} | {'Answer (Yes / No / Unknown)'}")
print("-" * 85)
for q in questions_to_check:
    answer = get_question_answer(q)
    print(f"{q:<55} | {answer}")