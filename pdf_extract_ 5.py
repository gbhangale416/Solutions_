import cv2
import numpy as np
import pytesseract

def get_checkbox_with_ocr(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.8 <= float(w)/h <= 1.25 and 10 <= w <= 40 and 10 <= h <= 40:
            boxes.append((x, y, w, h))

    # Deduplicate
    unique = []
    for b in sorted(boxes, key=lambda item: item[2]*item[3], reverse=True):
        if not any(abs(b[0] - u[0]) < 4 and abs(b[1] - u[1]) < 4 for u in unique):
            unique.append(b)

    unique.sort(key=lambda b: b[0])

    checked_labels = []
    for x, y, w, h in unique:
        # 1. Check if box is checked
        roi = thresh[y + 3 : y + h - 3, x + 3 : x + w - 3]
        fill_ratio = cv2.countNonZero(roi) / (roi.shape[0] * roi.shape[1]) if roi.size > 0 else 0
        
        # 2. Crop text immediately to the right of this box (width ~60px)
        text_crop = gray[max(0, y - 2) : min(gray.shape[0], y + h + 2), x + w : min(gray.shape[1], x + w + 70)]
        label = pytesseract.image_to_string(text_crop, config="--psm 7").strip()

        if fill_ratio > 0.12:
            checked_labels.append(label)

    return ", ".join(checked_labels) if checked_labels else "None Selected"