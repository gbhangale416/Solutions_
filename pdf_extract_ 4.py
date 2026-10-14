import cv2
import numpy as np

def get_checkbox_status_from_crop(image_path_or_array):
    # 1. Read input
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Binary threshold (invert: dark checkmarks become white)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 3. Detect contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_img, w_img = img.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Filter for square boxes: roughly square, sensible size relative to crop height
        if 0.8 <= aspect_ratio <= 1.25 and 10 <= w <= 40 and 10 <= h <= 40:
            boxes.append((x, y, w, h))

    # Remove nested/duplicate contours
    filtered_boxes = []
    for b in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        bx, by, bw, bh = b
        if not any(abs(bx - fx) < 4 and abs(by - fy) < 4 for fx, fy, _, _ in filtered_boxes):
            filtered_boxes.append(b)

    # Sort boxes from left to right
    filtered_boxes.sort(key=lambda b: b[0])

    if not filtered_boxes:
        return "No checkboxes detected"

    # Map options depending on how many boxes exist in the question line
    if len(filtered_boxes) == 2:
        labels = ["Yes", "No"]
    elif len(filtered_boxes) == 3:
        labels = ["Yes", "No", "Unknown"]
    else:
        labels = [f"Box_{i+1}" for i in range(len(filtered_boxes))]

    # 4. Check interior pixel fill ratio of each box
    checked_options = []
    margin = 3  # shave off outer border

    for (x, y, w, h), label in zip(filtered_boxes, labels):
        roi = thresh[y + margin : y + h - margin, x + margin : x + w - margin]
        if roi.size > 0:
            fill_ratio = cv2.countNonZero(roi) / (roi.shape[0] * roi.shape[1])
        else:
            fill_ratio = 0.0

        # Any checkmark / tick / cross exceeds 12% fill density
        if fill_ratio > 0.12:
            checked_options.append(label)

    return ", ".join(checked_options) if checked_options else "None Selected"


# Example usage on your cropped snippet:
answer = get_checkbox_status_from_crop("cropped_question.png")
print("Detected Answer:", answer)