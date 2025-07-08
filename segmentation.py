import cv2
import numpy as np

def extract_text_lines(binary_image):
    """
    Extract text lines from a binary image
    Returns a list of images, each containing one text line
    """
    # Find horizontal projection profile
    h_proj = np.sum(binary_image, axis=1)
    
    # Identify line boundaries
    line_boundaries = []
    in_line = False
    start = 0
    
    for i, proj in enumerate(h_proj):
        if not in_line and proj > 0:
            # Start of a new line
            in_line = True
            start = i
        elif in_line and proj == 0:
            # End of a line
            in_line = False
            # Only add if the line has some minimum height
            if i - start > 5:  # Minimum line height threshold
                line_boundaries.append((start, i))
    
    # If the last line extends to the bottom of the image
    if in_line:
        line_boundaries.append((start, len(h_proj)))
    
    # Extract each line as a separate image
    lines = []
    for start, end in line_boundaries:
        # Add a small margin
        start_margin = max(0, start - 2)
        end_margin = min(binary_image.shape[0], end + 2)
        
        line_img = binary_image[start_margin:end_margin, :]
        lines.append(line_img)
    
    return lines

def preprocess_handwriting(image):
    """
    Preprocess a handwriting image for analysis
    """
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply thresholding to create binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Remove noise
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def segment_characters(binary_image):
    """
    Segment individual characters from a binary image
    Returns a list of bounding boxes (x, y, w, h) for each character
    """
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to identify potential characters
    character_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small contours which might be noise
        if w > 5 and h > 10:
            character_boxes.append((x, y, w, h))
    
    # Sort boxes from left to right
    character_boxes.sort(key=lambda box: box[0])
    
    return character_boxes