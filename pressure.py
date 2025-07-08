import cv2
import numpy as np

def analyze_pressure(grayscale_image):
    """
    Analyze handwriting pressure by examining intensity variations
    Returns a pressure variation score (higher means more variable pressure)
    """
    # Invert the image so that writing is white (255) on black (0) background
    inverted = cv2.bitwise_not(grayscale_image)
    
    # Apply threshold to isolate the writing
    _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.5  # Default value if no writing is detected
    
    # Create a mask for the handwriting
    mask = np.zeros_like(grayscale_image)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    # Apply mask to the original grayscale image
    masked_writing = cv2.bitwise_and(inverted, mask)
    
    # Get intensity values of the handwriting
    writing_pixels = masked_writing[masked_writing > 0]
    
    if len(writing_pixels) == 0:
        return 0.5  # Default value if no writing pixels detected
    
    # Calculate standard deviation of intensity values to measure pressure variation
    pressure_variation = np.std(writing_pixels) / 255.0
    
    return pressure_variation

def analyze_pressure_zones(grayscale_image):
    """
    Analyze pressure in different zones of the handwriting
    Returns a list of pressure measurements for different areas
    """
    # Divide image into 3x3 grid
    h, w = grayscale_image.shape
    zone_h, zone_w = h // 3, w // 3
    
    pressure_zones = []
    
    for i in range(3):
        for j in range(3):
            # Extract zone
            zone = grayscale_image[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
            
            # Calculate pressure for this zone
            zone_pressure = analyze_pressure(zone)
            pressure_zones.append(zone_pressure)
    
    return pressure_zones

def detect_pressure_patterns(grayscale_image):
    """
    Detect patterns in handwriting pressure that might indicate dyslexia
    Returns a score between 0 and 1 (higher means more dyslexic patterns)
    """
    # General pressure variation
    pressure_var = analyze_pressure(grayscale_image)
    
    # Zone-based pressure analysis
    zone_pressures = analyze_pressure_zones(grayscale_image)
    
    # Pressure inconsistency across zones
    zone_pressure_std = np.std(zone_pressures)
    
    # Combine features into a score
    # High variation in both overall pressure and between zones can indicate dyslexia
    dyslexia_score = (pressure_var * 0.6) + (zone_pressure_std * 0.4)
    
    # Normalize to 0-1
    return min(dyslexia_score, 1.0)