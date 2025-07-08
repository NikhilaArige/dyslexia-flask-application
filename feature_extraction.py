import cv2
import numpy as np
from scipy import ndimage as nd
from skimage.filters import gabor, gabor_kernel

def extract_handwriting_features(image):
    """
    Extract comprehensive set of features from handwriting image
    Returns a feature vector
    """
    # Convert to grayscale if not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    features = []
    
    # 1. Basic image statistics
    features.append(np.mean(gray))  # Mean intensity
    features.append(np.std(gray))   # Standard deviation of intensity
    
    # 2. Contour-based features
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_areas = [cv2.contourArea(contour) for contour in contours if contour.size > 5]
    if contour_areas:
        features.append(np.mean(contour_areas))  # Mean contour area
        features.append(np.std(contour_areas))   # Variation in contour areas
    else:
        features.extend([0, 0])
    
    # 3. Slope/slant features
    slopes = []
    for contour in contours:
        if contour.shape[0] >= 5:  # Need at least 5 points to fit ellipse
            try:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                slopes.append(angle)
            except:
                pass
    
    if slopes:
        features.append(np.mean(slopes))  # Mean slope
        features.append(np.std(slopes))   # Slope consistency
    else:
        features.extend([0, 0])
    
    # 4. Spacing features
    h_proj = np.sum(binary, axis=0)
    spaces = []
    in_space = True
    space_start = 0
    
    for i, proj in enumerate(h_proj):
        if in_space and proj > 0:
            in_space = False
            space_width = i - space_start
            if space_width > 1:  # Minimum width to count as a space
                spaces.append(space_width)
        elif not in_space and proj == 0:
            in_space = True
            space_start = i
    
    if spaces:
        features.append(np.mean(spaces))  # Mean space width
        features.append(np.std(spaces))   # Space consistency
    else:
        features.extend([0, 0])
    
    # 5. Gabor features for texture analysis
    kernels = build_gabor_filters()
    gabor_features = extract_gabor_features(gray, kernels)
    features.extend(gabor_features)
    
    # 6. Pressure features (using intensity as proxy)
    # Darker pixels indicate higher pressure
    pixel_intensities = 255 - gray[binary > 0]
    if len(pixel_intensities) > 0:
        features.append(np.mean(pixel_intensities))  # Mean pressure
        features.append(np.std(pixel_intensities))   # Pressure consistency
    else:
        features.extend([0, 0])
    
    return features

def build_gabor_filters():
    """Build a set of Gabor filters for texture analysis"""
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 8):
        for sigma in [3, 5]:
            for frequency in [0.1, 0.25]:
                kernel = np.real(gabor_kernel(frequency, theta=theta, 
                                           sigma_x=sigma, sigma_y=sigma))
                filters.append(kernel)
    return filters

def extract_gabor_features(image, kernels):
    """Extract features using Gabor filters"""
    features = []
    for kernel in kernels:
        filtered = nd.convolve(image, kernel, mode='wrap')
        features.append(filtered.mean())
        features.append(filtered.var())
    return features

def extract_pressure_features(gray_image):
    """Extract features related to writing pressure"""
    # Binary threshold to segment writing
    _, binary = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Get intensity values only for pixels that are part of the writing
    writing_pixels = gray_image[binary > 0]
    
    features = []
    if len(writing_pixels) > 0:
        # Convert from grayscale to "pressure" (darker = higher pressure)
        pressure_values = 255 - writing_pixels
        
        # Calculate statistics
        mean_pressure = np.mean(pressure_values)
        std_pressure = np.std(pressure_values)
        max_pressure = np.max(pressure_values)
        min_pressure = np.min(pressure_values)
        pressure_range = max_pressure - min_pressure
        
        features.extend([mean_pressure, std_pressure, pressure_range])
    else:
        features.extend([0, 0, 0])
    
    return features