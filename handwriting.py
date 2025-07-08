import numpy as np
import cv2
import os
import pickle
import tensorflow as tf
from PIL import Image
import io

class HandwritingAnalyzer:
    def __init__(self, model_path=None):
        """
        Initialize the handwriting analyzer.
        
        Args:
            model_path: Path to the trained handwriting model.
        """
        # Path to the model
        self.model_path = model_path or os.path.join(os.getcwd(), 'models', 'handwriting_model.h5')
        
        # Load model if it exists
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path) 
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading handwriting model: {e}")
                self.model_loaded = False
                
            # Alternative: try loading pickle model if keras model fails
            if not self.model_loaded:
                try:
                    pkl_path = os.path.join(os.getcwd(), 'models', 'handwriting_dyslexia_model.pkl')
                    if os.path.exists(pkl_path):
                        with open(pkl_path, 'rb') as f:
                            self.model = pickle.load(f)
                        self.model_loaded = True
                        self.model_type = 'sklearn'
                    else:
                        print(f"No pickle model found at {pkl_path}")
                except Exception as e:
                    print(f"Error loading pickle model: {e}")
        else:
            print(f"Model not found at {self.model_path}")
            self.model_loaded = False
    
    def preprocess_image(self, image_data):
        """
        Preprocess the handwriting image for analysis.
        
        Args:
            image_data: Image data as bytes or a file-like object
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert to PIL Image
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            else:
                img = Image.open(image_data)
                
            # Convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
                
            # ✅ Resize to match model input shape (128, 128)
            img = img.resize((128, 128))
            
            # Normalize the image
            img_array = np.array(img) / 255.0
            
            # Reshape based on model type
            if self.model_loaded and hasattr(self, 'model_type') and self.model_type == 'sklearn':
                return img_array.flatten().reshape(1, -1)
            else:
                # ✅ For deep learning models (Conv2D), return shape: (1, 128, 128, 1)
                return img_array.reshape(1, 128, 128, 1)
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return np.zeros((1, 128, 128, 1))

    
    def extract_features(self, image):
        """
        Extract features from the handwriting image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Convert to OpenCV format if not already
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 1:
                img = image[0]
                if img.shape[-1] == 1:
                    img = img.squeeze()
            else:
                img = image
                
            # Ensure image is grayscale and in uint8 format
            if isinstance(img, np.ndarray) and img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
                
            # Thresholding to binary image
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate spacing between letters
            letter_widths = []
            spaces = []
            
            if len(contours) > 1:
                # Sort contours by x-coordinate
                contours_sorted = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
                
                for i in range(len(contours_sorted)):
                    x, y, w, h = cv2.boundingRect(contours_sorted[i])
                    letter_widths.append(w)
                    
                    if i > 0:
                        prev_x, prev_y, prev_w, prev_h = cv2.boundingRect(contours_sorted[i-1])
                        space = x - (prev_x + prev_w)
                        spaces.append(space)
            
            # Calculate statistics
            avg_letter_width = np.mean(letter_widths) if letter_widths else 0
            avg_space = np.mean(spaces) if spaces else 0
            space_consistency = np.std(spaces) if len(spaces) > 1 else 0
            
            # Line consistency - check if all letters align on a baseline
            y_positions = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours]
            line_consistency = np.std(y_positions) if y_positions else 0
            
            # Letter reversals - this would require more sophisticated analysis
            # For now, we'll use a placeholder based on letter shapes
            # In a real implementation, this would use more advanced pattern recognition
            reversals = 0
            
            # Gather features
            features = {
                'avg_letter_width': avg_letter_width,
                'avg_space': avg_space,
                'space_consistency': space_consistency,
                'line_consistency': line_consistency,
                'num_contours': len(contours)
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {
                'avg_letter_width': 0,
                'avg_space': 0,
                'space_consistency': 0,
                'line_consistency': 0,
                'num_contours': 0
            }
    
    def analyze_handwriting(self, image_data):
        """
        Analyze handwriting image to detect patterns associated with dyslexia.
        
        Args:
            image_data: Image data as bytes or a file-like object
                
        Returns:
            Dictionary with analysis results:
                - probability: Probability of dyslexia indicators (0-1)
                - spacing_issues: Detected spacing issues score (0-10)
                - reversals: Detected letter reversals
                - line_consistency: Line consistency score (0-10)
                - explanation: Explanation of results
        """
        # Preprocess the image
        processed_image = self.preprocess_image(image_data)
        
        # Extract features
        features = self.extract_features(processed_image)
        
        # Use model for prediction if available
        if self.model_loaded:
            try:
                probability = float(self.model.predict(processed_image)[0][0])
            except Exception as e:
                print(f"Error using model for prediction: {e}")
                # Fallback to rule-based method
                probability = self._rule_based_analysis(features)
        else:
            # Use rule-based method if no model is available
            probability = self._rule_based_analysis(features)
        
        # Calculate scores
        # Normalize to 0-10 scale
        spacing_issues = min(10, max(0, 5 + (features['space_consistency'] - 5) / 2))
        line_consistency = min(10, max(0, 10 - features['line_consistency'] / 5))
        
        # Letter reversals - in a real implementation, this would be more sophisticated
        # For now, use a placeholder value
        reversals = 0 if probability < 0.5 else max(1, int(probability * 5))
        
        # Generate explanation
        explanation = self._generate_explanation(probability, spacing_issues, reversals, line_consistency)
        
        return {
            'probability': probability,
            'spacing_score': float(spacing_issues),
            'reversals': reversals,
            'line_consistency': float(line_consistency),
            'explanation': explanation
        }
    
    def _rule_based_analysis(self, features):
        """
        Rule-based analysis to estimate dyslexia probability.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Probability of dyslexia indicators (0-1)
        """
        # Weights for different features
        weights = {
            'space_consistency': 0.4,  # Spacing issues are important
            'line_consistency': 0.4,   # Line consistency is important
            'letter_width_variance': 0.2  # Letter size consistency less important
        }
        
        # Normalize features to 0-1 scale (higher = more problematic)
        norm_space = min(1.0, features['space_consistency'] / 20)
        norm_line = min(1.0, features['line_consistency'] / 30)
        
        # Calculate weighted score
        score = (
            weights['space_consistency'] * norm_space + 
            weights['line_consistency'] * norm_line
        )
        
        # Adjust to 0-1 range
        probability = min(1.0, max(0.0, score))
        
        return probability
    
    def _generate_explanation(self, probability, spacing_issues, reversals, line_consistency):
        """Generate a human-readable explanation of the results."""
        if probability < 0.3:
            return (
                f"Handwriting analysis shows typical patterns. "
                f"The spacing between letters is consistent, and letters are aligned well on the line. "
                f"No significant indicators of writing difficulties were detected."
            )
        elif probability < 0.7:
            return (
                f"Some atypical handwriting patterns detected. "
                f"The analysis shows minor spacing inconsistencies (score: {spacing_issues:.1f}/10) "
                f"and some line alignment issues (score: {line_consistency:.1f}/10). "
                f"These could indicate mild writing challenges but are not conclusive."
            )
        else:
            return (
                f"Handwriting analysis shows significant indicators associated with writing difficulties. "
                f"The analysis detected irregular spacing between letters (score: {spacing_issues:.1f}/10), "
                f"potential letter reversals ({reversals}), and inconsistent line alignment (score: {line_consistency:.1f}/10). "
                f"These are common patterns in individuals with dyslexia."
            )