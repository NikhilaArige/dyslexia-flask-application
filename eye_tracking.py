import numpy as np
import cv2
import json
import pickle
import os
import h5py

class EyeTracker:
    def __init__(self, model_path=None):
        """
        Initialize the eye tracking processor.
        
        Args:
            model_path: Path to the trained eye movement model.
        """
        # Path to the model
        self.model_path = model_path or os.path.join(os.getcwd(), 'models', 'eye_movement_trained.h5')
        
        # Load model if it exists
        if os.path.exists(self.model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading eye tracking model: {e}")
                self.model_loaded = False
        else:
            print(f"Model not found at {self.model_path}")
            self.model_loaded = False
    
    def analyze_eye_movements(self, data):
        """
        Analyze eye movement data to detect patterns associated with dyslexia.
        
        Args:
            data: Dictionary containing eye tracking data:
                - fixations: Number of fixations
                - saccades: Number of saccades
                - regressions: Number of regressions
                
        Returns:
            Dictionary with analysis results:
                - probability: Probability of dyslexia indicators (0-1)
                - fixations: Analyzed fixation count
                - saccades: Analyzed saccade count
                - regressions: Analyzed regression count
                - explanation: Explanation of results
        """
        # Extract features
        fixations = data.get('fixations', 0)
        saccades = data.get('saccades', 0)
        regressions = data.get('regressions', 0)
        
        # Normalize the values (based on typical ranges)
        norm_fixations = min(fixations / 130, 2.0)  # Normal range ~100-130
        norm_saccades = min(saccades / 100, 2.0)    # Normal range ~80-100
        norm_regression_ratio = min(regressions / 15, 2.0)  # Normal range ~5-15
        
        # Calculate dyslexia indicators
        # Higher values for each indicate potential dyslexia
        # These weights are based on research showing regressions are particularly important
        weights = {
            'fixations': 0.3,  # More fixations often indicate difficulty
            'saccades': 0.2,   # More saccades can indicate difficulty
            'regressions': 0.5  # Regressions are strong indicators
        }
        
        # If we have a trained model, use it for prediction
        if self.model_loaded:
            try:
                features = np.array([[norm_fixations, norm_saccades, norm_regression_ratio]])
                probability = float(self.model.predict(features)[0][0])
            except Exception as e:
                print(f"Error using model for prediction: {e}")
                # Fallback to rule-based method
                probability = (
                    weights['fixations'] * (norm_fixations - 1.0) + 
                    weights['saccades'] * (norm_saccades - 1.0) +
                    weights['regressions'] * (norm_regression_ratio - 1.0)
                )
                # Normalize to 0-1 range
                probability = max(0.0, min(1.0, probability + 0.5))
        else:
            # Rule-based method if no model is available
            probability = (
                weights['fixations'] * (norm_fixations - 1.0) + 
                weights['saccades'] * (norm_saccades - 1.0) +
                weights['regressions'] * (norm_regression_ratio - 1.0)
            )
            # Normalize to 0-1 range
            probability = max(0.0, min(1.0, probability + 0.5))
        
        # Generate explanation
        explanation = self._generate_explanation(fixations, saccades, regressions, probability)
        
        return {
            'probability': probability,
            'fixations': fixations,
            'saccades': saccades,
            'regressions': regressions,
            'explanation': explanation
        }
    
    def _generate_explanation(self, fixations, saccades, regressions, probability):
        """Generate a human-readable explanation of the results."""
        if probability < 0.3:
            return (
                f"Eye movement patterns show typical reading behavior. "
                f"The number of fixations ({fixations}) and regressions ({regressions}) "
                f"are within normal ranges, suggesting fluid reading progression."
            )
        elif probability < 0.7:
            return (
                f"Some atypical eye movement patterns detected. "
                f"An increased number of fixations ({fixations}) and/or regressions ({regressions}) "
                f"may indicate some reading difficulty, but results are not conclusive."
            )
        else:
            return (
                f"Eye movement patterns show significant indicators associated with reading difficulties. "
                f"The high number of fixations ({fixations}) and regressions ({regressions}) "
                f"suggest potential challenges with visual processing during reading."
            )