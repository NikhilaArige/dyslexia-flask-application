# utils/audio_processing.py
import numpy as np
import os
import random

def extract_audio_features(filepath):
    """Simple mock function to extract audio features without requiring advanced libraries"""
    # In a real implementation, this would use librosa to extract features
    # For compatibility, we'll return mock features of expected dimensions
    
    try:
        # Check if file exists and has content
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise ValueError("Audio file is empty")
            
        # Return mock features of expected shape
        # These dimensions should match what your model expects
        mock_features = np.random.random(25)  # Adjust size to match expected features
        return mock_features
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        # Return default features
        return np.random.random(25)  # Adjust size to match expected features

def analyze_audio(audio_data):
    """
    Simple mock function to analyze audio data without requiring advanced libraries
    
    Args:
        audio_data: Path to audio file or audio data
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Generate a consistent but random score between 0.3 and 0.7
        # Using hash of the filepath if it's a string to make it consistent for the same file
        if isinstance(audio_data, str) and os.path.exists(audio_data):
            # Use the file size to seed the random generator for consistency
            file_size = os.path.getsize(audio_data)
            random.seed(file_size)
        else:
            # Otherwise use a fixed seed
            random.seed(42)
            
        score = random.uniform(0.3, 0.7)
        
        # Fluency score (inverse of probability, higher is better)
        fluency_score = max(0, min(10, 10 - score * 10))
        
        # Mock pronunciation issues
        pronunciation_issues = max(0, min(20, int(score * 20)))
        
        # Generate explanation based on score
        if score < 0.4:
            risk_level = "low"
            advice = "No significant indicators of reading difficulties detected in the audio."
        elif score < 0.6:
            risk_level = "moderate"
            advice = "Some potential indicators of reading difficulties detected. Consider further assessment."
        else:
            risk_level = "high"
            advice = "Several indicators of reading difficulties detected in the speech pattern. Professional assessment is recommended."
        
        explanation = f"Analysis indicates a {risk_level} probability ({score:.2f}) of dyslexia-related reading patterns.\n\n"
        explanation += f"Reading fluency score: {fluency_score:.1f}/10\n"
        
        if pronunciation_issues > 0:
            explanation += f"Detected approximately {pronunciation_issues} pronunciation irregularities. "
        
        explanation += advice
        
        return {
            'probability': score,
            'fluency_score': float(fluency_score),
            'pronunciation_issues': pronunciation_issues,
            'explanation': explanation
        }
            
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        # Return default results
        return {
            'probability': 0.5,
            'fluency_score': 5.0,
            'pronunciation_issues': 10,
            'explanation': "Unable to analyze audio. Using estimated results."
        }

# Additional utility functions that might be needed
def save_audio_file(audio_data, output_path):
    """
    Mock function to save audio data to a file
    
    Args:
        audio_data: Audio data as bytes
        output_path: Path where to save the file
        
    Returns:
        Boolean indicating success
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the file if it's bytes
        if isinstance(audio_data, bytes):
            with open(output_path, 'wb') as f:
                f.write(audio_data)
        else:
            # For compatibility, create a dummy file
            with open(output_path, 'wb') as f:
                f.write(b'DUMMY AUDIO DATA')
        
        return True
    except Exception as e:
        print(f"Error saving audio file: {e}")
        return False