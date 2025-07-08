import cv2
import numpy as np
import tensorflow as tf
import os

def main():
    # Check if model file exists
    model_path = 'trained_model_CNN1.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run Eye_movement_training.py first to create the model.")
        return
    
    # Load the trained model
    try:
        print(f"Loading model from: {os.path.abspath(model_path)}")
        eye_cnn = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load Haar cascade classifiers for face and eye detection
    try:
        # Adjust these paths if needed for your system
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        if face_cascade.empty() or eye_cascade.empty():
            print("Error: Haar cascade XML files couldn't be loaded.")
            print("Please check OpenCV installation and XML file paths.")
            return
    except Exception as e:
        print(f"Error loading cascade classifiers: {str(e)}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Define class labels (update these based on your training)
    class_labels = ["Normal", "Dyslexic"]  # Modify as needed based on your dataset
    
    print("Starting webcam capture. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # For each face, detect eyes and classify
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Region of interest for the face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                try:
                    # Detect eyes within the face region
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    
                    if len(eyes) == 0:
                        continue  # Skip if no eyes detected
                        
                    for (ex, ey, ew, eh) in eyes:
                        # Make sure eye region is valid
                        if ey < 0 or ex < 0 or ey+eh > roi_color.shape[0] or ex+ew > roi_color.shape[1]:
                            continue  # Skip invalid eye regions
                            
                        # Draw rectangle around eyes
                        cv2.rectangle(display_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                        
                        # Extract the eye image
                        eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                        
                        # Check if eye image is valid
                        if eye_img.size == 0 or eye_img.shape[0] == 0 or eye_img.shape[1] == 0:
                            continue  # Skip empty eye images
                        
                        # Resize to match the input size expected by the model
                        eye_img_resized = cv2.resize(eye_img, (128, 128))
                        
                        # Normalize the image
                        eye_img_normalized = eye_img_resized / 255.0
                        
                        # Reshape for model input
                        eye_img_input = np.expand_dims(eye_img_normalized, axis=0)
                        
                        # Make prediction
                        prediction = eye_cnn.predict(eye_img_input, verbose=0)
                        
                        if len(prediction) > 0 and len(prediction[0]) > 0:
                            predicted_class = np.argmax(prediction[0])
                            
                            # Safety check for class index
                            if predicted_class < len(class_labels):
                                confidence = prediction[0][predicted_class] * 100
                                
                                # Display the prediction
                                label = f"{class_labels[predicted_class]}: {confidence:.2f}%"
                                cv2.putText(display_frame, label, (x+ex, y+ey-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing eyes in face: {str(e)}")
                    continue  # Continue to next face if eye processing fails
                    
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
        
        # Display the resulting frame
        cv2.imshow('Eye Movement Analysis', display_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def run_main():
    """Function to be called from other modules"""
    main()

if __name__ == "__main__":
    main()