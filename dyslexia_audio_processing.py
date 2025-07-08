# audio_model.py
import os
import numpy as np
import pandas as pd
import librosa
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DyslexiaAudioModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean',
            'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean', 'mfcc9_mean', 'mfcc10_mean',
            'mfcc11_mean', 'mfcc12_mean', 'mfcc13_mean',
            'mfcc1_std', 'mfcc2_std', 'mfcc3_std', 'mfcc4_std', 'mfcc5_std',
            'mfcc6_std', 'mfcc7_std', 'mfcc8_std', 'mfcc9_std', 'mfcc10_std',
            'mfcc11_std', 'mfcc12_std', 'mfcc13_std',
            'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std', 'spectral_mean', 'spectral_std'
        ]
    
    def extract_features(self, audio_path):
        """Extract audio features from a single file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract features
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Root Mean Square Energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y=y)[0]
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            
            # Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_mean = np.mean(spectral_centroids)
            spectral_std = np.std(spectral_centroids)
            
            # Combine all features
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                [rms_mean, rms_std, zcr_mean, zcr_std, spectral_mean, spectral_std]
            ])
            
            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def process_dataset(self, base_dir):
        """Process the entire dataset and extract features"""
        features_list = []
        labels = []
        file_paths = []
        
        # Process dyslexic participants
        dyslexic_dir = os.path.join(base_dir, 'dyslexic')
        for gender_folder in ['F_Dys', 'M_Dys']:
            gender_path = os.path.join(dyslexic_dir, gender_folder)
            if not os.path.exists(gender_path):
                continue
                
            # Process each recording type folder (wav_arrayMic, wav_headMic, etc.)
            for root, dirs, files in os.walk(gender_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        features = self.extract_features(file_path)
                        if features is not None:
                            features_list.append(features)
                            labels.append(1)  # 1 for dyslexic
                            file_paths.append(file_path)
        
        # Process non-dyslexic participants
        non_dyslexic_dir = os.path.join(base_dir, 'non_dyslexic')
        for gender_folder in ['F_Con', 'M_Con']:
            gender_path = os.path.join(non_dyslexic_dir, gender_folder)
            if not os.path.exists(gender_path):
                continue
                
            # Process each recording type folder
            for root, dirs, files in os.walk(gender_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        features = self.extract_features(file_path)
                        if features is not None:
                            features_list.append(features)
                            labels.append(0)  # 0 for control
                            file_paths.append(file_path)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        return X, y, file_paths
    
    def train(self, X, y):
        """Train the model on the processed dataset"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Control', 'Dyslexic'])
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(report)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 10 most important features:")
        print(feature_importance.head(10))
        
        return accuracy, report
    
    def predict_single(self, audio_path):
        """Predict dyslexia for a single audio sample"""
        if not self.model or not self.scaler:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'label': 'Dyslexic' if prediction == 1 else 'Non-dyslexic',
            'probability': float(probability[prediction])
        }
        
        return result
    
    def save_model(self, model_path='models/audio_model.pkl', scaler_path='models/audio_scaler.pkl'):
        """Save the trained model and scaler"""
        if not self.model or not self.scaler:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='models/audio_model.pkl', scaler_path='models/audio_scaler.pkl'):
        """Load a trained model and scaler"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("Model and scaler loaded successfully.")

# Script to train and save the model
if __name__ == "__main__":
    # Path to your dataset
    dataset_path = "audio_dataset"
    
    # Initialize model
    audio_model = DyslexiaAudioModel()
    
    # Process dataset
    print("Processing dataset...")
    X, y, file_paths = audio_model.process_dataset(dataset_path)
    print(f"Dataset processed: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Dyslexic samples: {np.sum(y)}, Control samples: {len(y) - np.sum(y)}")
    
    # Train model
    print("Training model...")
    accuracy, _ = audio_model.train(X, y)
    
    # Save model
    audio_model.save_model()
    
    print(f"Model training complete with accuracy: {accuracy:.4f}")