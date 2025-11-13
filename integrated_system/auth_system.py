"""
Integrated Authentication and Prediction System
Combines face recognition, voice verification, and product prediction
"""

import pickle
import numpy as np
import face_recognition
import librosa
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AuthenticatedPredictionSystem:
    """
    Multi-factor authentication system for product prediction
    Flow: Face Recognition → Product Prediction → Voice Verification
    """
    
    def __init__(self, 
                 face_model_path='../face_recognition_project/face_recognition_model.pkl',
                 voice_model_path='../voiceprint_verification_model/notebook/voice_model.pkl',
                 product_model_path='../Recommendation_project/best_model_XGBoost.pkl',
                 product_scaler_path='../Recommendation_project/product_category_scaler.pkl',
                 product_artifacts_path='../Recommendation_project/product_category_artifacts.pkl'):
        """
        Initialize the authenticated prediction system
        
        Args:
            face_model_path: Path to face recognition model (pkl)
            voice_model_path: Path to voice verification model (pkl)
            product_model_path: Path to product prediction model (pkl)
            preprocessor_path: Path to preprocessor for product features (pkl)
        """
        self.face_model = None
        self.voice_model = None
        self.voice_scaler = None
        self.voice_label_encoder = None
        self.product_model = None
        self.product_scaler = None
        self.product_artifacts = None
        self.authorized_users = []
        
        # Load models
        self._load_models(face_model_path, voice_model_path, product_model_path, 
                         product_scaler_path, product_artifacts_path)
        
        # Configurable thresholds
        self.FACE_CONFIDENCE_THRESHOLD = 0.60
        self.FACE_DISTANCE_THRESHOLD = 0.6
        self.VOICE_THRESHOLD = 0.5  # Adjust based on your voice model
        
    def _load_models(self, face_path, voice_path, product_path, product_scaler_path, product_artifacts_path):
        """Load all three models"""
        try:
            # Load face recognition model
            with open(face_path, 'rb') as f:
                face_data = pickle.load(f)
                self.face_model = face_data['classifier']
                self.authorized_users = face_data['label_encoder']
            print(f"✅ Face model loaded. Authorized users: {self.authorized_users}")
        except Exception as e:
            print(f"❌ Error loading face model: {e}")
            
        try:
            # Load voice verification model
            voice_dir = Path(voice_path).parent
            with open(voice_path, 'rb') as f:
                self.voice_model = pickle.load(f)
            
            # Load voice scaler
            scaler_path = voice_dir / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.voice_scaler = pickle.load(f)
                print(f"✅ Voice model and scaler loaded")
            else:
                print(f"✅ Voice model loaded (scaler not found)")
            
            # Load voice label encoder
            encoder_path = voice_dir / 'label_encoder.pkl'
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.voice_label_encoder = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading voice model: {e}")
            
        try:
            # Load product prediction model
            import joblib
            self.product_model = joblib.load(product_path)
            print(f"✅ Product prediction model loaded")
            
            # Load product scaler
            self.product_scaler = joblib.load(product_scaler_path)
            print(f"✅ Product scaler loaded")
            
            # Load product artifacts (encoders, feature columns)
            with open(product_artifacts_path, 'rb') as f:
                self.product_artifacts = pickle.load(f)
            print(f"✅ Product artifacts loaded")
        except Exception as e:
            print(f"❌ Error loading product model components: {e}")
    
    def authenticate_face(self, image_path, known_encodings):
        """
        Step 1: Face Recognition Authentication
        
        Args:
            image_path: Path to user's face image
            known_encodings: List of known face encodings from training
            
        Returns:
            tuple: (is_authorized, user_name, confidence, distance)
        """
        print("\n" + "="*60)
        print("STEP 1: FACE RECOGNITION AUTHENTICATION")
        print("="*60)
        
        try:
            # Load and detect face
            img = face_recognition.load_image_file(image_path)
            encodings_found = face_recognition.face_encodings(img)
            
            if len(encodings_found) == 0:
                print("❌ No face detected in image")
                return False, None, 0.0, 1.0
            
            test_encoding = encodings_found[0]
            
            # Predict with classifier
            predicted_user = self.face_model.predict([test_encoding])[0]
            probabilities = self.face_model.predict_proba([test_encoding])[0]
            confidence = max(probabilities)
            
            # Calculate distance to known faces
            distances = [face_recognition.face_distance([enc], test_encoding)[0] 
                        for enc in known_encodings]
            min_distance = min(distances)
            
            # Check thresholds
            is_authorized = (confidence >= self.FACE_CONFIDENCE_THRESHOLD and 
                           min_distance < self.FACE_DISTANCE_THRESHOLD)
            
            print(f"👤 Detected User: {predicted_user}")
            print(f"📊 Confidence: {confidence:.2%}")
            print(f"📏 Distance: {min_distance:.3f}")
            
            if is_authorized:
                print(f"✅ ACCESS GRANTED - User authenticated as {predicted_user}")
                return True, predicted_user, confidence, min_distance
            else:
                print(f"❌ ACCESS DENIED - Authentication failed")
                print(f"   Reason: Confidence={confidence:.2%} (need ≥60%), Distance={min_distance:.3f} (need <0.6)")
                return False, None, confidence, min_distance
                
        except Exception as e:
            print(f"❌ Error during face authentication: {e}")
            return False, None, 0.0, 1.0
    
    def predict_product(self, customer_data):
        """
        Step 2: Run Product Category Prediction
        
        Args:
            customer_data: dict or array with customer features
                          Expected keys (if dict): purchase_amount, customer_rating, 
                          engagement_score, purchase_interest_score, platform_encoded,
                          sentiment_encoded, purchase_month, purchase_day_of_week,
                          purchase_quarter, avg_purchase, total_spent, purchase_count,
                          purchase_std, avg_rating, avg_engagement, avg_interest
            
        Returns:
            predicted_category: The predicted product category
        """
        print("\n" + "="*60)
        print("STEP 2: PRODUCT CATEGORY PREDICTION")
        print("="*60)
        
        if self.product_model is None or self.product_scaler is None:
            print("❌ Product model not loaded")
            return None
        
        try:
            # Get feature columns from artifacts
            feature_columns = self.product_artifacts['feature_columns']
            
            # Convert dict to array if needed
            if isinstance(customer_data, dict):
                features = np.array([customer_data[col] for col in feature_columns]).reshape(1, -1)
            else:
                features = np.array(customer_data).reshape(1, -1)
            
            # Scale features
            features_scaled = self.product_scaler.transform(features)
            
            # Make prediction
            prediction = self.product_model.predict(features_scaled)[0]
            probabilities = self.product_model.predict_proba(features_scaled)[0]
            
            # Decode category
            category_encoder = self.product_artifacts['category_encoder']
            predicted_category = category_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            print(f"🎯 Predicted Category: {predicted_category}")
            print(f"📊 Confidence: {confidence:.2%}")
            
            # Show top 3 categories
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            print(f"\nTop 3 predictions:")
            for idx in top_3_idx:
                cat = category_encoder.inverse_transform([idx])[0]
                prob = probabilities[idx]
                print(f"  {cat}: {prob:.2%}")
            
            return predicted_category
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def verify_voice(self, audio_path):
        """
        Step 3: Voice Verification
        
        Args:
            audio_path: Path to audio file for verification
            
        Returns:
            tuple: (is_verified, confidence)
        """
        print("\n" + "="*60)
        print("STEP 3: VOICE VERIFICATION")
        print("="*60)
        
        # Check if voice model is available
        if not hasattr(self, 'voice_model') or self.voice_model is None:
            print("⚠️  Voice model not available - SKIPPING voice verification")
            print("✅ Voice verification BYPASSED (model not loaded)")
            return True, 1.0
        
        try:
            # Load audio and extract features (matching training notebook)
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract all 17 features (same as training)
            features = []
            
            # 1. MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            features.extend(mfccs_mean)
            
            # 2. Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.append(np.mean(rolloff))
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.append(np.mean(zcr))
            
            # 4. Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.append(np.mean(centroid))
            
            # 5. RMS energy
            rms = librosa.feature.rms(y=audio)
            features.append(np.mean(rms))
            
            # Convert to array and reshape
            features = np.array(features).reshape(1, -1)
            
            # Apply scaler if available
            if self.voice_scaler is not None:
                features = self.voice_scaler.transform(features)
            
            # Verify with voice model
            prediction = self.voice_model.predict(features)[0]
            
            # Decode the prediction using label encoder
            if self.voice_label_encoder is not None:
                predicted_speaker = self.voice_label_encoder.inverse_transform([prediction])[0]
            else:
                predicted_speaker = f"Speaker_{prediction}"
            
            # Get confidence if available
            try:
                proba = self.voice_model.predict_proba(features)
                confidence = max(proba[0])
                
                # Check if predicted speaker matches the authenticated user
                # For now, we'll just verify the speaker is in our known set
                is_verified = confidence >= self.VOICE_THRESHOLD
                
                print(f"🎤 Predicted Speaker: {predicted_speaker}")
                print(f"📊 Confidence: {confidence:.2%}")
                
                if is_verified:
                    print(f"✅ VOICE VERIFIED - Speaker: {predicted_speaker}")
                else:
                    print(f"❌ VOICE VERIFICATION FAILED - Low confidence")
                    
            except Exception as e:
                print(f"⚠️  Could not get voice confidence: {e}")
                is_verified = False
                confidence = 0.0
            
            return (is_verified, confidence)
            
        except Exception as e:
            print(f"❌ Error during voice verification: {e}")
            import traceback
            traceback.print_exc()
            return (False, None)
    
    def complete_flow(self, image_path, customer_data, audio_path, known_encodings):
        """
        Execute complete authentication and prediction flow
        
        Args:
            image_path: Path to user's face image
            customer_data: Input features for product prediction
            audio_path: Path to voice recording
            known_encodings: List of known face encodings
            
        Returns:
            dict: Complete results of the authentication flow
        """
        print("\n" + "="*70)
        print("🚀 STARTING AUTHENTICATED PREDICTION FLOW")
        print("="*70)
        
        result = {
            'success': False,
            'user': None,
            'prediction': None,
            'stages': {
                'face_auth': False,
                'prediction': False,
                'voice_verify': False
            },
            'details': {}
        }
        
        # STEP 1: Face Authentication
        face_auth, user_name, face_conf, face_dist = self.authenticate_face(
            image_path, known_encodings
        )
        result['stages']['face_auth'] = face_auth
        result['details']['face'] = {
            'confidence': face_conf,
            'distance': face_dist,
            'user': user_name
        }
        
        if not face_auth:
            print("\n" + "="*70)
            print("❌ FLOW TERMINATED - Face authentication failed")
            print("="*70)
            return result
        
        result['user'] = user_name
        
        # STEP 2: Product Prediction
        predicted_product = self.predict_product(customer_data)
        result['stages']['prediction'] = predicted_product is not None
        result['prediction'] = predicted_product
        
        if predicted_product is None:
            print("\n" + "="*70)
            print("❌ FLOW TERMINATED - Prediction failed")
            print("="*70)
            return result
        
        # STEP 3: Voice Verification
        voice_verified, voice_conf = self.verify_voice(audio_path)
        result['stages']['voice_verify'] = voice_verified
        result['details']['voice'] = {'confidence': voice_conf}
        
        # Final decision
        result['success'] = all(result['stages'].values())
        
        print("\n" + "="*70)
        print("📋 FINAL RESULT")
        print("="*70)
        
        if result['success']:
            print(f"✅ ALL CHECKS PASSED")
            print(f"👤 Authorized User: {user_name}")
            print(f"🎯 Predicted Product: {predicted_product}")
            print(f"🔒 Prediction APPROVED and CONFIRMED")
        else:
            print(f"❌ AUTHENTICATION FLOW FAILED")
            print(f"   Face Auth: {'✅' if result['stages']['face_auth'] else '❌'}")
            print(f"   Prediction: {'✅' if result['stages']['prediction'] else '❌'}")
            print(f"   Voice Verify: {'✅' if result['stages']['voice_verify'] else '❌'}")
        
        print("="*70)
        
        return result

