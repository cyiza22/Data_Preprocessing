"""
Simple test script for the Authenticated Prediction System
Tests all three models without interactive prompts
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from auth_system import AuthenticatedPredictionSystem
import pandas as pd
import numpy as np

def load_known_face_encodings():
    """Load face encodings from training data"""
    try:
        df = pd.read_csv('../face_recognition_project/image_features.csv')
        encodings = df[[f'feature_{i}' for i in range(128)]].values
        return encodings
    except Exception as e:
        print(f"Warning: Could not load face encodings: {e}")
        return []


def test_single_user():
    """Test with a single user"""
    print("\n" + "="*70)
    print("🧪 TESTING INTEGRATED AUTHENTICATION SYSTEM")
    print("="*70)
    
    # Initialize system
    print("\n📦 Loading models...")
    system = AuthenticatedPredictionSystem(
        face_model_path='../face_recognition_project/face_recognition_model.pkl',
        voice_model_path='../voiceprint_verification_model/notebook/voice_model.pkl',
        product_model_path='../Recommendation_project/best_model_XGBoost.pkl',
        product_scaler_path='../Recommendation_project/product_category_scaler.pkl',
        product_artifacts_path='../Recommendation_project/product_category_artifacts.pkl'
    )
    
    # Load known encodings
    known_encodings = load_known_face_encodings()
    print(f"\n✅ Loaded {len(known_encodings)} known face encodings")
    
    # Test data
    print("\n📋 Test Configuration:")
    print("-" * 70)
    print("Face Image: ../face_recognition_project/faces/Christian.jpg")
    print("Voice Sample: ../voiceprint_verification_model/data/raw/audio/member 1/Yes_approve.opus")
    print("Customer Profile: Electronics purchase pattern")
    print("-" * 70)
    
    # Customer features matching the product category model (17 features)
    customer_features = {
        'purchase_amount': 450.0,
        'customer_rating': 4.5,
        'engagement_score': 85,
        'purchase_interest_score': 4.2,
        'platform_encoded': 2,  # Instagram
        'sentiment_encoded': 1,  # Positive
        'purchase_month': 11,
        'purchase_day_of_week': 3,
        'purchase_quarter': 4,
        'avg_purchase': 380.0,
        'total_spent': 2280.0,
        'purchase_count': 6,
        'purchase_std': 120.5,
        'avg_rating': 4.3,
        'avg_engagement': 82.0,
        'avg_interest': 3.9
    }
    
    # Run complete authentication flow
    result = system.complete_flow(
        image_path='../face_recognition_project/faces/Christian.jpg',
        customer_data=customer_features,
        audio_path='../voiceprint_verification_model/data/raw/audio/member 1/Yes_approve.opus',
        known_encodings=known_encodings
    )
    
    # Display results
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    print(f"Overall Success: {'✅ PASSED' if result['success'] else '❌ FAILED'}")
    print(f"\nStage Results:")
    print(f"  Face Authentication: {'✅' if result['stages']['face_auth'] else '❌'}")
    print(f"  Product Prediction: {'✅' if result['stages']['prediction'] else '❌'}")
    print(f"  Voice Verification: {'✅' if result['stages']['voice_verify'] else '❌'}")
    
    if result['user']:
        print(f"\nIdentified User: {result['user']}")
    if result['prediction']:
        print(f"Predicted Category: {result['prediction']}")
    
    print("\nDetailed Metrics:")
    if 'face' in result['details']:
        face = result['details']['face']
        print(f"  Face Confidence: {face['confidence']:.2%}")
        print(f"  Face Distance: {face['distance']:.3f}")
    if 'voice' in result['details']:
        voice = result['details']['voice']
        if voice['confidence'] is not None:
            print(f"  Voice Confidence: {voice['confidence']:.2%}")
        else:
            print(f"  Voice Confidence: N/A (verification failed)")
    
    print("="*70)
    
    return result


def test_each_component_separately():
    """Test each component individually"""
    print("\n" + "="*70)
    print("🔬 TESTING INDIVIDUAL COMPONENTS")
    print("="*70)
    
    system = AuthenticatedPredictionSystem(
        face_model_path='../face_recognition_project/face_recognition_model.pkl',
        voice_model_path='../voiceprint_verification_model/notebook/voice_model.pkl',
        product_model_path='../Recommendation_project/best_model_XGBoost.pkl',
        product_scaler_path='../Recommendation_project/product_category_scaler.pkl',
        product_artifacts_path='../Recommendation_project/product_category_artifacts.pkl'
    )
    
    known_encodings = load_known_face_encodings()
    
    # Test 1: Face Recognition Only
    print("\n\n" + "="*70)
    print("TEST 1: Face Recognition")
    print("="*70)
    face_result = system.authenticate_face(
        '../face_recognition_project/faces/Christian.jpg',
        known_encodings
    )
    print(f"Result: {'✅ PASSED' if face_result[0] else '❌ FAILED'}")
    
    # Test 2: Product Prediction Only
    print("\n\n" + "="*70)
    print("TEST 2: Product Category Prediction")
    print("="*70)
    customer_data = {
        'purchase_amount': 450.0,
        'customer_rating': 4.5,
        'engagement_score': 85,
        'purchase_interest_score': 4.2,
        'platform_encoded': 2,
        'sentiment_encoded': 1,
        'purchase_month': 11,
        'purchase_day_of_week': 3,
        'purchase_quarter': 4,
        'avg_purchase': 380.0,
        'total_spent': 2280.0,
        'purchase_count': 6,
        'purchase_std': 120.5,
        'avg_rating': 4.3,
        'avg_engagement': 82.0,
        'avg_interest': 3.9
    }
    prediction = system.predict_product(customer_data)
    print(f"Result: {'✅ PASSED' if prediction else '❌ FAILED'}")
    
    # Test 3: Voice Verification Only
    print("\n\n" + "="*70)
    print("TEST 3: Voice Verification")
    print("="*70)
    voice_result = system.verify_voice(
        '../voiceprint_verification_model/data/raw/audio/member 1/Yes_approve.opus'
    )
    print(f"Result: {'✅ PASSED' if voice_result[0] else '❌ FAILED'}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        AUTOMATED TEST SUITE - AUTHENTICATION SYSTEM              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    try:
        # Test complete flow
        result = test_single_user()
        
        # Test individual components
        test_each_component_separately()
        
        print("\n✅ All tests completed!\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
