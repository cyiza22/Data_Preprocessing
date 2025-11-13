"""
Demo script for the Authenticated Prediction System
Shows how to use the system with your existing models
"""

import sys
import os
import pickle
import numpy as np
sys.path.append(os.path.dirname(__file__))

from auth_system import AuthenticatedPredictionSystem


def load_known_face_encodings():
    """
    Load the face encodings from your training data
    These are needed for the face distance calculation
    """
    try:
        # Load from your image_features.csv or training data
        import pandas as pd
        df = pd.read_csv('../face_recognition_project/image_features.csv')
        encodings = df[[f'feature_{i}' for i in range(128)]].values
        return encodings
    except Exception as e:
        print(f"Warning: Could not load face encodings: {e}")
        return []


def demo_single_user():
    """
    Demo: Single user attempting to get a prediction
    """
    print("\n" + "🎬 "*20)
    print("DEMO: SINGLE USER AUTHENTICATION FLOW")
    print("🎬 "*20 + "\n")
    
    # Initialize system
    system = AuthenticatedPredictionSystem(
        face_model_path='../face_recognition_project/face_recognition_model.pkl',
        voice_model_path='../voiceprint_verification_model/notebook/voice_model.pkl',
        product_model_path='../Recommendation_project/best_model_XGBoost.pkl',
        product_scaler_path='../Recommendation_project/product_category_scaler.pkl',
        product_artifacts_path='../Recommendation_project/product_category_artifacts.pkl'
    )
    
    # Load known encodings
    known_encodings = load_known_face_encodings()
    
    # Example user attempt
    print("\n📝 User Attempt Details:")
    print("-" * 70)
    print("Face Image: faces/Christian.jpg")
    print("Voice Sample: voice_samples/christian_voice.wav")
    print("Customer Features: Purchase history and engagement data")
    print("-" * 70)
    
    # Run complete flow with proper features for product category prediction
    # Features match the new product_category_prediction model (17 features)
    customer_features = {
        'purchase_amount': 450.0,
        'customer_rating': 4.5,
        'engagement_score': 85,
        'purchase_interest_score': 4.2,
        'platform_encoded': 2,  # e.g., Instagram
        'sentiment_encoded': 1,  # e.g., Positive
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
    
    result = system.complete_flow(
        image_path='../face_recognition_project/faces/Christian.jpg',
        customer_data=customer_features,
        audio_path='../voiceprint_verification_model/data/raw/audio/member 1/Yes_approve.opus',
        known_encodings=known_encodings
    )
    
    return result


def demo_multiple_scenarios():
    """
    Demo: Multiple scenarios (authorized, unauthorized, wrong voice)
    """
    print("\n" + "🎬 "*20)
    print("DEMO: MULTIPLE AUTHENTICATION SCENARIOS")
    print("🎬 "*20 + "\n")
    
    system = AuthenticatedPredictionSystem()
    known_encodings = load_known_face_encodings()
    
    scenarios = [
        {
            'name': 'Authorized User - Electronics Purchase',
            'image': '../face_recognition_project/faces/limpho-neutral.jpeg',
            'audio': '../voiceprint_verification_model/data/raw/audio/member 1/Yes_approve.opus',
            'data': {
                'purchase_amount': 335.0, 'customer_rating': 4.2, 'engagement_score': 88,
                'purchase_interest_score': 4.5, 'platform_encoded': 1, 'sentiment_encoded': 1,
                'purchase_month': 11, 'purchase_day_of_week': 2, 'purchase_quarter': 4,
                'avg_purchase': 310.0, 'total_spent': 1860.0, 'purchase_count': 6,
                'purchase_std': 95.0, 'avg_rating': 4.1, 'avg_engagement': 85.0, 'avg_interest': 4.3
            }
        },
        {
            'name': 'Unauthorized User - Face Fail',
            'image': '../face_recognition_project/test_images/download.jpeg',
            'audio': '../voiceprint_verification_model/data/raw/audio/member 1/Yes_approve.opus',
            'data': {
                'purchase_amount': 250.0, 'customer_rating': 3.5, 'engagement_score': 70,
                'purchase_interest_score': 3.2, 'platform_encoded': 0, 'sentiment_encoded': 0,
                'purchase_month': 10, 'purchase_day_of_week': 4, 'purchase_quarter': 4,
                'avg_purchase': 200.0, 'total_spent': 800.0, 'purchase_count': 4,
                'purchase_std': 50.0, 'avg_rating': 3.3, 'avg_engagement': 65.0, 'avg_interest': 3.0
            }
        },
        {
            'name': 'Authorized User - Voice Fail',
            'image': '../face_recognition_project/faces/orpheus-smiling.jpeg',
            'audio': '../voiceprint_verification_model/data/raw/audio/member 2/Confirm_transaction.opus',
            'data': {
                'purchase_amount': 180.0, 'customer_rating': 4.0, 'engagement_score': 75,
                'purchase_interest_score': 3.8, 'platform_encoded': 3, 'sentiment_encoded': 2,
                'purchase_month': 11, 'purchase_day_of_week': 1, 'purchase_quarter': 4,
                'avg_purchase': 160.0, 'total_spent': 960.0, 'purchase_count': 6,
                'purchase_std': 40.0, 'avg_rating': 3.9, 'avg_engagement': 72.0, 'avg_interest': 3.6
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print('='*70)
        
        result = system.complete_flow(
            image_path=scenario['image'],
            customer_data=scenario['data'],
            audio_path=scenario['audio'],
            known_encodings=known_encodings
        )
        
        results.append({
            'scenario': scenario['name'],
            'success': result['success'],
            'user': result['user'],
            'prediction': result['prediction']
        })
        
        input("\nPress Enter to continue to next scenario...")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY OF ALL SCENARIOS")
    print("="*70)
    for i, r in enumerate(results, 1):
        status = "✅ APPROVED" if r['success'] else "❌ DENIED"
        print(f"\n{i}. {r['scenario']}")
        print(f"   Status: {status}")
        if r['user']:
            print(f"   User: {r['user']}")
        if r['prediction']:
            print(f"   Prediction: {r['prediction']}")
    print("="*70)


def demo_api_style():
    """
    Demo: API-style usage for integration
    """
    print("\n" + "🎬 "*20)
    print("DEMO: API-STYLE INTEGRATION EXAMPLE")
    print("🎬 "*20 + "\n")
    
    # Initialize once
    system = AuthenticatedPredictionSystem()
    known_encodings = load_known_face_encodings()
    
    # Simulate API request
    request_data = {
        'user_image': '../face_recognition_project/faces/Christian.jpg',
        'voice_sample': '../voiceprint_verification_model/data/raw/audio/member 1/Yes_approve.opus',
        'prediction_input': {
            'purchase_amount': 420.0,
            'customer_rating': 4.7,
            'engagement_score': 90,
            'purchase_interest_score': 4.6,
            'platform_encoded': 2,
            'sentiment_encoded': 1,
            'purchase_month': 11,
            'purchase_day_of_week': 3,
            'purchase_quarter': 4,
            'avg_purchase': 390.0,
            'total_spent': 2340.0,
            'purchase_count': 6,
            'purchase_std': 110.0,
            'avg_rating': 4.5,
            'avg_engagement': 88.0,
            'avg_interest': 4.4
        }
    }
    
    print("📥 Incoming Request:")
    print("-" * 70)
    for key, value in request_data.items():
        print(f"  {key}: {value}")
    print("-" * 70)
    
    # Process request
    result = system.complete_flow(
        image_path=request_data['user_image'],
        customer_data=request_data['prediction_input'],
        audio_path=request_data['voice_sample'],
        known_encodings=known_encodings
    )
    
    # Format response
    response = {
        'status': 'success' if result['success'] else 'denied',
        'user': result['user'],
        'prediction': result['prediction'],
        'authentication': {
            'face': bool(result['stages']['face_auth']),
            'voice': bool(result['stages']['voice_verify'])
        }
    }
    
    print("\n📤 API Response:")
    print("-" * 70)
    import json
    print(json.dumps(response, indent=2))
    print("-" * 70)
    
    return response


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   AUTHENTICATED PRODUCT PREDICTION SYSTEM - DEMO                ║
    ║   Multi-Factor Authentication with Face & Voice Recognition     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run API-Style Integration demo only
    demo_api_style()
    
    print("\n✅ Demo completed!")
