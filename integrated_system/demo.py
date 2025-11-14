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


def demo_api_style():
    """
    Demo: API-style usage for integration 
    """
  
    print("DEMO: API-STYLE INTEGRATION EXAMPLE")
    
    # Initialize once
    system = AuthenticatedPredictionSystem()
    known_encodings = load_known_face_encodings()
    
    # Simulate API request
    request_data = {
        'user_image': '../face_recognition_project/faces/henriette_normal.jpeg',
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
    
    print("Incoming Request:")
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
    
    print("\n API Response:")
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
