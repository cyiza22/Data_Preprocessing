# Integrated Authentication & Prediction System

Multi-factor authentication system combining face recognition, voice verification, and product prediction.

## 🔐 Authentication Flow

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────────────────┐
│ Face Recognition    │ ──❌ Fail──► Access Denied
│ Authentication      │
└────┬────────────────┘
     │ ✅ Pass
     ▼
┌─────────────────────┐
│ Product Prediction  │
│ Model Execution     │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│ Voice Verification  │ ──❌ Fail──► Prediction Rejected
│ Confirmation        │
└────┬────────────────┘
     │ ✅ Pass
     ▼
┌─────────────────────┐
│ Display Predicted   │
│ Product (Approved)  │
└─────────────────────┘
```

## 📦 Required Models

1. **face_recognition_model.pkl** - Face authentication
   - Location: `face_recognition_project/face_recognition_model.pkl`
   - Purpose: Identify and authorize users

2. **voice_model.pkl** - Voice verification
   - Location: `voiceprint_verification_model/models/voice_model.pkl`
   - Purpose: Verify voice samples

3. **best_model_XGBoost.pkl** - Product prediction
   - Location: Root directory
   - Purpose: Predict products for customers

## 🚀 Quick Start

### Installation

```bash
# Navigate to integrated system directory
cd integrated_system

# Install required packages (if not already installed)
pip install numpy pandas scikit-learn face_recognition librosa soundfile
```

### Basic Usage

```python
from auth_system import AuthenticatedPredictionSystem

# Initialize system
system = AuthenticatedPredictionSystem(
    face_model_path='../face_recognition_project/face_recognition_model.pkl',
    voice_model_path='../voiceprint_verification_model/models/voice_model.pkl',
    product_model_path='../best_model_XGBoost.pkl'
)

# Load known face encodings
import pandas as pd
df = pd.read_csv('../face_recognition_project/image_features.csv')
known_encodings = df[[f'feature_{i}' for i in range(128)]].values

# Run complete authentication flow
result = system.complete_flow(
    image_path='path/to/user/face.jpg',
    customer_data=[299.99, 5000.0, 30, 2, 0],  # [product_price_mean, user_total_spend, days_since_last_purchase, category_id, clicked]
    audio_path='path/to/voice/sample.wav',
    known_encodings=known_encodings
)

# Check result
if result['success']:
    print(f"User: {result['user']}")
    print(f"Predicted Product: {result['prediction']}")
else:
    print("Authentication failed")
```

### Run Demo

```bash
python demo.py
```

## 📋 System Components

### AuthenticatedPredictionSystem Class

**Methods:**

- `authenticate_face(image_path, known_encodings)` - Step 1: Face authentication
- `predict_product(customer_data)` - Step 2: Product prediction
- `verify_voice(audio_path)` - Step 3: Voice verification
- `complete_flow(...)` - Execute full authentication pipeline

**Configuration:**

```python
# Adjust thresholds in __init__ or after initialization
system.FACE_CONFIDENCE_THRESHOLD = 0.60  # Face recognition confidence
system.FACE_DISTANCE_THRESHOLD = 0.6     # Face distance threshold
system.VOICE_THRESHOLD = 0.5             # Voice verification threshold
```

## 🔧 Integration Examples

### Example 1: Web API Integration

```python
from flask import Flask, request, jsonify
from auth_system import AuthenticatedPredictionSystem

app = Flask(__name__)
system = AuthenticatedPredictionSystem()

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded files
    face_image = request.files['face']
    voice_audio = request.files['voice']
    customer_data = request.json['data']  # [product_price_mean, user_total_spend, days_since_last_purchase, category_id, clicked]
    
    # Save temporarily
    face_image.save('temp_face.jpg')
    voice_audio.save('temp_voice.wav')
    
    # Run authentication
    result = system.complete_flow(
        image_path='temp_face.jpg',
        customer_data=customer_data,
        audio_path='temp_voice.wav',
        known_encodings=known_encodings
    )
    
    return jsonify({
        'success': result['success'],
        'user': result['user'],
        'prediction': result['prediction']
    })
```

### Example 2: Command Line Tool

```python
import sys
from auth_system import AuthenticatedPredictionSystem

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_prediction.py <face_image> <voice_audio> <customer_data>")
        print("customer_data format: [product_price_mean,user_total_spend,days_since_last_purchase,category_id,clicked]")
        return
    
    system = AuthenticatedPredictionSystem()
    
    result = system.complete_flow(
        image_path=sys.argv[1],
        customer_data=eval(sys.argv[3]),  # e.g., [299.99, 5000.0, 30, 2, 0]
        audio_path=sys.argv[2],
        known_encodings=load_known_encodings()
    )
    
    if result['success']:
        print(f"✅ Prediction: {result['prediction']}")
    else:
        print("❌ Access Denied")

if __name__ == "__main__":
    main()
```

### Example 3: Batch Processing

```python
import pandas as pd
from auth_system import AuthenticatedPredictionSystem

def batch_process(requests_csv):
    """
    Process multiple prediction requests from CSV
    CSV columns: user_id, face_image, voice_audio, product_price_mean, user_total_spend, days_since_last_purchase, category_id, clicked
    """
    system = AuthenticatedPredictionSystem()
    df = pd.read_csv(requests_csv)
    results = []
    
    for idx, row in df.iterrows():
        result = system.complete_flow(
            image_path=row['face_image'],
            customer_data=[row['product_price_mean'], row['user_total_spend'], 
                          row['days_since_last_purchase'], row['category_id'], row['clicked']],
            audio_path=row['voice_audio'],
            known_encodings=known_encodings
        )
        
        results.append({
            'user_id': row['user_id'],
            'success': result['success'],
            'prediction': result['prediction']
        })
    
    return pd.DataFrame(results)
```

## 🔍 Troubleshooting

### Common Issues

**1. Face not detected**
- Ensure image has good lighting
- Face should be clearly visible and front-facing
- Supported formats: JPG, PNG, JPEG

**2. Voice verification fails**
- Check audio quality (16kHz recommended)
- Ensure audio is at least 1-2 seconds long
- Supported formats: WAV, OPUS, MP3

**3. Models not loading**
- Verify all model paths are correct
- Check that pkl files are not corrupted
- Ensure you have read permissions

**4. Import errors**
```bash
# Install missing packages
pip install face_recognition librosa soundfile
# For macOS with HEIC support
pip install pillow-heif
```

## 🎯 Testing

Test the system with known users:

```python
# Test with authorized users
test_cases = [
    {
        'name': 'Christian',
        'image': '../face_recognition_project/faces/Christian.jpg',
        'audio': '../voiceprint_verification_model/data/raw/audio/member 1/Approve_yes.opus',
        'data': [299.99, 5000.0, 30, 2, 0]  # [product_price_mean, user_total_spend, days_since_last_purchase, category_id, clicked]
    },
    {
        'name': 'Limpho',
        'image': '../face_recognition_project/faces/limpho-neutral.jpeg',
        'audio': '../voiceprint_verification_model/data/raw/audio/member 1/Approve_yes.opus',
        'data': [199.99, 3500.0, 15, 1, 0]
    }
]

for test in test_cases:
    print(f"\nTesting {test['name']}...")
    result = system.complete_flow(
        image_path=test['image'],
        customer_data=test['data'],
        audio_path=test['audio'],
        known_encodings=known_encodings
    )
    print(f"Result: {'✅ PASS' if result['success'] else '❌ FAIL'}")
```

## 📊 Expected Output

```
======================================================================
🚀 STARTING AUTHENTICATED PREDICTION FLOW
======================================================================

============================================================
STEP 1: FACE RECOGNITION AUTHENTICATION
============================================================
👤 Detected User: Christian
📊 Confidence: 89.45%
📏 Distance: 0.432
✅ ACCESS GRANTED - User authenticated as Christian

============================================================
STEP 2: PRODUCT PREDICTION
============================================================
🎯 Predicted Product: Premium Laptop
📊 Prediction Confidence: 78.23%

============================================================
STEP 3: VOICE VERIFICATION
============================================================
🎤 Voice Verification: Approved
📊 Confidence: 85.67%
✅ VOICE VERIFIED - Prediction confirmed

======================================================================
📋 FINAL RESULT
======================================================================
✅ ALL CHECKS PASSED
👤 Authorized User: Christian
🎯 Predicted Product: Premium Laptop
🔒 Prediction APPROVED and CONFIRMED
======================================================================
```

## 🔐 Security Considerations

1. **Threshold Tuning**: Adjust thresholds based on security requirements
2. **Rate Limiting**: Implement request limits to prevent abuse
3. **Logging**: Log all authentication attempts
4. **Model Updates**: Regularly retrain models with new data
5. **Encryption**: Encrypt stored biometric data

## 📝 License

This system is part of the Data_Preprocessing project.
