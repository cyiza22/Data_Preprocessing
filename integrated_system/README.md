# Integrated Authentication & Prediction System

Multi-factor authentication system combining face recognition, voice verification, and product prediction.

## Authentication Flow

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

## Required Models

1. **face_recognition_model.pkl** - Face authentication
   - Location: `face_recognition_project/face_recognition_model.pkl`
   - Purpose: Identify and authorize users

2. **voice_model.pkl** - Voice verification
   - Location: `voiceprint_verification_model/models/voice_model.pkl`
   - Purpose: Verify voice samples

3. **best_model_XGBoost.pkl** - Product prediction
   - Location: Root directory
   - Purpose: Predict products for customers

##  Quick Start

### Installation

```bash
# Navigate to integrated system directory
cd integrated_system

# Install required packages (if not already installed)
pip install numpy pandas scikit-learn face_recognition librosa soundfile
```

### Run Demo

```bash
python demo.py
```

## System Components

### AuthenticatedPredictionSystem Class

**Methods:**

- `authenticate_face(image_path, known_encodings)` - Step 1: Face authentication
- `predict_product(customer_data)` - Step 2: Product prediction
- `verify_voice(audio_path)` - Step 3: Voice verification
- `complete_flow(...)` - Execute full authentication pipeline