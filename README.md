# Multi-Factor Authentication System with Product Recommendation

A comprehensive authentication system that combines **Face Recognition**, **Voice Verification**, and **Product Category Prediction** using machine learning models.

---

## What This Project Does

This system provides a **3-stage authentication and prediction pipeline**:

1. **Face Recognition** - Identifies users from facial images using SVM classifier
2. **Product Category Prediction** - Predicts product categories (Electronics, Clothing, Sports, Books, Groceries) based on customer behavior
3. **Voice Verification** - Authenticates users through voice biometrics using Random Forest classifier

All three stages must pass for complete authentication and prediction approval.

---

##  Project Structure

```
Data_Preprocessing/
├── integrated_system/              # Main authentication system
│   ├── auth_system.py              # Core authentication logic
│   ├── demo.py                     # API-style demo
│   ├── test_system.py              # Automated testing suite
│   └── README.md                   # Integration documentation
│
├── face_recognition_project/       # Face recognition model
│   ├── face_recognition_model.pkl  # Trained SVM model
│   ├── facial_recognition.ipynb    # Training notebook
│   ├── image_features.csv          # Face encodings dataset
│   └── faces/                      # Training images
│
├── voiceprint_verification_model/  # Voice verification model
│   ├── notebook/
│   │   ├── voice_model.pkl         # Trained Random Forest model
│   │   ├── scaler.pkl              # Feature scaler
│   │   ├── label_encoder.pkl       # Label encoder
│   │   └── voiceprint_training.ipynb  # Training notebook
│   └── data/raw/audio/             # Audio samples
│
├── Recommendation_project/         # Product prediction model
│   ├── best_model_XGBoost.pkl      # Trained XGBoost model
│   ├── product_category_scaler.pkl # Feature scaler
│   ├── product_category_artifacts.pkl  # Encoders & metadata
│   └── product_category_prediction.ipynb  # Training notebook
│
└── data/
    └── cleaned_customer_data.csv   # Customer transaction data
```

---

## Quick Start

### Prerequisites

- **Python 3.11** or higher
- **Conda** (recommended) or virtualenv

### Installation

1. **Clone the repository**
   ```bash
   cd /Data_Preprocessing
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n facial_rec python=3.11
   conda activate facial_rec
   ```

3. **Install required packages**
   ```bash
   # Face recognition dependencies
   pip install face-recognition pillow pillow-heif numpy pandas scikit-learn

   # Voice verification dependencies
   pip install librosa soundfile audioread

   # Product prediction dependencies
   pip install xgboost joblib

   # General ML packages
   pip install matplotlib seaborn jupyter
   ```

---

## How to Run

### Option 1: Run Complete System Demo (Recommended)

```bash
cd integrated_system
conda activate facial_rec
python demo.py
```

**What it does:**
- Loads all three trained models
- Runs API-style integration demo
- Shows complete authentication flow with results

**Expected Output:**
```
✅ Face Recognition: Christian (81.20% confidence)
✅ Product Prediction: Sports (62.85% confidence)
✅ Voice Verification: member 1 (92.89% confidence)
✅ ALL CHECKS PASSED
```

## System Requirements

### Models
- **Face Recognition Model**: 128-dimensional face encodings, SVM classifier
- **Voice Model**: 17 audio features (13 MFCCs + 4 spectral), Random Forest (200 trees)
- **Product Model**: 17 customer features, XGBoost classifier

### Input Requirements

1. **Face Image**: JPEG/PNG format, clear frontal face
2. **Voice Sample**: WAV/OPUS format, 1-5 seconds of speech
3. **Customer Data** (17 features):
   - `purchase_amount`, `customer_rating`, `engagement_score`
   - `purchase_interest_score`, `platform_encoded`, `sentiment_encoded`
   - `purchase_month`, `purchase_day_of_week`, `purchase_quarter`
   - `avg_purchase`, `total_spent`, `purchase_count`
   - `purchase_std`, `avg_rating`, `avg_engagement`, `avg_interest`

---

## Model Performance

- **Face Recognition**: 81.20% confidence on test images
- **Voice Verification**: 92.89% confidence with data augmentation
- **Product Prediction**: Predicts 5 categories with cross-validation

### Authentication Thresholds
- Face Confidence: ≥ 60%
- Face Distance: < 0.6
- Voice Confidence: ≥ 50%

---
## Troubleshooting

### "ModuleNotFoundError: No module named 'face_recognition'"
```bash
conda activate facial_rec
pip install face-recognition
```

## Features

✅ Multi-factor authentication (Face + Voice)  
✅ Product category prediction integration  
✅ Real-time confidence scores  
✅ Data augmentation for voice training  
✅ EXIF orientation handling for images  
✅ Comprehensive logging and error handling  
✅ API-ready architecture  

