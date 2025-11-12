# Personalized Product Recommender System

A **machine learning-powered recommendation engine** that predicts the likelihood of a customer purchasing a product based on:
- Purchase history
- User behavior (total spend, recency)
- Product metadata (price, category)

Built with **Python**, **Pandas**, **XGBoost**, **scikit-learn**, and **joblib**.

---

## Features

| Feature | Description |
|--------|-------------|
| **Negative Sampling** | 3× negative samples per purchase for balanced training |
| **Feature Engineering** | `user_total_spend`, `days_since_last_purchase`, `product_price_mean`, `category_id` |
| **Model Comparison** | XGBoost, Random Forest, Logistic Regression |
| **Auto-Save Best Model** | Saves top-performing model + preprocessor + metadata |
| **Real-Time Recommendations** | Top-5 personalized product suggestions |
| **Production Ready** | Model + artifacts saved via `joblib` & `pickle` |

---

## Project Structure
recommender-system/
├── data/                     # (Optional) Place your raw CSV here
├── best_model_XGBoost.pkl    # Trained model (auto-generated)
├── preprocessor.pkl          # Feature transformer
├── recommendation_artifacts.pkl  # Mappings + data snapshot
├── recommender.ipynb         # Full Jupyter notebook
├── recommender.py            # Full pipeline in one script
├── recommend_api.py          # FastAPI endpoint
├── README.md                 # This file
├── requirements.txt          # Dependencies
└── .gitignore
