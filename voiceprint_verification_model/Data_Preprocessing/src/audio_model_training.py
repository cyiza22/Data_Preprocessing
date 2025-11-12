import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

def load_features(filepath):
    """Load pre-extracted audio features from CSV."""
    return pd.read_csv(filepath)

def prepare_data(df, label_col='label'):
    """Split features and labels, create stratified train/test split."""
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_voiceprint_model(X_train, y_train):
    """Train a Random Forest classifier on voiceprint features."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    """Evaluate model and print accuracy, F1-score, and classification report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return acc, f1

def main():
    feature_path = '../data/processed/audio_features.csv'
    df = load_features(feature_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_voiceprint_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    joblib.dump(model, '../models/voiceprint_model.pkl')
    print("Model saved to ../models/voiceprint_model.pkl")

if __name__ == '__main__':
    main()
