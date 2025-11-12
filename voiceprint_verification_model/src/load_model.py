from pathlib import Path
import sys
import warnings

try:
    import joblib
except Exception:  # pragma: no cover - optional runtime dependency
    joblib = None


def load_model(model_path: Path):
    if joblib is None:
        raise RuntimeError("joblib is not installed in this environment. Activate your venv and install requirements.txt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}\nRun training first to create the model at this path.")
    model = joblib.load(model_path)
    return model


def main():
    default_model = Path(__file__).resolve().parents[1] / 'models' / 'voiceprint_model.pkl'
    arg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_model

    try:
        model = load_model(arg_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(2)

    print(f"Loaded model from: {arg_path}")
    print(f"Model type: {type(model)}")

    # Show a short summary of capabilities
    has_predict = hasattr(model, 'predict')
    has_feature_importances = hasattr(model, 'feature_importances_')
    print(f"Has predict method: {has_predict}")
    print(f"Has feature_importances_: {has_feature_importances}")

    if has_feature_importances:
        try:
            fi = getattr(model, 'feature_importances_')
            print(f"Feature importances length: {len(fi)}")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Could not read feature_importances_: {exc}")


if __name__ == '__main__':
    main()
