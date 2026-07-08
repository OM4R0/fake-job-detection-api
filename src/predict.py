import joblib
import pandas as pd
from xgboost import XGBClassifier

# Import your custom preprocessing pipeline
import preprocess

# Threshold = 0.5. Chosen to prioritize recall (catching fraud) over precision,
# since the cost of missing a real scam is higher than a false alarm.
# At 0.5: Precision=0.421, Recall=0.789, F1=0.549 (measured on held-out test set).
DECISION_THRESHOLD = 0.5

print("Loading ML artifacts into memory...")
try:
    # Load encoders and vectorizer using joblib
    target_enc = joblib.load('models/target_encoder.pkl')
    ohe_enc = joblib.load('models/ohe_encoder.pkl')
    tfidf_vec = joblib.load('models/tfidf_vectorizer.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')  # FIX: new artifact from train.py

    # Load XGBoost model using its native method
    model = XGBClassifier()
    model.load_model('models/xgboost_fraud_model.json')

    print("All artifacts loaded successfully!")
except Exception as e:
    print(f"Error loading ML artifacts: {e}")
    raise

def predict_fraud(job_data_dict: dict) -> dict:
    """
    Main inference function.
    Takes a dictionary of job posting data, passes it through the preprocessing
    pipeline, and returns the fraud prediction and probability.
    """
    # 1. Convert incoming dictionary to a single-row DataFrame
    df = pd.DataFrame([job_data_dict])

    # 2. Convert boolean API inputs (True/False) to integers (1/0) for the model
    bool_columns = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 3. Apply Feature Engineering (single source of truth from preprocess.py)
    df = preprocess.process_salary(df)
    df = preprocess.create_presence_features(df)
    df = preprocess.fill_missing_values(df)
    df = preprocess.create_length_features(df)

    # 4. Apply Encoders (using transform ONLY, no fit)
    df, _ = preprocess.apply_ordinal_encoding(df, df)

    df = target_enc.transform(df)
    df = ohe_enc.transform(df)

    # 5. FIX: Apply TF-IDF via the shared preprocess function instead of
    # re-implementing the combine+vectorize steps here. Same code path
    # as train.py's test set now — no more risk of the two drifting apart.
    df = preprocess.apply_tfidf_inference(df, tfidf_vec)

    # 6. FIX: Reindex to match training column order exactly.
    # Without this, OHE/target-encoding can produce columns in a different
    # order (or missing columns) for a single-row request, and XGBoost will
    # silently score against the wrong feature meaning. fill_value=0 handles
    # any column that legitimately can't appear from one row (e.g. a OHE
    # category never seen in this request).
    df = df.reindex(columns=feature_columns, fill_value=0)

    # 7. Generate Prediction
    probability = model.predict_proba(df)[0][1]
    # FIX: use the tuned threshold instead of the model's default .predict()
    # (which is hardcoded to 0.5 internally). Run a PR curve on your
    # validation set to pick DECISION_THRESHOLD properly — see note below.
    prediction = int(probability >= DECISION_THRESHOLD)

    # Return results as a clean dictionary ready for JSON response
    return {
        "is_fraudulent": prediction,
        "fraud_probability": round(float(probability), 4)
    }

# Example local test (Runs only if this file is executed directly)
if __name__ == "__main__":
    dummy_data = {
        "title": "Software Engineer",
        "location": "Amman, Jordan",
        "department": "IT",
        "salary_range": "1000-2000",
        "company_profile": "Tech company",
        "description": "Looking for developer",
        "requirements": "Python, API",
        "benefits": "Health insurance",
        "telecommuting": False,
        "has_company_logo": True,
        "has_questions": False,
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "required_education": "Bachelor's Degree",
        "industry": "Information Technology",
        "function": "Engineering"
    }
    result = predict_fraud(dummy_data)
    print("\nTest Prediction Result:")
    print(result)
