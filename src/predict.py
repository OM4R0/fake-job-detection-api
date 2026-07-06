import joblib
import pandas as pd
from xgboost import XGBClassifier

# Import your custom preprocessing pipeline
import preprocess

print("Loading ML artifacts into memory...")
try:
    # Load encoders and vectorizer using joblib
    target_enc = joblib.load('models/target_encoder.pkl')
    ohe_enc = joblib.load('models/ohe_encoder.pkl')
    tfidf_vec = joblib.load('models/tfidf_vectorizer.pkl')
    
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
            
    # 3. Apply Feature Engineering (Single source of truth from preprocess.py)
    df = preprocess.process_salary(df)
    df = preprocess.create_presence_features(df)
    df = preprocess.fill_missing_values(df)
    df = preprocess.create_length_features(df)
    
    # 4. Apply Encoders (using transform ONLY, no fit)
    # Passing df twice as a workaround if the function expects train/test tuple
    df, _ = preprocess.apply_ordinal_encoding(df, df) 
    
    df = target_enc.transform(df)
    df = ohe_enc.transform(df)
    
    # 5. Apply TF-IDF Vectorization for text columns
    text_columns = ['company_profile', 'title', 'description', 'requirements', 'benefits']
    df[text_columns] = df[text_columns].fillna('')
    combined_text = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    text_tfidf = pd.DataFrame(tfidf_vec.transform(combined_text).toarray(), index=df.index)
    
    # Drop original text columns and merge the TF-IDF features
    df = df.drop(columns=text_columns)
    df = pd.concat([df, text_tfidf], axis=1)
    
    # Ensure column order matches the training data exactly (if needed)
    # df = df[model.feature_names_in_] 

    # 6. Generate Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    # Return results as a clean dictionary ready for JSON response
    return {
        "is_fraudulent": int(prediction),
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