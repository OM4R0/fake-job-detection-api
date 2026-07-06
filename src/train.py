import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score
from xgboost import XGBClassifier

# Import custom preprocessing module
import preprocess

def evaluate_model(y_true, y_pred):
    """Calculate and print evaluation metrics for logging purposes."""
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- Training Evaluation ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("---------------------------\n")

def main():
    """Main pipeline: Load data, preprocess, train model, evaluate, and save artifacts."""
    
    print("Loading raw data...")
    df = pd.read_csv('data/raw/fake_job_postings.csv')
    df.drop(columns=['job_id'], inplace=True)
    df.drop_duplicates(inplace=True)

    print("Splitting data into train/test sets...")
    X = df.drop(columns=['fraudulent'])
    y = df['fraudulent']
    
    # Stratify ensures the minority fraud cases are distributed evenly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Executing feature engineering pipeline...")
    X_train = preprocess.process_salary(X_train)
    X_test = preprocess.process_salary(X_test)

    X_train = preprocess.create_presence_features(X_train)
    X_test = preprocess.create_presence_features(X_test)

    X_train = preprocess.fill_missing_values(X_train)
    X_test = preprocess.fill_missing_values(X_test)

    X_train = preprocess.create_length_features(X_train)
    X_test = preprocess.create_length_features(X_test)

    print("Applying encoders and text vectorization...")
    X_train, X_test = preprocess.apply_ordinal_encoding(X_train, X_test)
    X_train, X_test, target_enc = preprocess.apply_target_encoding(X_train, X_test, y_train)
    X_train, X_test, ohe_enc = preprocess.apply_ohe(X_train, X_test)
    X_train, X_test, tfidf_vec = preprocess.apply_tfidf(X_train, X_test)

    print("Training XGBClassifier with optimal hyperparameters...")
    # Best parameters extracted from Optuna study
    model = XGBClassifier(
        n_estimators=370,
        learning_rate=0.012491787341519192,
        max_depth=3,
        scale_pos_weight=124,
        subsample=0.6860363727116727,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    print("Evaluating model on test data...")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    print("Saving model and transformers for API deployment...")
    os.makedirs('models', exist_ok=True)
    
    model.save_model('models/xgboost_fraud_model.json')
    joblib.dump(target_enc, 'models/target_encoder.pkl')
    joblib.dump(ohe_enc, 'models/ohe_encoder.pkl')
    joblib.dump(tfidf_vec, 'models/tfidf_vectorizer.pkl')
    df.to_csv('data/processed/fake_job_postings_cleaned.csv', index=False)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    print("Pipeline executed successfully. Artifacts saved in 'models/'.")

if __name__ == "__main__":
    main()