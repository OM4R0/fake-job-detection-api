# FraudGuard: ML-Powered Job Posting Detection API

FraudGuard is an end-to-end Machine Learning pipeline and REST API that detects fraudulent or fake job postings in real time. It uses NLP feature engineering and an XGBoost classifier to analyze job descriptions, company profiles, and metadata, returning a fraud probability score.

**🔗 Live demo:** https://fakejobdetection-08dl.onrender.com/
**📄 API docs (Swagger):** https://fakejobdetection-08dl.onrender.com/docs

> Note: hosted on Render's free tier — the first request after a period of inactivity may take 30-50 seconds to wake the server up.

## 🚀 Features

- **Real-time Inference API:** Built with FastAPI, with interactive documentation (Swagger UI) and a simple web UI served directly at the root URL.
- **Advanced Feature Engineering:**
  - **TF-IDF Vectorization** (1000 features) for text fields (job description, requirements, company profile).
  - **Target Encoding** for high-cardinality categorical variables (e.g. location, industry, function).
  - **One-Hot Encoding** for low-cardinality variables (e.g. employment type).
  - **Ordinal Encoding** for naturally ranked fields (experience level, education level).
- **Imbalanced Data Handling:** `scale_pos_weight` tuned via Optuna, with the decision threshold deliberately set to prioritize recall — see [Model Performance](#-model-performance) below.
- **Dockerized:** Production-ready image based on Python 3.12 slim, deployed on Render.

## 📁 Project Structure

```text
├── data/                     # Gitignored — regenerate via src/train.py
│   ├── raw/                  # Original Kaggle dataset
│   └── processed/            # Cleaned data and train/test splits
├── frontend/
│   └── index.html            # Dark-mode UI for testing the API, served at "/"
├── models/                   # Serialized ML artifacts (XGBoost model, TF-IDF, encoders)
├── src/
│   ├── main.py                # FastAPI application and endpoints
│   ├── predict.py             # Inference logic (loads models, runs the pipeline)
│   ├── preprocess.py           # Shared feature engineering (used by both training and inference)
│   ├── schemas.py              # Pydantic schemas for strict API input validation
│   ├── train.py                # Model training, evaluation, and artifact export
│   └── utils.py                 # Shared constants (valid categories for API/schema)
├── Dockerfile                # Production Docker configuration
└── requirements.txt
```

## 🛠️ Tech Stack

- **Machine Learning:** `xgboost`, `scikit-learn`, `pandas`, `category_encoders`
- **Backend API:** `FastAPI`, `pydantic`, `uvicorn`
- **Deployment:** `Docker`, Render
- **Frontend:** HTML5, CSS3, vanilla JavaScript

## 📊 Model Performance

Evaluated on a held-out test set (20% split, stratified) from ~18k job postings (~5% fraudulent):

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.3 | 0.287 | 0.877 | 0.432 |
| 0.4 | 0.353 | 0.830 | 0.496 |
| **0.5 (used)** | **0.421** | **0.789** | **0.549** |
| 0.6 | 0.496 | 0.749 | 0.597 |
| 0.7 | 0.551 | 0.637 | 0.591 |

**Decision:** the deployed threshold is **0.5**, chosen deliberately to prioritize recall over precision. In this problem, a missed fraudulent posting (false negative) is more harmful to a job seeker than an occasional false alarm on a real posting — so the model is tuned to catch as much fraud as possible rather than to minimize false positives.

## 💻 Running Locally

### Using Python
```bash
pip install -r requirements.txt
python src/main.py
```
Then open `http://127.0.0.1:8000/` in your browser, or the API docs at `http://127.0.0.1:8000/docs`.

### Using Docker
```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

### Retraining the model
The raw and processed data are gitignored. To regenerate them and retrain from scratch:
```bash
python src/train.py
```
This re-runs the full pipeline (load → clean → feature engineering → train → evaluate) and overwrites the artifacts in `models/`.

## 🧠 Pipeline Overview

1. **Preprocessing (`preprocess.py`):** Cleans salary formats, handles missing values, extracts text-length features. All transformations are shared between training and inference to guarantee they never drift apart.
2. **Training (`train.py`):** Splits data first, then fits the TF-IDF vectorizer and categorical encoders strictly on the training set — preventing data leakage into the test set.
3. **Inference (`predict.py`):** Loads saved artifacts, applies the same transformations to incoming JSON payloads, reindexes columns to match the training layout exactly, and returns a fraud probability.

## 🌍 API Usage Example

**POST `/predict`**
```json
{
  "title": "Senior Software Engineer",
  "location": "San Francisco, CA",
  "salary_range": "120000-150000",
  "description": "We are seeking a backend engineer...",
  "telecommuting": false,
  "has_company_logo": true,
  "has_questions": true,
  "employment_type": "Full-time",
  "required_experience": "Mid-Senior level",
  "required_education": "Bachelor's Degree"
}
```

**Response**
```json
{
  "is_fraudulent": 0,
  "fraud_probability": 0.4211
}
```

## 👨‍💻 About

Built as an end-to-end ML engineering portfolio project, covering data pipelines, model training with leakage prevention, API design, and containerized deployment.