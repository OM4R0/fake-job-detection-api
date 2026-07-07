# FraudGuard: ML-Powered Job Posting Detection API

FraudGuard is an end-to-end Machine Learning pipeline and REST API designed to detect fraudulent or fake job postings in real time. It uses Natural Language Processing (NLP) and an XGBoost classifier to analyze job descriptions, company profiles, and metadata, returning a probability score of fraudulence.

## 🚀 Features

- **Real-time Inference API:** Built with FastAPI, offering high performance and automatic interactive documentation (Swagger UI).
- **Advanced Feature Engineering:** 
  - **TF-IDF Vectorization** (1000 features) for analyzing text fields (Job Description, Requirements, Company Profile).
  - **Target Encoding** for high-cardinality categorical variables (e.g., Location, Title, Department).
  - **One-Hot Encoding** for low-cardinality variables (e.g., Employment Type, Education).
- **Imbalanced Data Handling:** Uses `scale_pos_weight` optimized for high-recall fraud detection, prioritizing the identification of fake postings over false positives.
- **Interactive UI:** A dark-themed, responsive vanilla HTML/JS frontend that interacts seamlessly with the backend API.
- **Dockerized:** Ready for production deployment on platforms like Render or AWS using an optimized Python 3.12 slim image.

## 📁 Project Structure

```text
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned data and encoded features
├── frontend/
│   └── index.html           # Professional dark-mode UI for testing the API
├── models/                  # Serialized ML artifacts (XGBoost model, TF-IDF, Encoders)
├── src/
│   ├── main.py              # FastAPI application and endpoints
│   ├── predict.py           # Inference logic (loading models, transforming data)
│   ├── preprocess.py        # Feature engineering and data cleaning pipeline
│   ├── schemas.py           # Pydantic schemas for strict API input validation
│   ├── train.py             # Model training, hyperparameter tuning, and evaluation
│   └── utils.py             # Shared constants and configurations
├── Dockerfile               # Production-ready Docker configuration
└── requirements.txt         # Project dependencies
```

## 🛠️ Tech Stack

- **Machine Learning:** `xgboost`, `scikit-learn`, `pandas`, `category_encoders`
- **Backend API:** `FastAPI`, `pydantic`, `uvicorn`
- **Deployment:** `Docker`
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)

## 💻 Running Locally

### 1. Using Python (Virtual Environment)
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python src/main.py
```
Then, open `frontend/index.html` in your web browser to use the UI. You can also view the API documentation at `http://127.0.0.1:8000/docs`.

### 2. Using Docker
```bash
# Build the image
docker build -t fraud-detection-api .

# Run the container
docker run -p 8000:8000 fraud-detection-api
```

## 🧠 Model Pipeline Overview

1. **Preprocessing (`preprocess.py`):** Cleans salary formats, handles missing values, and merges text fields.
2. **Training (`train.py`):** Splits data, fits the TF-IDF vectorizer and categorical encoders (Target & OHE) strictly on the training set to prevent data leakage, and trains the XGBoost model.
3. **Inference (`predict.py`):** Loads saved artifacts, applies transformations to real-time JSON payloads, ensures column alignment using `reindex`, and predicts the fraud probability.

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

## 👨‍💻 About the Developer
This project was developed to demonstrate end-to-end ML engineering skills, focusing on robust data pipelines, API development, and production-ready deployments.