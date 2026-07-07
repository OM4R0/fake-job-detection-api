from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the Pydantic schema for input validation
from schemas import JobPostingRequest

# Import the prediction logic from your ML pipeline
from predict import predict_fraud

# Initialize FastAPI application
app = FastAPI(
    title="Fraudulent Job Posting Detection API",
    description="An ML-powered API to detect fake job postings in real-time.",
    version="1.0.0"
)

# Allow the HTML frontend to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    """
    Root endpoint for basic health checking.
    Useful for Docker containers or load balancers to verify the API is alive.
    """
    return {
        "status": "active",
        "message": "Fraud Detection API is running smoothly!"
    }

@app.post("/predict")
def predict_job_fraud(job_request: JobPostingRequest):
    """
    Main inference endpoint.
    Accepts job posting details, validates them, and returns a fraud prediction.
    """
    try:
        # Convert the validated Pydantic object to a standard Python dictionary
        job_data = job_request.model_dump()
        
        # Pass the data to the prediction engine
        prediction_result = predict_fraud(job_data)
        
        return prediction_result
        
    except Exception as e:
        # Catch any unexpected pipeline errors and return a clean 500 response
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# This block allows running the server directly via standard python command if needed
if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)