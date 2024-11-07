from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr
from typing import Dict, Any
from predict import Predictor

app = FastAPI()

# Initialize the predictor
predictor = Predictor()
predictor.load_model()  # Load the model from the specified directory

class PredictionRequest(BaseModel):
    text: str  # Input text for prediction

class PredictionResponse(BaseModel):
    label: str  # Predicted label
    confidence_score: float  # Confidence score of the prediction

@app.get("/")
def read_root():
    return {"message": "Model endpoint is running. Use the POST /predict endpoint for predictions."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Validate input text
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty or whitespace.")
        
        prediction = predictor.predict(request.text)
        
        # Check if prediction contains expected keys
        if "label" not in prediction or "confidence score" not in prediction:
            raise ValueError("Prediction output is missing required fields.")

        return {
            "label": prediction["label"],
            "confidence_score": prediction["confidence score"]
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # Log and raise a generic error for unexpected issues
        predictor.logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")

