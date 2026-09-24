"""
FastAPI Backend for Food Delivery ETA Prediction

This API serves the MLflow-registered Linear Regression model
for predicting food delivery times.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import mlflow
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food Delivery ETA Prediction API",
    description="Predict food delivery time using MLflow-registered Linear Regression model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MLFLOW_TRACKING_URI = "sqlite:///../mlflow.db"
MODEL_NAME = "Food_Delivery_ETA_Model"
MODEL_ALIAS = "production"

# Global model variable
model = None


class DeliveryTimeRequest(BaseModel):
    """Request schema for delivery time prediction."""
    
    Distance_km: float = Field(..., gt=0, description="Delivery distance in kilometers")
    Preparation_Time_min: float = Field(..., gt=0, description="Restaurant preparation time in minutes")
    Courier_Experience_yrs: float = Field(..., ge=0, description="Courier experience in years")
    Total_Estimated_Time: float = Field(..., gt=0, description="Total estimated time (preparation + delivery)")
    Distance_Preparation_Interaction: float = Field(..., description="Distance × Preparation Time interaction")
    Distance_Traffic_Interaction: float = Field(..., description="Distance × Traffic Level interaction")
    Estimated_Delivery_Time: float = Field(..., gt=0, description="Estimated delivery time based on distance")
    Weather: str = Field(..., description="Weather condition (Clear, Rainy, Snowy, etc.)")
    Traffic_Level: str = Field(..., description="Traffic level (Low, Medium, High)")
    Time_of_Day: str = Field(..., description="Time of day (Morning, Afternoon, Evening, Night)")
    Vehicle_Type: str = Field(..., description="Vehicle type (Bike, Car, Scooter, etc.)")


class DeliveryTimeResponse(BaseModel):
    """Response schema for delivery time prediction."""
    
    predicted_delivery_time_min: float = Field(..., description="Predicted delivery time in minutes")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service health status")


def load_model():
    """Load the production model from MLflow Model Registry."""
    global model
    
    try:
        # Set MLflow tracking URI
        project_root = Path(__file__).parent.parent
        mlflow_db_path = project_root / "mlflow.db"
        mlflow.set_tracking_uri(f'sqlite:///{mlflow_db_path}')
        
        # Try loading using production alias first
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        logger.info(f"Loading model from: {model_uri}")
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully via alias")
            return True
        except Exception as alias_error:
            logger.warning(f"Failed to load model via alias: {alias_error}")
            # Fallback to direct model path
            logger.info("Attempting to load model from direct path...")
            # Get the model version by alias to find the source
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            # Use the source directly
            direct_uri = model_version.source
            logger.info(f"Loading model from direct path: {direct_uri}")
            model = mlflow.sklearn.load_model(direct_uri)
            logger.info("Model loaded successfully via direct path")
            return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up FastAPI application...")
    if not load_model():
        logger.error("Application started but model loading failed")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="healthy")


@app.post("/predict", response_model=DeliveryTimeResponse)
async def predict(request: DeliveryTimeRequest):
    """
    Predict delivery time based on input features.
    
    Args:
        request: DeliveryTimeRequest with 11 features
        
    Returns:
        DeliveryTimeResponse with predicted delivery time
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame with exact feature order
        feature_order = [
            'Distance_km',
            'Preparation_Time_min',
            'Courier_Experience_yrs',
            'Total_Estimated_Time',
            'Distance_Preparation_Interaction',
            'Distance_Traffic_Interaction',
            'Estimated_Delivery_Time',
            'Weather',
            'Traffic_Level',
            'Time_of_Day',
            'Vehicle_Type'
        ]
        
        # Create DataFrame from request
        data = {
            'Distance_km': request.Distance_km,
            'Preparation_Time_min': request.Preparation_Time_min,
            'Courier_Experience_yrs': request.Courier_Experience_yrs,
            'Total_Estimated_Time': request.Total_Estimated_Time,
            'Distance_Preparation_Interaction': request.Distance_Preparation_Interaction,
            'Distance_Traffic_Interaction': request.Distance_Traffic_Interaction,
            'Estimated_Delivery_Time': request.Estimated_Delivery_Time,
            'Weather': request.Weather,
            'Traffic_Level': request.Traffic_Level,
            'Time_of_Day': request.Time_of_Day,
            'Vehicle_Type': request.Vehicle_Type
        }
        
        df = pd.DataFrame([data], columns=feature_order)
        
        # Make prediction
        prediction = model.predict(df)
        predicted_time = float(prediction[0])
        
        logger.info(f"Prediction successful: {predicted_time:.2f} minutes")
        
        return DeliveryTimeResponse(predicted_delivery_time_min=predicted_time)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
