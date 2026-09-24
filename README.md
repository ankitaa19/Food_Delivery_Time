# Food Delivery ETA Prediction

A machine learning application for predicting food delivery times using Linear Regression. The project includes data processing, model training, MLflow experiment tracking, and a web interface for predictions.

## Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose (for containerized deployment)

### Docker Deployment (Recommended)

1. **Build and start the application:**

```bash
docker compose build
docker compose up -d
```

2. **Access the application:**

- **Frontend:** http://localhost:8501
- **FastAPI Backend:** http://localhost:8000
- **FastAPI Swagger UI:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

3. **Stop the application:**

```bash
docker compose down
```

### Local Development

For local development without Docker:

1. **Install dependencies:**

```bash
# Backend
cd api
pip install -r requirements.txt

```

2. **Start the backend:**

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Start the frontend:**

```bash
cd frontend
python -m http.server 8501
```

## Docker Architecture

The Docker setup uses a service-separated architecture:

```
Browser
   ↓
Frontend (:8501)  →  FastAPI (:8000)
   ↓
MLflow Production Model
```

### Services

- **api:** FastAPI backend service that loads the MLflow-registered production model
- **frontend:** Web UI that calls the prediction API from the browser

### MLflow Model Access

The FastAPI container accesses the MLflow production model through:
- Mounted `mlflow.db` file for the model registry
- Mounted `mlruns/` directory for model artifacts
- Production alias: `Food_Delivery_ETA_Model@production`

The application uses a fallback mechanism:
1. First attempts to load via MLflow alias
2. Falls back to direct model path if alias resolution fails

### Container Communication

The frontend communicates with the backend using Docker Compose networking:
- Service name: `api`
- Environment variable: `FASTAPI_URL=http://api:8000`
- Health checks ensure API is ready before frontend starts

## Model Performance

- **Model:** Linear Regression
- **MAE:** 5.91 minutes
- **RMSE:** 8.83 minutes
- **R²:** 0.826

## Supported ATS Platforms

None (not applicable to this project)

## Project Structure

```
Food_Delivery_Times/
├── api/                    # FastAPI backend
│   ├── main.py           # API endpoints and model loading
│   └── requirements.txt  # Backend dependencies
├── src/                   # Source code
│   ├── data/            # Data processing
│   ├── features/        # Feature engineering
│   └── models/          # Model training
├── data/                 # Raw data
├── mlflow.db            # MLflow tracking database
├── mlruns/              # MLflow experiment artifacts
├── frontend/            # Delivery ETA web UI
├── app.py               # Previous Streamlit form
├── Dockerfile.api        # FastAPI container definition
├── Dockerfile.frontend  # Frontend container definition
├── docker-compose.yml   # Docker Compose configuration
└── .dockerignore        # Docker build exclusions
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /predict` - Delivery time prediction endpoint

## Features

- 11 engineered features for prediction
- MLflow experiment tracking
- Model versioning with production alias
- Real-time prediction via web interface
- Containerized deployment with Docker Compose

## Testing

Test the prediction API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Distance_km": 5.3,
    "Preparation_Time_min": 16,
    "Courier_Experience_yrs": 5.0,
    "Total_Estimated_Time": 31.9,
    "Distance_Preparation_Interaction": 84.8,
    "Distance_Traffic_Interaction": 5.3,
    "Estimated_Delivery_Time": 15.9,
    "Weather": "Clear",
    "Traffic_Level": "Low",
    "Time_of_Day": "Evening",
    "Vehicle_Type": "Bike"
  }'
```

Expected response: approximately 35.34 minutes

## Troubleshooting

### Docker Issues

**Container won't start:**
```bash
docker compose logs api
docker compose logs frontend
```

**MLflow model loading error:**
- Ensure `mlflow.db` and `mlruns/` are properly mounted
- Check that the production alias exists in the model registry

**Frontend can't connect to API:**
- Verify both containers are on the same Docker network
- Check that `FASTAPI_URL` environment variable is set correctly
- Ensure API health check passes

### Local Development Issues

**Port already in use:**
```bash
# Change ports in docker-compose.yml or use different ports locally
export PORT=8001
```

**MLflow not finding model:**
- Ensure MLflow tracking URI is set correctly
- Verify the production alias exists in the model registry
