"""
Streamlit Frontend for Food Delivery ETA Prediction

This application provides a user interface for predicting food delivery times
by calling the FastAPI backend.
"""

import streamlit as st
import requests
import os
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Food Delivery ETA Prediction",
    page_icon="🍕",
    layout="wide"
)

# API configuration
API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
HEALTH_ENDPOINT = f"{API_URL}/health"
PREDICT_ENDPOINT = f"{API_URL}/predict"


def check_api_health() -> bool:
    """Check if the FastAPI backend is healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def predict_delivery_time(features: dict) -> Optional[float]:
    """
    Call the FastAPI backend to predict delivery time.
    
    Args:
        features: Dictionary of input features
        
    Returns:
        Predicted delivery time in minutes, or None if prediction fails
    """
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json=features,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        return result.get("predicted_delivery_time_min")
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🍕 Food Delivery ETA Prediction")
    st.subheader("Predict food delivery time using machine learning")
    
    # API status indicator
    api_healthy = check_api_health()
    if api_healthy:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Unavailable")
        st.warning("Please ensure the FastAPI backend is running at: " + API_URL)
    
    st.markdown("---")
    
    # Input section
    st.header("Enter Delivery Details")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delivery Information")
        
        distance_km = st.number_input(
            "Distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=5.3,
            step=0.1,
            help="Delivery distance in kilometers"
        )
        
        preparation_time = st.number_input(
            "Preparation Time (minutes)",
            min_value=1,
            max_value=120,
            value=16,
            step=1,
            help="Restaurant preparation time in minutes"
        )
        
        courier_experience = st.number_input(
            "Courier Experience (years)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Courier experience in years"
        )
        
        weather = st.selectbox(
            "Weather",
            options=["Clear", "Rainy", "Snowy", "Foggy", "Windy", "Cloudy"],
            index=0,
            help="Current weather conditions"
        )
        
        traffic_level = st.selectbox(
            "Traffic Level",
            options=["Low", "Medium", "High"],
            index=0,
            help="Current traffic conditions"
        )
    
    with col2:
        st.subheader("Additional Features")
        
        total_estimated_time = st.number_input(
            "Total Estimated Time (minutes)",
            min_value=1.0,
            max_value=200.0,
            value=31.9,
            step=0.1,
            help="Total estimated time (preparation + delivery)"
        )
        
        distance_prep_interaction = st.number_input(
            "Distance × Preparation Interaction",
            min_value=0.0,
            max_value=10000.0,
            value=84.8,
            step=0.1,
            help="Distance × Preparation Time interaction feature"
        )
        
        distance_traffic_interaction = st.number_input(
            "Distance × Traffic Interaction",
            min_value=0.0,
            max_value=1000.0,
            value=5.3,
            step=0.1,
            help="Distance × Traffic Level interaction feature"
        )
        
        estimated_delivery_time = st.number_input(
            "Estimated Delivery Time (minutes)",
            min_value=1.0,
            max_value=200.0,
            value=15.9,
            step=0.1,
            help="Estimated delivery time based on distance"
        )
        
        time_of_day = st.selectbox(
            "Time of Day",
            options=["Morning", "Afternoon", "Evening", "Night"],
            index=2,
            help="Time of day for delivery"
        )
        
        vehicle_type = st.selectbox(
            "Vehicle Type",
            options=["Bike", "Car", "Scooter", "Motorcycle"],
            index=0,
            help="Type of delivery vehicle"
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🚀 Predict Delivery Time", type="primary", use_container_width=True):
        if not api_healthy:
            st.error("Cannot make prediction: API is not available")
        else:
            # Construct feature dictionary
            features = {
                "Distance_km": distance_km,
                "Preparation_Time_min": preparation_time,
                "Courier_Experience_yrs": courier_experience,
                "Total_Estimated_Time": total_estimated_time,
                "Distance_Preparation_Interaction": distance_prep_interaction,
                "Distance_Traffic_Interaction": distance_traffic_interaction,
                "Estimated_Delivery_Time": estimated_delivery_time,
                "Weather": weather,
                "Traffic_Level": traffic_level,
                "Time_of_Day": time_of_day,
                "Vehicle_Type": vehicle_type
            }
            
            # Make prediction
            with st.spinner("Predicting delivery time..."):
                prediction = predict_delivery_time(features)
            
            # Display result
            if prediction is not None:
                st.markdown("---")
                st.header("📊 Prediction Result")
                
                # Display metric
                st.metric(
                    label="Estimated Delivery Time",
                    value=f"{prediction:.2f} minutes",
                    delta=None
                )
                
                st.info(
                    "The estimate is generated by the trained machine learning "
                    "model through the FastAPI backend."
                )
    
    # Footer
    st.markdown("---")
    st.caption(
        "Built with Streamlit, FastAPI, and MLflow | "
        "Model: Linear Regression (MAE: 5.91 min, RMSE: 8.83 min, R²: 0.826)"
    )


if __name__ == "__main__":
    main()
