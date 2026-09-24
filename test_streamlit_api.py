"""
Test script to verify Streamlit frontend can call FastAPI backend correctly.
"""

import requests

API_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{API_URL}/health"
PREDICT_ENDPOINT = f"{API_URL}/predict"

# Test health check
print("=== Testing API Health ===")
try:
    response = requests.get(HEALTH_ENDPOINT, timeout=5)
    if response.status_code == 200:
        print(f"✅ API Healthy: {response.json()}")
    else:
        print(f"❌ API Unhealthy: Status {response.status_code}")
except Exception as e:
    print(f"❌ API Health Check Failed: {e}")

# Test prediction with verified sample
print("\n=== Testing Prediction ===")
test_features = {
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
}

try:
    response = requests.post(PREDICT_ENDPOINT, json=test_features, timeout=10)
    response.raise_for_status()
    result = response.json()
    prediction = result.get("predicted_delivery_time_min")
    print(f"✅ Prediction Successful: {prediction:.2f} minutes")
    print(f"   Expected: ~35.34 minutes")
    print(f"   Difference: {abs(prediction - 35.34):.2f} minutes")
except Exception as e:
    print(f"❌ Prediction Failed: {e}")

print("\n=== Test Complete ===")
