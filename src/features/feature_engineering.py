"""
Feature Engineering Script for Food Delivery ETA Prediction

This script reproduces the feature engineering from 04_feature_engineering.ipynb:
- Interaction features (Distance × Traffic, Distance × Preparation, Experience × Distance)
- Ratio features (Distance per Preparation, Experience per Distance)
- Estimated time features (Estimated_Delivery_Time, Total_Estimated_Time)
- Time-related features (Time_of_Day_Num, Is_Rush_Hour, Is_Adverse_Weather)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def engineer_features(df):
    """
    Engineer features for food delivery ETA prediction.
    
    Args:
        df: Clean dataframe
    
    Returns:
        Dataframe with engineered features
    """
    df_engineered = df.copy()
    
    # Interaction features
    traffic_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    df_engineered['Traffic_Num'] = df_engineered['Traffic_Level'].map(traffic_mapping)
    df_engineered['Distance_Traffic_Interaction'] = df_engineered['Distance_km'] * df_engineered['Traffic_Num']
    df_engineered['Distance_Preparation_Interaction'] = df_engineered['Distance_km'] * df_engineered['Preparation_Time_min']
    df_engineered['Experience_Distance_Interaction'] = df_engineered['Courier_Experience_yrs'] * df_engineered['Distance_km']
    
    # Ratio features
    df_engineered['Distance_per_Preparation'] = df_engineered['Distance_km'] / (df_engineered['Preparation_Time_min'] + 1)
    df_engineered['Experience_per_Distance'] = df_engineered['Courier_Experience_yrs'] / (df_engineered['Distance_km'] + 1)
    
    # Estimated time features (assuming average speed of 20 km/h for delivery)
    df_engineered['Estimated_Delivery_Time'] = df_engineered['Distance_km'] / 20 * 60  # in minutes
    df_engineered['Total_Estimated_Time'] = df_engineered['Preparation_Time_min'] + df_engineered['Estimated_Delivery_Time']
    
    # Time-related features
    time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
    df_engineered['Time_of_Day_Num'] = df_engineered['Time_of_Day'].map(time_mapping)
    df_engineered['Is_Rush_Hour'] = (df_engineered['Time_of_Day'] == 'Evening').astype(int)
    
    # Weather indicator
    adverse_weather = ['Rainy', 'Snowy', 'Foggy', 'Windy']
    df_engineered['Is_Adverse_Weather'] = df_engineered['Weather'].isin(adverse_weather).astype(int)
    
    return df_engineered


def main():
    """Main function to engineer features and save."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    clean_data_path = project_root / 'data' / 'processed' / 'food_delivery_clean.csv'
    features_data_path = project_root / 'data' / 'processed' / 'food_delivery_features.csv'
    
    # Load clean data
    print(f"Loading clean data from: {clean_data_path}")
    df = pd.read_csv(clean_data_path)
    print(f"Clean dataset shape: {df.shape}")
    
    # Engineer features
    print("Engineering features...")
    df_engineered = engineer_features(df)
    
    # Summary
    print(f"Engineered dataset shape: {df_engineered.shape}")
    print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")
    
    # Check for missing values in engineered features
    print(f"Missing values in engineered dataset: {df_engineered.isnull().sum().sum()}")
    
    # Save engineered data
    df_engineered.to_csv(features_data_path, index=False)
    print(f"Engineered dataset saved to: {features_data_path}")


if __name__ == "__main__":
    main()
