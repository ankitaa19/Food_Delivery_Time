"""
Data Cleaning Script for Food Delivery ETA Prediction

This script reproduces the cleaning decisions from 03_data_cleaning.ipynb:
- Missing value imputation (median for numerical, mode for categorical)
- Data type corrections (category for categorical)
- No duplicate removal (none found)
- No outlier removal (values appear genuine)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clean_food_delivery_data(df):
    """
    Clean the food delivery dataset.
    
    Args:
        df: Raw dataframe
    
    Returns:
        Cleaned dataframe
    """
    df_cleaned = df.copy()
    
    # Handle missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().any():
            if df_cleaned[col].dtype in ['int64', 'float64']:
                # Numerical: median imputation
                imputation_value = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(imputation_value)
            else:
                # Categorical: mode imputation
                imputation_value = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(imputation_value)
    
    # Convert categorical to category dtype
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'str']).columns
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].astype('category')
    
    return df_cleaned


def main():
    """Main function to clean data and save."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'data' / 'raw' / 'Food_Delivery_Times.csv'
    clean_data_path = project_root / 'data' / 'processed' / 'food_delivery_clean.csv'
    
    # Create output directory
    clean_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    print(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Check duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")
    
    # Clean data
    print("Cleaning data...")
    df_cleaned = clean_food_delivery_data(df)
    
    # Summary
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")
    print(f"Duplicates after cleaning: {df_cleaned.duplicated().sum()}")
    
    # Save cleaned data
    df_cleaned.to_csv(clean_data_path, index=False)
    print(f"Clean dataset saved to: {clean_data_path}")


if __name__ == "__main__":
    main()
