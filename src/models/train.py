"""
Model Training Script for Food Delivery ETA Prediction

This script reproduces the model training from 06_model_experiments.ipynb:
- Train/test split (80/20, random_state=42)
- Preprocessing pipeline (StandardScaler + OneHotEncoder)
- Model comparison (Baseline, Linear Regression, Random Forest, Gradient Boosting)
- Select best model based on MAE, RMSE, R²
- Save complete pipeline
- Track experiments with MLflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path


def main():
    """Main function to train models and save the best one."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    features_data_path = project_root / 'data' / 'processed' / 'food_delivery_features.csv'
    model_path = project_root / 'models' / 'food_delivery_eta_model.pkl'
    x_test_path = project_root / 'data' / 'processed' / 'X_test.csv'
    y_test_path = project_root / 'data' / 'processed' / 'y_test.csv'
    mlflow_db_path = project_root / 'mlflow.db'
    
    # Create output directories
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_test_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load features data
    print(f"Loading features data from: {features_data_path}")
    df = pd.read_csv(features_data_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Define features and target
    target = 'Delivery_Time_min'
    features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 
               'Total_Estimated_Time', 'Distance_Preparation_Interaction', 
               'Distance_Traffic_Interaction', 'Estimated_Delivery_Time',
               'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    
    X = df[features]
    y = df[target]
    
    print(f"Features: {len(features)}")
    print(f"Target: {target}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Preprocessing pipeline
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    print("Preprocessing pipeline created")
    
    # Set up MLflow
    mlflow.set_tracking_uri(f'sqlite:///{mlflow_db_path}')
    mlflow.set_experiment("Food_Delivery_ETA_Prediction")
    print("MLflow experiment set")
    
    # Experiment 1: Baseline
    print("\n=== Baseline Model ===")
    with mlflow.start_run(run_name="Baseline_Mean"):
        baseline_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', DummyRegressor(strategy='mean'))
        ])
        
        baseline_pipeline.fit(X_train, y_train)
        y_pred = baseline_pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({'MAE': mae, 'RMSE': rmse, 'R2': r2})
        
        print(f"Baseline - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    # Experiment 2: Linear Regression
    print("\n=== Linear Regression ===")
    with mlflow.start_run(run_name="Linear_Regression"):
        lr_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        
        lr_pipeline.fit(X_train, y_train)
        y_pred = lr_pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({'MAE': mae, 'RMSE': rmse, 'R2': r2})
        mlflow.sklearn.log_model(lr_pipeline, "model", serialization_format='cloudpickle')
        
        print(f"Linear Regression - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    # Experiment 3: Random Forest
    print("\n=== Random Forest ===")
    with mlflow.start_run(run_name="Random_Forest"):
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        rf_pipeline.fit(X_train, y_train)
        y_pred = rf_pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({'MAE': mae, 'RMSE': rmse, 'R2': r2})
        mlflow.log_params({'n_estimators': 100, 'random_state': 42})
        mlflow.sklearn.log_model(rf_pipeline, "model", serialization_format='cloudpickle')
        
        print(f"Random Forest - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    # Experiment 4: Gradient Boosting (Final Model)
    print("\n=== Gradient Boosting (Final Model) ===")
    with mlflow.start_run(run_name="Gradient_Boosting_Final"):
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        final_pipeline.fit(X_train, y_train)
        y_pred = final_pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({'MAE': mae, 'RMSE': rmse, 'R2': r2})
        mlflow.log_params({'n_estimators': 100, 'random_state': 42})
        mlflow.sklearn.log_model(final_pipeline, "model", serialization_format='cloudpickle')
        
        print(f"Gradient Boosting - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    # Save final model
    joblib.dump(final_pipeline, model_path)
    print(f"\nFinal model saved to: {model_path}")
    
    # Save test data for evaluation
    X_test.to_csv(x_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print(f"Test data saved for evaluation")
    
    print("\n=== Training Complete ===")
    print("All experiments logged to MLflow")
    print("Final model: Gradient Boosting Regressor")


if __name__ == "__main__":
    main()
