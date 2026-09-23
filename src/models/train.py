"""
Model Training Script for Food Delivery ETA Prediction

Models compared:
- Baseline
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

The best model is selected automatically using lowest RMSE.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate a model."""
    print(f"\n=== {name} ===")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

    return {
        "name": name,
        "pipeline": pipeline,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def main():
    """Train, compare, and select the best ETA prediction model."""

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    project_root = Path(__file__).parent.parent.parent

    features_data_path = (
        project_root / "data" / "processed" / "food_delivery_features.csv"
    )

    model_path = (
        project_root / "models" / "food_delivery_eta_model.pkl"
    )

    x_test_path = (
        project_root / "data" / "processed" / "X_test.csv"
    )

    y_test_path = (
        project_root / "data" / "processed" / "y_test.csv"
    )

    mlflow_db_path = project_root / "mlflow.db"

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    print(f"Loading features data from: {features_data_path}")

    df = pd.read_csv(features_data_path)

    print(f"Dataset loaded: {df.shape}")

    # ---------------------------------------------------------
    # Features and target
    # ---------------------------------------------------------
    target = "Delivery_Time_min"

    features = [
        "Distance_km",
        "Preparation_Time_min",
        "Courier_Experience_yrs",
        "Total_Estimated_Time",
        "Distance_Preparation_Interaction",
        "Distance_Traffic_Interaction",
        "Estimated_Delivery_Time",
        "Weather",
        "Traffic_Level",
        "Time_of_Day",
        "Vehicle_Type"
    ]

    X = df[features]
    y = df[target]

    print(f"Features: {len(features)}")
    print(f"Target: {target}")

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # ---------------------------------------------------------
    # Identify feature types
    # ---------------------------------------------------------
    numerical_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # ---------------------------------------------------------
    # Preprocessor for sklearn / XGBoost / LightGBM
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    # ---------------------------------------------------------
    # MLflow setup
    # ---------------------------------------------------------
    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db_path}"
    )

    mlflow.set_experiment(
        "Food_Delivery_ETA_Prediction"
    )

    print("MLflow experiment set")

    # =========================================================
    # 1. Baseline
    # =========================================================
    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DummyRegressor(strategy="mean"))
    ])

    results = []

    with mlflow.start_run(run_name="Baseline_Mean"):

        result = evaluate_model(
            "Baseline",
            baseline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        mlflow.log_metrics({
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        results.append(result)

    # =========================================================
    # 2. Linear Regression
    # =========================================================
    linear_regression = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    with mlflow.start_run(run_name="Linear_Regression"):

        result = evaluate_model(
            "Linear Regression",
            linear_regression,
            X_train,
            X_test,
            y_train,
            y_test
        )

        mlflow.log_metrics({
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        mlflow.sklearn.log_model(
            linear_regression,
            "model",
            serialization_format="cloudpickle"
        )

        results.append(result)

    # =========================================================
    # 3. Random Forest
    # =========================================================
    random_forest = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    with mlflow.start_run(run_name="Random_Forest"):

        result = evaluate_model(
            "Random Forest",
            random_forest,
            X_train,
            X_test,
            y_train,
            y_test
        )

        mlflow.log_metrics({
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        mlflow.log_params({
            "n_estimators": 100,
            "random_state": 42
        })

        mlflow.sklearn.log_model(
            random_forest,
            "model",
            serialization_format="cloudpickle"
        )

        results.append(result)

    # =========================================================
    # 4. Gradient Boosting
    # =========================================================
    gradient_boosting = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    with mlflow.start_run(run_name="Gradient_Boosting"):

        result = evaluate_model(
            "Gradient Boosting",
            gradient_boosting,
            X_train,
            X_test,
            y_train,
            y_test
        )

        mlflow.log_metrics({
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        mlflow.log_params({
            "n_estimators": 100,
            "random_state": 42
        })

        mlflow.sklearn.log_model(
            gradient_boosting,
            "model",
            serialization_format="cloudpickle"
        )

        results.append(result)

    # =========================================================
    # 5. XGBoost
    # =========================================================
    xgboost_model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
                objective="reg:squarederror"
            )
        )
    ])

    with mlflow.start_run(run_name="XGBoost"):

        result = evaluate_model(
            "XGBoost",
            xgboost_model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        mlflow.log_metrics({
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42
        })

        mlflow.sklearn.log_model(
            xgboost_model,
            "model",
            serialization_format="cloudpickle"
        )

        results.append(result)

    # =========================================================
    # 6. LightGBM
    # =========================================================
    lightgbm_model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                random_state=42,
                verbosity=-1
            )
        )
    ])

    with mlflow.start_run(run_name="LightGBM"):

        result = evaluate_model(
            "LightGBM",
            lightgbm_model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        mlflow.log_metrics({
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        mlflow.log_params({
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "random_state": 42
        })

        mlflow.sklearn.log_model(
            lightgbm_model,
            "model",
            serialization_format="cloudpickle"
        )

        results.append(result)

    # =========================================================
    # 7. CatBoost
    # =========================================================
    # CatBoost handles categorical features natively.
    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()

    for col in categorical_features:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)

    catboost_model = CatBoostRegressor(
        iterations=100,
        depth=4,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    print("\n=== CatBoost ===")

    with mlflow.start_run(run_name="CatBoost"):

        catboost_model.fit(
            X_train_cat,
            y_train,
            cat_features=categorical_features
        )

        y_pred = catboost_model.predict(X_test_cat)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )
        r2 = r2_score(y_test, y_pred)

        print(
            f"CatBoost - MAE: {mae:.2f}, "
            f"RMSE: {rmse:.2f}, "
            f"R²: {r2:.3f}"
        )

        mlflow.log_metrics({
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        mlflow.log_params({
            "iterations": 100,
            "depth": 4,
            "learning_rate": 0.05,
            "random_seed": 42
        })

        mlflow.sklearn.log_model(
            catboost_model,
            "model",
            serialization_format="cloudpickle"
        )

        results.append({
            "name": "CatBoost",
            "pipeline": catboost_model,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    # =========================================================
    # Model comparison
    # =========================================================
    results_df = pd.DataFrame([
        {
            "Model": r["name"],
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "R2": r["R2"]
        }
        for r in results
    ])

    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    print(results_df.to_string(index=False))

    # ---------------------------------------------------------
    # Select best model using lowest RMSE
    # ---------------------------------------------------------
    best_result = min(
        results,
        key=lambda x: x["RMSE"]
    )

    best_model_name = best_result["name"]
    best_pipeline = best_result["pipeline"]

    print("\n" + "=" * 65)
    print("BEST MODEL")
    print("=" * 65)
    print(f"Model: {best_model_name}")
    print(f"RMSE:  {best_result['RMSE']:.2f}")
    print(f"MAE:   {best_result['MAE']:.2f}")
    print(f"R²:    {best_result['R2']:.3f}")

    # ---------------------------------------------------------
    # Save best model
    # ---------------------------------------------------------
    joblib.dump(
        best_pipeline,
        model_path
    )

    print(
        f"\nBest model saved to: {model_path}"
    )

    # ---------------------------------------------------------
    # Save test data
    # ---------------------------------------------------------
    X_test.to_csv(
        x_test_path,
        index=False
    )

    y_test.to_csv(
        y_test_path,
        index=False
    )

    print("Test data saved for evaluation")

    print("\n=== Training Complete ===")
    print("All experiments logged to MLflow")
    print(
        f"Final selected model: {best_model_name}"
    )


if __name__ == "__main__":
    main()
