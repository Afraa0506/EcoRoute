import mlflow
import mlflow.xgboost
import joblib
import pandas as pd
from train_regressor import train_soc_regressor

def run_mlflow_experiment():
    mlflow.set_experiment("EcoRoute_Energy_Prediction")
    
    with mlflow.start_run():
        mae, rmse = train_soc_regressor()
        
        # Log parameters & metrics
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.05)
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        # Log saved model
        mlflow.log_artifact("models/xgboost_soc_model.pkl")
        print("MLflow Run Logged Successfully!")

if __name__ == "__main__":
    run_mlflow_experiment()