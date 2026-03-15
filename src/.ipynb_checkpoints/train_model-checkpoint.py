import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Configuration
DATA_SOURCE = "data\\VisaFile.csv" 
MODEL_OUTPUT = "../models/visa_model.pkl"
FEATURE_SCHEMA = "../models/feature_schema.pkl"

class VisaModelTrainer:
    def __init__(self, source_path):
        self.source_path = source_path
        self.trained_engine = None
        self.column_schema = None

    def load_and_transform(self):
        """Standardizes dataset and extracts training matrices."""
        # Load data
        raw_df = pd.read_csv(self.source_path, encoding="latin1")

        # FIX: Corrected column standardization
        raw_df.columns = raw_df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Drop non-predictive/high-cardinality metadata
        unnecessary_cols = [
            'case_id', 'case_number', 'employer_name', 'work_city', 
            'case_received_date', 'decision_date', 'job_title'
        ]
        feature_df = raw_df.drop(columns=[c for c in unnecessary_cols if c in raw_df.columns])

        # One-Hot Encoding for categorical variables
        processed_df = pd.get_dummies(feature_df, drop_first=True)
        
        # Isolate Predictors and Target
        predictor_matrix = processed_df.drop('processing_time_days', axis=1)
        target_vector = processed_df['processing_time_days']
        
        self.column_schema = predictor_matrix.columns.tolist()
        
        return train_test_split(predictor_matrix, target_vector, test_size=0.2, random_state=42)

    def optimize_and_train(self, x_train, x_test, y_train, y_test):
        """Evaluates and tunes the best performing regressor."""
        
        # Testing XGBoost as the primary architecture
        base_regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
        
        param_space = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.8, 1.0]
        }

        print("[*] Optimizing hyperparameters via RandomizedSearch...")
        tuner = RandomizedSearchCV(
            base_regressor, 
            param_distributions=param_space, 
            n_iter=10, 
            scoring='neg_root_mean_squared_error', 
            cv=3, 
            n_jobs=-1
        )
        
        tuner.fit(x_train, y_train)
        self.trained_engine = tuner.best_estimator_

        # Final Evaluation
        holdout_preds = self.trained_engine.predict(x_test)
        final_rmse = np.sqrt(mean_squared_error(y_test, holdout_preds))
        final_r2 = r2_score(y_test, holdout_preds)

        print(f"--- Training Results ---")
        print(f"RMSE: {final_rmse:.2f}")
        print(f"R2 Score: {final_r2:.3f}")
        print(f"Best Params: {tuner.best_params_}")

    def save_model_artifacts(self):
        """Serializes the model and feature names for deployment."""
        if not os.path.exists("../models"):
            os.makedirs("../models")
            
        joblib.dump(self.trained_engine, MODEL_OUTPUT)
        joblib.dump(self.column_schema, FEATURE_SCHEMA)
        print(f"[+] Model and Schema exported to ../models/")

    def visualize_impact_factors(self):
        """Plots top 10 features influencing processing time."""
        weights = self.trained_engine.feature_importances_
        rank_df = pd.DataFrame({'feature': self.column_schema, 'weight': weights})
        rank_df = rank_df.sort_values(by='weight', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='weight', y='feature', data=rank_df, palette='viridis')
        plt.title('Predictive Influence: Top 10 Features')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    trainer = VisaModelTrainer(DATA_SOURCE)
    
    # 1. Transform Data
    x_train, x_test, y_train, y_test = trainer.load_and_transform()
    
    # 2. Train & Tune
    trainer.optimize_and_train(x_train, x_test, y_train, y_test)
    
    # 3. Analyze
    trainer.visualize_impact_factors()
    
    # 4. Export
    trainer.save_model_artifacts()


# import pandas as pd
# import numpy as np
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor

# # Configuration
# SOURCE_DATA = "data\\VisaFile.csv"
# MODEL_DIR = "../models"

# def main():
#     # 1. Load Data
#     print("[*] Loading dataset...")
#     df = pd.read_csv(SOURCE_DATA, encoding="latin1")
    
#     # Standardize column names
#     df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

#     # 2. Final Engineering/Cleanup for Training
#     # Ensure processing_time_days is the target
#     if 'processing_time_days' not in df.columns:
#         print("[-] Error: 'processing_time_days' not found. Check Milestone 1/2 output.")
#         return

#     # Drop non-predictive columns
#     irrelevant_cols = ['case_id', 'case_number', 'employer_name', 'work_city', 
#                        'case_received_date', 'decision_date', 'job_title']
#     df = df.drop(columns=[c for c in irrelevant_cols if c in df.columns])

#     # Convert categorical to dummy variables
#     df = pd.get_dummies(df, drop_first=True)
    
#     # Split Features and Target
#     X = df.drop('processing_time_days', axis=1)
#     y = df['processing_time_days']
    
#     feature_names = X.columns.tolist()

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # 3. Model Selection and Training
#     print("[*] Training Boosting Engines...")
    
#     # We use XGBoost as our primary engine
#     xgb_engine = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    
#     # Hyperparameter Tuning
#     param_dist = {
#         'n_estimators': [100, 300, 500],
#         'max_depth': [4, 6, 10],
#         'learning_rate': [0.01, 0.05, 0.1]
#     }
    
#     tuner = RandomizedSearchCV(xgb_engine, param_distributions=param_dist, n_iter=5, cv=3, scoring='neg_root_mean_squared_error')
#     tuner.fit(X_train, y_train)
    
#     best_model = tuner.best_estimator_
    
#     # 4. Evaluation
#     preds = best_model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, preds))
#     r2 = r2_score(y_test, preds)
    
#     print(f"\n--- Model Performance ---")
#     print(f"RMSE: {round(rmse, 2)} Days")
#     print(f"R2 Score: {round(r2, 3)}")

#     # 5. Save Artifacts
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)
        
#     joblib.dump(best_model, os.path.join(MODEL_DIR, "visa_model.pkl"))
#     joblib.dump(rmse, os.path.join(MODEL_DIR, "model_rmse.pkl"))
#     joblib.dump(feature_names, os.path.join(MODEL_DIR, "model_features.pkl"))
    
#     print(f"\n[+] Success: All 3 artifacts saved in {MODEL_DIR}")

# if __name__ == "__main__":
#     main()