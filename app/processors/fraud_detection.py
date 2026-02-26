import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import joblib
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Production inference - models should be pre-trained
MODEL_DIR = Path(__file__).parent.parent / 'models'

# Load pre-trained models (train separately using scripts/train.py)
try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / 'fraud_xgboost.json'))
    iso_model = joblib.load(MODEL_DIR / 'fraud_iso_forest.pkl')
    scaler = joblib.load(MODEL_DIR / 'fraud_scaler.pkl')
    ohe = joblib.load(MODEL_DIR / 'fraud_ohe.pkl')
except FileNotFoundError:
    xgb_model = None
    iso_model = None
    scaler = None
    ohe = None

def predict_fraud_transaction(new_tx_dict, threshold=0.5):
    """Production inference for fraud detection"""
    if not all([xgb_model, iso_model, scaler, ohe]):
        return "UNKNOWN", 0.0
    
    df_new = pd.DataFrame([new_tx_dict])
    
    # Time features
    df_new['hour'] = (df_new['step'] - 1) % 24
    df_new['is_night'] = df_new['hour'].between(22, 23) | df_new['hour'].between(0, 4)
    
    # Log amount
    df_new['log_amount'] = np.log1p(df_new['amount'])
    df_new['is_high_amount'] = (df_new['amount'] > 200000).astype(int)
    
    # Type one-hot
    type_encoded = ohe.transform(df_new[['type']])
    df_new = pd.concat([df_new, pd.DataFrame(type_encoded, columns=ohe.get_feature_names_out())], axis=1)
    
    # Simple velocity/graph
    df_new['orig_tx_count_last_24h'] = 1
    df_new['orig_amount_sum_last_24h'] = df_new['amount']
    df_new['orig_velocity_last_24h'] = df_new['amount']
    df_new['orig_out_degree'] = 1
    df_new['dest_in_degree'] = 1
    df_new['is_merchant'] = df_new['nameDest'].str.startswith('M').astype(int)
    
    # Drop non-features
    df_new = df_new.drop(['step', 'type', 'nameOrig', 'nameDest'], axis=1)
    
    # Scale
    numeric_cols = df_new.select_dtypes(include=np.number).columns
    df_new[numeric_cols] = scaler.transform(df_new[numeric_cols])
    
    # Iso score
    iso_score = iso_model.decision_function(df_new)
    df_new['iso_anomaly_score'] = iso_score
    
    # XGBoost prob
    prob = xgb_model.predict_proba(df_new)[0][1]
    
    return "FRAUD" if prob >= threshold else "LEGIT", round(prob * 100, 2)