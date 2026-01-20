import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score
from .config import MODELS_DIR, TICKERS, MODEL_FILENAME, RANDOM_STATE
from .data_loader import load_data, download_data
from .indicators import add_features

def create_target(df, horizon=7, target_pct=0.03):
    """
    Creates target label: 1 if Max High in next 'horizon' days >= Current Close * (1 + target_pct)
    """
    # Forward looking max high (next 7 days)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    future_max_high = df['High'].shift(-1).rolling(window=indexer, min_periods=1).max()
    
    # Target condition
    df['Target'] = (future_max_high >= df['Close'] * (1 + target_pct)).astype(int)
    
    # Drop the last 'horizon' rows
    df = df.iloc[:-horizon]
    
    return df

def train_model():
    print("--- Starting Training Pipeline (XGBoost) ---")
    
    # 1. Aggregate Data
    combined_train_data = []
    
    # Ensure data exists
    download_data()
    
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        df = load_data(ticker)
        
        # Add Features
        df = add_features(df)
        
        # Add Target
        df = create_target(df)
        
        combined_train_data.append(df)
        
    full_df = pd.concat(combined_train_data)
    full_df.dropna(inplace=True)
    
    # 2. Select Features for Training
    feature_cols = [
        'Ret_5d', 
        'SMA_10_Dist', 'SMA_20_Dist', 'SMA_50_Dist', 
        'RSI_14', 
        'Vol_Pct_Change'
    ]
    
    X = full_df[feature_cols]
    y = full_df['Target']
    
    print(f"\nTotal Samples: {len(X)}")
    
    # Calculate scale_pos_weight for class imbalance
    # formula: sum(negative instances) / sum(positive instances)
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"Class Balance: {y.value_counts(normalize=True).to_dict()}")
    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

    # 3. Time-Series Split
    # Sort by Date index before splitting.
    full_df = full_df.sort_index()
    X = full_df[feature_cols]
    y = full_df['Target']
    
    # Use last 15% for validation
    split_idx = int(len(X) * 0.85)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} samples, Validating on {len(X_test)} samples")
    
    # 4. Train Model (XGBoost)
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,            # Keep it shallow to prevent overfitting
        learning_rate=0.05,     # Lower rate for better generalization
        subsample=0.8,          # Randomly sample rows
        colsample_bytree=0.8,   # Randomly sample columns
        scale_pos_weight=scale_pos_weight, # Handle class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    print("\n--- Model Evaluation (Test Set) ---")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    
    # Check strict precision (Trade only if prob > 60%)
    high_conf_idx = np.where(y_proba >= 0.60)[0]
    if len(high_conf_idx) > 0:
        real_precision = precision_score(y_test.iloc[high_conf_idx], (y_proba[high_conf_idx] >= 0.60).astype(int))
        print(f"Precision at 60% Confidence Threshold: {real_precision:.2f} ({len(high_conf_idx)} trades generated)")
    else:
        print("No trades met the 60% confidence threshold in test set.")
    
    # 6. Save
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    train_model()
