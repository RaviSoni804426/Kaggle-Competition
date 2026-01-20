import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from .config import MODELS_DIR, OUTPUT_DIR, MODEL_FILENAME, TICKERS, PROB_THRESHOLD
from .data_loader import download_data, load_data
from .indicators import add_features

def run_weekly_prediction():
    print("--- Running Weekly Swing Trading Prediction ---")
    
    # 1. Load Model
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run 'src/train.py' first.")
        return
        
    clf = joblib.load(model_path)
    
    # 2. Refresh Data
    download_data()
    
    predictions = []
    
    feature_cols = [
        'Ret_5d', 
        'SMA_10_Dist', 'SMA_20_Dist', 'SMA_50_Dist', 
        'RSI_14', 
        'Vol_Pct_Change'
    ]
    
    print("\n--- Analyzing Stocks ---")
    
    for ticker in TICKERS:
        try:
            df = load_data(ticker)
            if df.empty:
                continue
                
            # Compute features on full history to ensure indicators are correct
            df = add_features(df)
            
            # Get latest available data (Friday Close)
            latest = df.iloc[-1]
            last_date = df.index[-1]
            
            # Extract features for prediction
            # Reshape to 2D array: (1, n_features)
            X_new = latest[feature_cols].values.reshape(1, -1)
            
            # Predict
            prob = clf.predict_proba(X_new)[0][1]  # Probability of Class 1 (Upside)
            
            current_price = latest['Close']
            
            print(f"{ticker}: Probability {prob:.1%}")
            
            # Filter Candidates
            if prob >= PROB_THRESHOLD:
                predictions.append({
                    'Stock Name': ticker,
                    'Entry Price': current_price,
                    'Model Probability': prob,
                    'Last Date': last_date
                })
                
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            
    # 3. Apply Trading Rules
    # Sort by probability (descending)
    predictions.sort(key=lambda x: x['Model Probability'], reverse=True)
    
    # Cap at Max 2 trades
    selected_trades = predictions[:2]
    
    # 4. Generate Output
    if not selected_trades:
        print("\nResult: NO TRADE THIS WEEK (No stocks met >60% probability criteria)")
        return
        
    print(f"\nResult: Found {len(selected_trades)} Potential Trades")
    
    trade_plan = []
    for trade in selected_trades:
        entry = trade['Entry Price']
        stop_loss = entry * 0.98  # 2% Risk
        target = entry * 1.05     # 5% Target
        
        trade_plan.append({
            'Stock Name': trade['Stock Name'],
            'Entry Price': round(entry, 2),
            'Stop Loss': round(stop_loss, 2),
            'Target': round(target, 2),
            'Model Probability': f"{trade['Model Probability']:.1%}",
            'Upside/Risk': "2.5:1",
            'Analysis Date': trade['Last Date'].strftime('%Y-%m-%d')
        })
        
    # Create DataFrame
    plan_df = pd.DataFrame(trade_plan)
    
    # Save to Excel
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(OUTPUT_DIR, f"Weekly_Trade_Plan_{timestamp}.xlsx")
    
    plan_df.to_excel(output_file, index=False)
    
    print("\n---------------------------------------------------")
    print(plan_df.to_string(index=False))
    print("---------------------------------------------------")
    print(f"\nTrade plan saved to: {output_file}")
    
if __name__ == "__main__":
    run_weekly_prediction()
