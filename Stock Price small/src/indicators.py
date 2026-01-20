import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Fill NaN caused by initial window
    return rsi.fillna(50) 

def add_features(df):
    """
    Adds technical indicators to the dataframe.
    Expects DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    df = df.copy()
    
    # Ensure Close is float
    close_prices = df['Close'].astype(float)
    
    # 1. 5-Day Returns
    # (Close - Close_5d_ago) / Close_5d_ago
    df['Ret_5d'] = close_prices.pct_change(periods=5)
    
    # 2. SMAs (10, 20, 50)
    df['SMA_10'] = close_prices.rolling(window=10).mean()
    df['SMA_20'] = close_prices.rolling(window=20).mean()
    df['SMA_50'] = close_prices.rolling(window=50).mean()
    
    # Normalize SMAs relative to Close to make them stationary-ish features for ML
    # (Price / SMA) - 1.0 helps the tree compare relative extensions rather than raw price levels
    df['SMA_10_Dist'] = (close_prices / df['SMA_10']) - 1
    df['SMA_20_Dist'] = (close_prices / df['SMA_20']) - 1
    df['SMA_50_Dist'] = (close_prices / df['SMA_50']) - 1
    
    # 3. RSI (14)
    df['RSI_14'] = calculate_rsi(close_prices, window=14)
    
    # 4. Volume Percentage Change
    # Compare current volume to 20-day average volume
    df['Vol_Avg_20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Pct_Change'] = (df['Volume'] / df['Vol_Avg_20']) - 1
    
    # Cleanup NaNs created by rolling windows
    df.dropna(inplace=True)
    
    return df
