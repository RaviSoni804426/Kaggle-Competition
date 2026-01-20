import yfinance as yf
import pandas as pd
import os
import time
from .config import DATA_DIR, TICKERS

def download_data(lookback_years=8):
    """
    Downloads daily data for all tickers in config.
    Saves individual CSVs to data/ directory.
    """
    print(f"Downloading data for: {TICKERS}")
    
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=lookback_years)).strftime('%Y-%m-%d')
    
    for ticker in TICKERS:
        print(f"Fetching {ticker}...", end=" ")
        try:
            # Download
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            
            if df.empty:
                print("FAILED (Empty Data)")
                continue
                
            # If using auto_adjust=True, columns are just Open, High, Low, Close, Volume
            # Standardize names just in case
            if 'Adj Close' in df.columns:
                df = df.drop(columns=['Close']).rename(columns={'Adj Close': 'Close'})
                
            # Flatten columns if multi-index (Price, Ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure columns are standardized
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Save
            file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            df.to_csv(file_path)
            print(f"Done. Rows: {len(df)}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            
    print("\nData download complete.")

def load_data(ticker):
    """
    Loads data from local CSV.
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data for {ticker} not found. Run download_data first.")
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

if __name__ == "__main__":
    download_data()
