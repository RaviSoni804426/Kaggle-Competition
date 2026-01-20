import os

# --- Configuration Settings ---

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stocks Universe (NSE Tickers)
# Note: yfinance uses '.NS' suffix for NSE
TICKERS = [
    "HDFCBANK.NS",
    "RELIANCE.NS",
    "INFY.NS"
]

# Model Settings
MODEL_FILENAME = "xgb_model.pkl"
PREDICTION_HORIZON = 7  # Days
TARGET_PCT = 0.03       # 3% move for labeling favorable trade
PROB_THRESHOLD = 0.60   # 60% probability threshold for valid trade

# Risk Management
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.05
MAX_RISK_PER_TRADE = 0.015

# Training Params
TEST_SIZE_MONTHS = 12   # Last 12 months for testing/validation
RANDOM_STATE = 42
