# ML Swing Trading System

A local, privacy-focused Machine Learning trading system for the Indian Stock Market (NSE).
Target Stocks: HDFC Bank, Reliance, Infosys.

## 1. Setup

### Prerequisites
- Python 3.8+ installed
- Internet connection (for data download)

### Installation
1. Open a terminal/command prompt in this folder.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Usage

### Option A: Interactive Menu (Recommended)
Run the main script:
```bash
python main.py
```
- Select **Option 1** to Download Data and Train the Model (Do this first, and every 3 months).
- Select **Option 2** on **Sunday Evenings** to generate your Weekly Trade Plan.

### Option B: Manual Execution
- **Training**: `python src/train.py`
- **Prediction**: `python src/predict.py`

## 3. Workflow

1. **First Time Setup**: Run "Train Model". This will download 7+ years of data and build `models/rf_model.pkl`.
2. **Weekly Routine (Sunday)**:
   - Run "Weekly Prediction".
   - The system will fetch Friday's closing data.
   - It outputs a Trade Plan in the console and saves an Excel file to `outputs/`.
3. **Execution**:
   - Open the Excel file.
   - Place Limit Orders for Monday morning based on "Entry Price".
   - Set Stop Loss and Targets immediately.

## 4. Maintenance

- **Retraining**: Run Option 1 (Train) every 3 months to incorporate recent market behavior into the model.
- **Data**: The system automatically updates daily data when running predictions, so no manual data management is needed.

## 5. Strategy Details

- **Model**: XGBoost Classifier
- **Features**: RSI(14), SMA(10, 20, 50), Vol Change, 5-Day Returns.
- **Target**: 3% Upside in next 7 trading days.
- **Risk Management**:
  - Max 2 trades per week.
  - Minimum 60% probability required.
  - 1.5% Risk per trade (calculated via position size, but output gives levels).
  - Stop Loss: ~2%
  - Target: ~5%

## Disclaimer
This system is for educational purposes only. Trading stocks involves risk. The model is probabilistic, not prophetic.
