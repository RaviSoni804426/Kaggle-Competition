# Stock Decision Assistant

An AI-powered stock decision assistant for NSE (India) stocks that provides data-driven investment plans.

## ⚠️ Disclaimer

**This is NOT financial advice.** This tool is for educational and informational purposes only. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

## Features

- Analyze any NSE stock symbol
- Select holding period between 3-15 days
- Get data-driven recommendations:
  - BUY / AVOID decision
  - Entry price, Stop-loss, Target
  - Confidence percentage
  - Plain-language explanation

## Architecture

```
/data        → Data fetching & validation
/features    → Feature engineering
/models      → ML models (XGBoost)
/rules       → Risk & decision rules
/llm         → Explanation layer
/api         → FastAPI backend
/ui          → Frontend (Streamlit)
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models (Required First)

```bash
python -m models.train
```

This will:
- Fetch 5+ years of historical data for 40+ NSE stocks
- Train XGBoost models for each horizon (3, 5, 7, 10, 15 days)
- Save models to `models/saved/`
- Display training metrics (accuracy, ROC-AUC, etc.)

Training takes approximately 5-10 minutes depending on your internet connection.

### 3. Run the API Server

```bash
python -m api.main
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

### 4. Run the UI (Optional)

In a new terminal:

```bash
streamlit run ui/app.py
```

The UI will open at `http://localhost:8501`

## Usage Examples

### API Usage

```bash
# Generate plan via curl
curl -X POST http://localhost:8000/generate-plan \
  -H "Content-Type: application/json" \
  -d '{"stock": "TCS", "days": 7}'
```

### Python Usage

```python
from data.fetcher import StockDataFetcher
from data.validator import DataValidator
from models.predictor import StockPredictor
from rules.engine import RulesEngine

# Initialize components
fetcher = StockDataFetcher()
validator = DataValidator()
predictor = StockPredictor()
rules_engine = RulesEngine()

# Fetch and validate data
df, error = fetcher.fetch("TCS")
df = validator.clean_data(df)

# Get prediction
ml_output = predictor.predict(df, horizon=7)

# Apply rules
plan = rules_engine.evaluate(ml_output, horizon=7, latest_price=ml_output["latest_close"])

print(f"Decision: {plan.decision}")
print(f"Confidence: {plan.confidence}%")
print(f"Entry: ₹{plan.entry_price}")
print(f"Stop-Loss: ₹{plan.stop_loss}")
print(f"Target: ₹{plan.target_price}")
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI API Key (optional - for LLM explanations)
OPENAI_API_KEY=your_key_here

# Model (optional)
OPENAI_MODEL=gpt-4o-mini
```

Note: The system works without an OpenAI key - it will use template-based explanations instead.

### Config File

Edit `config.py` to customize:

- Risk thresholds
- Target thresholds per horizon
- XGBoost parameters
- Validation rules

## Technical Details

### Data Pipeline
- **Source**: Yahoo Finance (NSE stocks with `.NS` suffix)
- **Minimum History**: 5 years (~1000 trading days)
- **Validation**: Liquidity, price gaps, missing data checks

### Features (Strictly Defined)
- N-day returns (1, 3, 5, 10 days)
- SMA 10 / 20 / 50
- RSI (14)
- Volatility (20-day annualized)
- Volume % change
- Volume SMA ratio
- Price above/below SMA 50

### ML Model
- **Algorithm**: XGBoost Classifier
- **One model per horizon**: 3, 5, 7, 10, 15 days
- **Target**: Binary (1 if return >= horizon-scaled threshold)
- **Training**: 40+ diversified NSE stocks across sectors

### Risk Rules
- Probability < 55% → AVOID
- Price below SMA 50 (with weak signal) → AVOID
- RSI > 80 (overbought) → Reduced confidence
- Stop-loss: ~2% below entry
- Risk-reward minimum: 1.5:1

## API Reference

### POST /generate-plan

Generate investment plan for a stock.

**Request:**
```json
{
  "stock": "INFOSYS",
  "days": 7
}
```

**Response:**
```json
{
  "decision": "BUY",
  "confidence": 65,
  "entry": 1595.0,
  "stop_loss": 1563.1,
  "target": 1674.75,
  "explanation": "...",
  "stock": "INFOSYS",
  "days": 7,
  "risk_reward_ratio": 2.5,
  "position_risk_pct": 2.0,
  "ml_probability": 0.65,
  "ml_trend": "bullish",
  "timestamp": "2024-01-20T10:30:00",
  "warnings": ["..."]
}
```

### GET /health

Health check endpoint.

### GET /available-horizons

Get list of available prediction horizons.

### GET /stock-info/{symbol}

Get basic information about a stock.

### GET /validate/{symbol}

Validate if a stock has sufficient data.

## Project Structure

```
Stock Price/
├── config.py              # Central configuration
├── requirements.txt       # Dependencies
├── README.md             # This file
│
├── data/                 # Data layer
│   ├── __init__.py
│   ├── fetcher.py        # Yahoo Finance data fetcher
│   └── validator.py      # Data validation
│
├── features/             # Feature engineering
│   ├── __init__.py
│   └── engineer.py       # Technical indicators
│
├── models/               # ML models
│   ├── __init__.py
│   ├── predictor.py      # Model inference
│   ├── train.py          # Training script
│   └── saved/            # Saved model files
│       └── xgb_Xd.pkl
│
├── rules/                # Decision rules
│   ├── __init__.py
│   └── engine.py         # Risk & rules engine
│
├── llm/                  # LLM explanations
│   ├── __init__.py
│   └── explainer.py      # Plan explainer
│
├── api/                  # FastAPI backend
│   ├── __init__.py
│   └── main.py           # API endpoints
│
└── ui/                   # Frontend
    ├── __init__.py
    └── app.py            # Streamlit app
```

## Constraints

- ✅ Local execution only (no cloud dependencies)
- ✅ Python backend
- ❌ No Google Colab
- ❌ No brokerage API execution
- ❌ No deep learning models
- ❌ No future guarantees
- ✅ Designed to avoid look-ahead bias

## Troubleshooting

### "No models available"
Run `python -m models.train` first to train the models.

### "Cannot connect to API"
Ensure the API server is running: `python -m api.main`

### "Data fetch failed"
- Check internet connection
- Verify the stock symbol is valid for NSE
- Some stocks may have insufficient history

### "Validation failed"
The stock may be:
- Too illiquid (low volume)
- Too new (insufficient history)
- Have data quality issues

## License

This project is for educational purposes only. Use at your own risk.
