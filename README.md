# ML-Driven Automated Crypto Trading System

A production-ready automated trading system that handles data ingestion from Polygon.io, ML-based trend prediction using XGBoost, realistic backtesting, and live trading execution via CCXT.

## 🚀 Features

- **Data Pipeline**: Polygon.io flat file integration for historical crypto data
- **ML Pipeline**: XGBoost 3-class classifier for trend prediction (up/down/flat)
- **Feature Engineering**: 50+ technical indicators with causal design (no lookahead bias)
- **Backtesting**: Realistic simulation with fees, slippage, and volatility-based position sizing
- **Live Trading**: CCXT integration with Coinbase Advanced Trade and Kraken
- **Risk Management**: Position sizing, stop-losses, and portfolio limits
- **24/7 Automation**: APScheduler for continuous operation
- **Monitoring**: Comprehensive logging and performance metrics

## 📊 Recent Performance

- **Model Accuracy**: 88.8% overall accuracy
- **Win Rate**: 91.5% in backtesting
- **Profit Factor**: 11.83
- **Total Trades**: 1,180 simulated trades

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- uv package manager
- Polygon.io API key
- Exchange API keys (Coinbase/Kraken for live trading)

### Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:iklobato/trade.git
   cd trade
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure API keys**:
   - Update `app/config.yaml` with your Polygon.io API key
   - Set environment variables for exchange API keys:
     ```bash
     export COINBASE_API_KEY="your_coinbase_key"
     export COINBASE_SECRET="your_coinbase_secret"
     export KRAKEN_API_KEY="your_kraken_key"
     export KRAKEN_SECRET="your_kraken_secret"
     ```

## 🎯 Usage

### Test Connection
```bash
uv run python app/main.py --test-connection
```

### Download Data
```bash
uv run python app/main.py --download-data
```

### Train Model
```bash
uv run python app/main.py --train-model
```

### Run Backtest
```bash
uv run python app/main.py --backtest-only
```

### Paper Trading
```bash
uv run python app/main.py --mode paper
```

### Live Trading
```bash
uv run python app/main.py --mode live
```

## 📁 Project Structure

```
app/
├── config.yaml              # System configuration
├── main.py                  # Main entry point
├── data/                    # Data ingestion
│   ├── polygon_flatfiles_downloader.py
│   └── polygon_loader.py
├── features/                # Feature engineering
│   └── engineer.py
├── labels/                  # Label generation
│   └── targets.py
├── models/                  # ML models
│   └── classifier.py
├── backtest/                # Backtesting engine
│   ├── simulator.py
│   └── metrics.py
├── execution/               # Live trading
│   ├── gateway.py
│   └── trade_manager.py
├── orchestration/           # Scheduling
│   └── scheduler.py
├── utils/                   # Utilities
│   ├── seed.py
│   ├── splits.py
│   └── costs.py
└── tests/                   # Test suite
    └── test_integration.py
```

## 🔧 Configuration

Key configuration parameters in `app/config.yaml`:

- **Symbol**: Trading pair (default: BTC/USD)
- **Timeframe**: Data frequency (default: 1min)
- **Model Parameters**: XGBoost hyperparameters
- **Risk Settings**: Position sizing and limits
- **Exchange Settings**: Primary/secondary exchanges
- **Scheduler**: Retraining and execution intervals

## 📈 Model Performance

### Top Features by Importance
1. Volume SMA 60 (676.0)
2. Volatility 60 (584.0)
3. ADX (577.0)
4. Volatility Price Ratio (574.0)
5. Volume SMA 20 (536.0)

### Classification Performance
- **Down Trend**: Precision=98.0%, Recall=46.7%, F1=63.3%
- **Flat Trend**: Precision=88.7%, Recall=96.3%, F1=92.3%
- **Up Trend**: Precision=88.9%, Recall=73.5%, F1=80.5%

## ⚠️ Risk Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues, please open an issue on GitHub.