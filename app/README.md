# ML-Driven Automated Crypto Trading System

A production-ready automated trading system that uses machine learning to predict short-term crypto price movements and execute trades via CCXT (Coinbase Advanced Trade and Kraken).

## 🚀 Features

- **Data Ingestion**: Automated download and processing of Polygon.io crypto flat files
- **ML Pipeline**: XGBoost-based trend classification with causal feature engineering
- **Backtesting**: Realistic simulation with fees, slippage, and volatility-based position sizing
- **Live Trading**: CCXT integration with Coinbase Advanced Trade and Kraken
- **Automation**: 24/7 operation with APScheduler for hourly predictions and daily retraining
- **Risk Management**: Comprehensive position sizing, stop-loss, and portfolio management
- **Monitoring**: Detailed logging, performance metrics, and trade tracking

## 📁 Project Structure

```
app/
├── config.yaml                    # System configuration
├── main.py                       # Main entry point
├── data/                         # Data ingestion
│   ├── polygon_flatfiles_downloader.py
│   └── polygon_loader.py
├── features/                     # Feature engineering
│   └── engineer.py
├── labels/                       # Label generation
│   └── targets.py
├── models/                       # ML models
│   └── classifier.py
├── backtest/                     # Backtesting engine
│   ├── simulator.py
│   └── metrics.py
├── execution/                    # Live trading
│   ├── gateway.py
│   └── trade_manager.py
├── orchestration/                # Scheduling
│   └── scheduler.py
├── utils/                        # Utilities
│   ├── seed.py
│   ├── splits.py
│   └── costs.py
├── tests/                        # Test suite
│   └── test_integration.py
├── artifacts/                    # Model artifacts
├── reports/                      # Performance reports
└── execution/logs/               # Trading logs
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- uv (Python package manager)
- Polygon.io API key
- Exchange API keys (Coinbase Advanced Trade, Kraken)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /Users/iklo/crypto
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   ```bash
   export POLYGON_API_KEY="2XQxI4hHcFwcGA3wZQJ53bRpt7mJ0hig"
   export COINBASE_API_KEY="your_coinbase_api_key"
   export COINBASE_SECRET="your_coinbase_secret"
   export COINBASE_PASSPHRASE="your_coinbase_passphrase"
   export KRAKEN_API_KEY="your_kraken_api_key"
   export KRAKEN_SECRET="your_kraken_secret"
   ```

4. **Configure the system:**
   Edit `app/config.yaml` to adjust parameters like:
   - Trading symbol (default: BTC/USD)
   - Risk parameters
   - Model hyperparameters
   - Exchange settings

## 🎯 Quick Start

### 1. Test Connection
```bash
uv run python app/main.py --test-connection
```

### 2. Download Historical Data
```bash
uv run python app/main.py --download-data
```

### 3. Train Model
```bash
uv run python app/main.py --train-model
```

### 4. Run Backtest
```bash
uv run python app/main.py --backtest-only
```

### 5. Start Live Trading
```bash
uv run python app/main.py --mode live
```

## 📊 System Architecture

### Data Flow

1. **Data Ingestion**: Polygon.io flat files → Local cache → OHLCV dataset
2. **Feature Engineering**: Technical indicators → Causal features (no lookahead bias)
3. **Label Generation**: Forward returns → Classification labels (-1, 0, 1)
4. **Model Training**: XGBoost classifier → Trend prediction probabilities
5. **Backtesting**: Simulated trading → Performance metrics
6. **Live Trading**: Real-time predictions → CCXT order execution

### ML Pipeline

- **Features**: 50+ technical indicators (EMAs, RSI, MACD, Bollinger Bands, ATR, etc.)
- **Labels**: 3-class classification (up/down/flat) based on 10-minute forward returns
- **Model**: XGBoost with optimized hyperparameters
- **Validation**: Time-series cross-validation with causal splits

### Risk Management

- **Position Sizing**: Volatility-based sizing with Kelly criterion
- **Stop Loss**: ATR-based trailing stops
- **Take Profit**: Risk-reward ratio targets
- **Portfolio Limits**: Maximum position size and daily loss limits

## 🔧 Configuration

### Key Parameters in `config.yaml`

```yaml
# Trading Configuration
symbol: "BTC/USD"
timeframe: "1min"
horizon_minutes: 10

# Risk Management
risk:
  risk_per_trade: 0.005    # 0.5% risk per trade
  max_pos: 1.0             # Maximum position size
  vol_window: 60           # Volatility window

# Model Parameters
model:
  type: "xgb"
  params:
    max_depth: 5
    eta: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    n_rounds: 400

# Execution Settings
execution:
  engine: "ccxt"
  primary_exchange: "coinbase"
  secondary_exchange: "kraken"
  trade_mode: "live"       # or "paper"
```

## 📈 Usage Examples

### Paper Trading Mode
```bash
uv run python app/main.py --mode paper --symbol BTC/USD
```

### Live Trading with Custom Symbol
```bash
uv run python app/main.py --mode live --symbol ETH/USD
```

### Dry Run (Test without trades)
```bash
uv run python app/main.py --dry-run
```

### Backtest Only
```bash
uv run python app/main.py --backtest-only
```

## 📊 Performance Monitoring

### Real-time Monitoring

The system provides real-time monitoring through:

- **Console Logs**: Live trading status, predictions, and trade execution
- **Log Files**: Detailed logs in `app/execution/logs/YYYY-MM-DD.log`
- **Status Files**: JSON status updates in `app/reports/trading_status.json`
- **Performance Reports**: Comprehensive metrics in `app/reports/`

### Key Metrics

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss ratio

### Reports Generated

- `trade_log.csv`: All executed trades with PnL
- `equity_curve.csv`: Portfolio value over time
- `summary.json`: Performance metrics
- `trading_status.json`: Current system status

## 🧪 Testing

### Run Tests
```bash
uv run pytest app/tests/
```

### Test Coverage
```bash
uv run pytest --cov=app --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Mock Tests**: Exchange API simulation
- **Data Validation**: Feature causality and label alignment

## 🔒 Security & Risk

### API Key Security

- Store API keys in environment variables
- Never commit keys to version control
- Use read-only keys for backtesting
- Enable IP whitelisting on exchanges

### Risk Controls

- **Position Limits**: Maximum position size per trade
- **Daily Loss Limits**: Automatic shutdown on excessive losses
- **Stop Losses**: Automatic position closure on adverse moves
- **Slippage Protection**: Buffer for order execution

### Paper Trading

Always start with paper trading mode:
```bash
uv run python app/main.py --mode paper
```

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API keys and permissions
   - Check network connectivity
   - Ensure exchange API is operational

2. **Data Download Issues**
   - Verify Polygon API key
   - Check data availability for date range
   - Ensure sufficient disk space

3. **Model Training Errors**
   - Verify sufficient historical data
   - Check feature generation
   - Ensure label alignment

4. **Trading Execution Errors**
   - Verify exchange permissions
   - Check account balance
   - Ensure symbol is supported

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
uv run python app/main.py --mode paper
```

### Log Analysis

Check system logs:
```bash
tail -f trading_system.log
tail -f app/execution/logs/$(date +%Y-%m-%d).log
```

## 📚 Advanced Usage

### Custom Feature Engineering

Add custom features in `app/features/engineer.py`:

```python
def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Your custom feature logic here
    df['custom_feature'] = df['close'].rolling(20).std()
    return df
```

### Custom Risk Management

Modify risk parameters in `app/execution/trade_manager.py`:

```python
def _calculate_position_size(self, symbol: str, price: float) -> float:
    # Your custom position sizing logic
    return custom_size
```

### Model Hyperparameter Tuning

Update model parameters in `config.yaml`:

```yaml
model:
  params:
    max_depth: 6
    eta: 0.03
    subsample: 0.9
    colsample_bytree: 0.9
    n_rounds: 500
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## ⚠️ Disclaimer

- **Educational Purpose**: This system is for educational and research purposes
- **No Financial Advice**: Not financial advice or investment recommendations
- **Risk Warning**: Cryptocurrency trading involves significant risk
- **Past Performance**: Past performance does not guarantee future results
- **Use at Own Risk**: Use at your own risk and responsibility

## 📄 License

This project is provided under the MIT License. See LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review system logs
3. Run diagnostic tests
4. Create an issue with detailed information

---

**Ready to start trading? Run `uv run python app/main.py --mode paper` to begin with paper trading!**
