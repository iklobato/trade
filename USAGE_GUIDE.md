# 🚀 ML Trading System Usage Guide

## Quick Start

### 1. **Setup Environment**

```bash
# Clone the repository
git clone git@github.com:iklobato/trade.git
cd trade

# Install dependencies
uv sync

# Set up API keys
export POLYGON_API_KEY="your_polygon_api_key"
export KRAKEN_API_KEY="your_kraken_api_key" 
export KRAKEN_SECRET="your_kraken_secret"
```

### 2. **Configure the System**

Edit `app/config.yaml`:

```yaml
# Update with your API key
polygon_api_key: "your_polygon_api_key"

# Set trading mode
execution:
  exchange: "kraken"
  trade_mode: "paper"  # Start with paper trading!
```

## 📊 **Basic Usage Commands**

### **Test System Components**

```bash
# Test all connections
uv run python app/main.py --test-connection

# Test data download
uv run python app/main.py --download-data

# Test model training
uv run python app/main.py --train-model
```

### **Run Backtesting**

```bash
# Run backtest with current model
uv run python app/main.py --backtest-only

# Run backtest with specific config
uv run python app/main.py --config app/config.yaml --backtest-only
```

### **Start Trading**

```bash
# Paper trading (safe testing)
uv run python app/main.py --mode paper

# Live trading (real money!)
uv run python app/main.py --mode live
```

## 🎯 **Step-by-Step Workflow**

### **Step 1: Download Historical Data**

```bash
uv run python app/main.py --download-data
```

This will:
- Download Polygon.io flat files for BTC/USD and ETH/USD
- Cache data in `app/data/cache/`
- Process and merge daily files

**Expected Output:**
```
📥 Downloading Polygon.io data...
✅ Downloaded 2023-01-01.csv
✅ Downloaded 2023-01-02.csv
...
📊 Total samples: 1,440,000 (1000 days × 1440 minutes)
```

### **Step 2: Train the ML Model**

```bash
uv run python app/main.py --train-model
```

This will:
- Generate 52 technical features
- Create trend labels (up/down/flat)
- Train XGBoost classifier
- Save model to `app/artifacts/`

**Expected Output:**
```
🤖 Training XGBoost model...
📊 Generated 52 features
🏷️  Label distribution: Down: 15%, Flat: 70%, Up: 15%
✅ Model trained - Accuracy: 88.8%
💾 Model saved to app/artifacts/model_2024-01-15/
```

### **Step 3: Run Backtesting**

```bash
uv run python app/main.py --backtest-only
```

This will:
- Simulate trades using the trained model
- Apply realistic fees and slippage
- Generate performance metrics

**Expected Output:**
```
📈 Running backtest...
💰 Total Return: 12.5%
📊 Sharpe Ratio: 1.8
📉 Max Drawdown: 8.2%
🎯 Win Rate: 91.5%
📋 Number of Trades: 1,180
💾 Results saved to app/reports/
```

### **Step 4: Start Paper Trading**

```bash
uv run python app/main.py --mode paper
```

This will:
- Start the trading system in paper mode
- Make simulated trades every hour
- Log all activity to `app/logs/`

**Expected Output:**
```
🚀 Starting ML Trading System (Paper Mode)
⏰ Scheduler started - Running 24/7
📊 Monitoring BTC/USD on Kraken
🔄 Hourly predictions enabled
📈 Daily retraining enabled
```

## 🔧 **Advanced Configuration**

### **Custom Model Parameters**

Edit `app/config.yaml`:

```yaml
model:
  type: "xgb"
  params:
    max_depth: 6        # Increase for more complex patterns
    eta: 0.03          # Lower learning rate
    subsample: 0.9     # More data per tree
    colsample_bytree: 0.9
    n_rounds: 500      # More training rounds
```

### **Risk Management Settings**

```yaml
risk:
  risk_per_trade: 0.01    # Risk 1% per trade
  max_pos: 0.5           # Max 50% of capital
vol_window: 30           # 30-minute volatility window
```

### **Trading Thresholds**

```yaml
thresholds:
  up: 0.003              # Enter long if P(up) > 0.3%
  down: -0.003           # Enter short if P(down) > 0.3%
```

## 📈 **Monitoring and Logs**

### **View Trading Logs**

```bash
# Real-time log monitoring
tail -f app/logs/trading.log

# View daily logs
cat app/logs/2024-01-15.log
```

### **Check Performance**

```bash
# View latest backtest results
cat app/reports/summary.json

# View trade log
head -20 app/reports/trade_log.csv
```

### **Monitor System Status**

```bash
# Check if system is running
ps aux | grep "python app/main.py"

# View system metrics
cat app/logs/system_metrics.json
```

## 🎛️ **Command Line Options**

```bash
# Full command reference
uv run python app/main.py --help

# Available options:
--config PATH          # Custom config file
--mode {paper,live}    # Trading mode
--test-connection      # Test all connections
--download-data        # Download historical data
--train-model          # Train ML model
--backtest-only        # Run backtest only
--symbol SYMBOL        # Override symbol (BTC/USD, ETH/USD)
--start-date DATE      # Override start date
--end-date DATE        # Override end date
```

## 🔄 **Automated Operation**

### **24/7 Trading Setup**

The system runs automatically with APScheduler:

- **Every Hour**: Generate predictions and execute trades
- **Daily at 00:00 UTC**: Retrain model and run backtest
- **Continuous**: Monitor positions and manage risk

### **Systemd Service (Linux)**

Create `/etc/systemd/system/ml-trading.service`:

```ini
[Unit]
Description=ML Trading System
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/path/to/trade
ExecStart=/path/to/trade/.venv/bin/python app/main.py --mode live
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ml-trading
sudo systemctl start ml-trading
sudo systemctl status ml-trading
```

## 📊 **Understanding the Outputs**

### **Model Performance**

- **Accuracy**: Overall prediction accuracy (target: >85%)
- **Precision**: How often "up" predictions are correct
- **Recall**: How often actual "up" moves are caught
- **F1-Score**: Balanced measure of precision and recall

### **Trading Metrics**

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.5)
- **Max Drawdown**: Largest peak-to-trough loss
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### **Feature Importance**

The system shows which features are most predictive:

```
Top Features:
1. volume_sma_60     (676.0) - Volume trend
2. volatility_60     (584.0) - Price volatility  
3. adx              (577.0) - Trend strength
4. bb_width         (470.0) - Bollinger Band width
5. ema_9            (441.0) - Short-term trend
```

## ⚠️ **Safety Guidelines**

### **Start with Paper Trading**

```bash
# Always test first!
uv run python app/main.py --mode paper
```

### **Monitor Closely**

- Check logs daily
- Review performance weekly
- Adjust parameters based on results

### **Risk Management**

- Start with small position sizes
- Set conservative risk limits
- Use stop-losses for live trading

### **Emergency Stop**

```bash
# Kill the trading system
pkill -f "python app/main.py"

# Or use systemctl
sudo systemctl stop ml-trading
```

## 🆘 **Troubleshooting**

### **Common Issues**

**"No data loaded"**
```bash
# Check Polygon API key
echo $POLYGON_API_KEY

# Download data manually
uv run python app/main.py --download-data
```

**"Model not found"**
```bash
# Train model first
uv run python app/main.py --train-model
```

**"Exchange connection failed"**
```bash
# Check Kraken API keys
echo $KRAKEN_API_KEY
echo $KRAKEN_SECRET

# Test connection
uv run python app/main.py --test-connection
```

### **Debug Mode**

```bash
# Run with verbose logging
uv run python app/main.py --mode paper --debug
```

## 📚 **Next Steps**

1. **Start with Paper Trading**: Test the system safely
2. **Analyze Results**: Review backtest performance
3. **Optimize Parameters**: Adjust model and risk settings
4. **Go Live**: Switch to live trading with small amounts
5. **Scale Up**: Increase position sizes as confidence grows

## 🎯 **Success Metrics**

A well-performing system should show:
- **Accuracy**: >85%
- **Sharpe Ratio**: >1.5
- **Win Rate**: >60%
- **Max Drawdown**: <15%
- **Consistent Profits**: Positive returns over time

---

**Ready to start? Run:**
```bash
uv run python app/main.py --mode paper
```

Happy trading! 🚀📈
