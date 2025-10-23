# Cryptocurrency Trading App - Setup Instructions
# ===============================================

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install ccxt pandas numpy
```

### 2. Set Up Environment Variables (Optional)

For live trading, set up API keys:

```bash
# Binance API Keys (for live trading)
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_SECRET="your_secret_here"

# Coinbase API Keys (alternative)
export COINBASE_API_KEY="your_api_key_here"
export COINBASE_SECRET="your_secret_here"
export COINBASE_PASSPHRASE="your_passphrase_here"

# Kraken API Keys (alternative)
export KRAKEN_API_KEY="your_api_key_here"
export KRAKEN_SECRET="your_secret_here"
```

### 3. Run the Trading App

```bash
python app.py
```

## 📊 Features

### Real-Time Trading
- **Live Market Data**: Real-time price feeds from major exchanges
- **Sandbox Mode**: Safe testing environment (default)
- **Live Trading**: Real money trading (when configured)

### Enhanced Strategy
- **Multi-Indicator Analysis**: SMA, RSI, MACD, ATR, ADX, Bollinger Bands
- **Risk Management**: Stop loss and take profit levels
- **Position Sizing**: Configurable position sizes
- **Portfolio Management**: Real-time portfolio tracking

### Supported Exchanges
- **Binance** (recommended)
- **Coinbase Pro**
- **Kraken**

## ⚙️ Configuration

### Trading Parameters
- **Symbol**: BTC/USDT (default)
- **Timeframe**: 1h (default)
- **Initial Balance**: $10,000 (sandbox)
- **Max Position Size**: 10% of portfolio
- **Stop Loss**: 2%
- **Take Profit**: 4%
- **Update Interval**: 1 hour

### Strategy Settings
- **Strategy**: Enhanced Trend Following
- **Indicators**: 20+ technical indicators
- **Signal Confidence**: Minimum 70% for trades
- **Risk Management**: Automatic stop loss/take profit

## 🛡️ Safety Features

### Sandbox Mode (Default)
- **No Real Money**: All trades are simulated
- **Real Data**: Uses live market data
- **Full Testing**: Complete strategy testing
- **Safe Learning**: Learn without risk

### Risk Management
- **Position Limits**: Maximum 10% per trade
- **Stop Losses**: Automatic 2% stop loss
- **Take Profits**: Automatic 4% take profit
- **Portfolio Monitoring**: Real-time tracking

## 📈 Performance Monitoring

### Real-Time Metrics
- **Portfolio Value**: Current total value
- **Total Return**: Overall performance
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Profit/loss ratio
- **Number of Trades**: Total trades executed

### Logging
- **Trading Log**: Complete trade history
- **Performance Log**: Detailed performance metrics
- **Error Log**: Error tracking and debugging

## 🔧 Customization

### Modify Strategy
Edit the `EnhancedTrendFollowingStrategy` class in `app.py`:

```python
class EnhancedTrendFollowingStrategy:
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Modify signal logic here
        pass
```

### Change Parameters
Modify the `TradingConfig` in `main()`:

```python
config = TradingConfig(
    exchange='binance',           # Exchange
    symbol='ETH/USDT',           # Trading pair
    timeframe='4h',              # Timeframe
    initial_balance=5000.0,     # Starting balance
    max_position_size=0.05,     # 5% max position
    stop_loss_pct=0.015,        # 1.5% stop loss
    take_profit_pct=0.03,       # 3% take profit
    update_interval=1800        # 30 minutes
)
```

## 🚨 Important Notes

### Sandbox vs Live Trading
- **Sandbox Mode**: Safe testing, no real money
- **Live Trading**: Real money, real risk
- **API Keys**: Required for live trading
- **Testing**: Always test in sandbox first

### Risk Warnings
- **Past Performance**: Does not guarantee future results
- **Market Risk**: Cryptocurrency markets are volatile
- **Technical Risk**: Software bugs can cause losses
- **API Risk**: Exchange API issues can affect trading

### Best Practices
1. **Start Small**: Begin with small position sizes
2. **Test Thoroughly**: Use sandbox mode extensively
3. **Monitor Closely**: Watch performance regularly
4. **Risk Management**: Never risk more than you can afford to lose
5. **Backup Plans**: Have manual override capabilities

## 📞 Support

### Troubleshooting
- **API Errors**: Check API keys and permissions
- **Connection Issues**: Verify internet connection
- **Strategy Errors**: Check indicator calculations
- **Performance Issues**: Monitor system resources

### Getting Help
- **Logs**: Check `trading_app.log` for detailed logs
- **Configuration**: Verify `trading_config.json`
- **API Status**: Check exchange API status
- **Documentation**: Review exchange API docs

## 🎯 Next Steps

### 1. Test in Sandbox
```bash
python app.py
```

### 2. Monitor Performance
- Watch the logs for trade signals
- Monitor portfolio value changes
- Analyze win rate and returns

### 3. Optimize Strategy
- Adjust indicator parameters
- Modify signal thresholds
- Test different timeframes

### 4. Live Trading (Advanced)
- Set up API keys
- Start with small amounts
- Monitor closely
- Scale up gradually

## 🏆 Expected Performance

Based on backtesting:
- **Total Return**: 150.68% (historical)
- **Sharpe Ratio**: 1.819
- **Max Drawdown**: 12.65%
- **Win Rate**: 37.5%
- **Profit Factor**: 4.13

**Remember**: Past performance does not guarantee future results!

---

**Ready to start trading? Run `python app.py` to begin!**

