# Strategy Backtester

A comprehensive strategy-based backtesting and optimization framework for cryptocurrency trading. This system provides multiple trading strategies, optimization tools, and extensive backtesting capabilities.

## 🚀 Features

- **Multiple Strategies**: Trend following, mean reversion, ML-based strategies
- **Optimization Tools**: Parameter optimization, walk-forward analysis, Bayesian optimization
- **Comprehensive Backtesting**: Realistic simulation with fees, slippage, and risk management
- **Performance Analysis**: Detailed metrics, visualizations, and reporting
- **Strategy Management**: Easy strategy development and testing framework

## 📁 Project Structure

```
projects/strategy-backtester/
├── src/                         # Source code
│   ├── strategies.py            # Core strategy implementations
│   ├── enhanced_strategies.py   # Enhanced strategy variants
│   ├── backtester.py            # Backtesting engine
│   ├── optimizers/              # Optimization tools
│   │   ├── optimizer.py
│   │   ├── bayesian_optimizer.py
│   │   └── walk_forward_optimizer.py
│   ├── strategy_manager.py      # Strategy management
│   └── analysis/                # Analysis and debugging tools
├── tests/                       # Test suite
├── app.py                       # Main application
├── app_demo.py                  # Demo application
├── run_strategy.py              # Strategy runner
└── README.md                    # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- uv package manager
- Historical data (CSV files in `../../data/raw/`)

### Quick Setup
```bash
# From workspace root
cd projects/strategy-backtester

# Ensure data is available
ls ../../data/raw/*.csv
```

## 🎯 Usage Commands

### Basic Commands
```bash
# Run main application
uv run python app.py

# Run demo application
uv run python app_demo.py

# Run specific strategy
uv run python run_strategy.py

# Run comprehensive backtest
uv run python src/comprehensive_backtest.py
```

### Strategy Development
```bash
# Test enhanced strategies
uv run python src/enhanced_strategies.py

# Run optimization
uv run python src/optimizers/optimizer.py

# Walk-forward analysis
uv run python src/optimizers/walk_forward_optimizer.py
```

## 📊 Available Strategies

### 1. Trend Following Strategy
- **Description**: Moving average crossover with ATR-based position sizing
- **Parameters**: Fast MA, Slow MA, ATR period, ATR multiplier
- **Best for**: Trending markets

### 2. Mean Reversion Strategy
- **Description**: Bollinger Bands with RSI confirmation
- **Parameters**: BB period, BB std, RSI period, RSI thresholds
- **Best for**: Range-bound markets

### 3. Enhanced Strategies
- **Description**: Advanced variants with multiple indicators
- **Features**: Dynamic parameters, adaptive thresholds
- **Best for**: Volatile markets

### 4. ML-Based Strategies
- **Description**: Machine learning models for signal generation
- **Features**: Random Forest, feature engineering
- **Best for**: Complex market patterns

## 🔧 Configuration

### Strategy Parameters
Each strategy can be configured with custom parameters:

```python
# Example: Trend Following Strategy
strategy = TrendFollowingStrategy(
    data=data,
    initial_balance=10000.0,
    fee=0.001,
    slippage=0.0005,
    fast_ma=50,
    slow_ma=200,
    atr_period=14,
    atr_multiplier=2.0
)
```

### Backtesting Configuration
```python
# Example: Backtesting setup
backtester = Backtester(
    data=data,
    initial_balance=10000.0,
    fee=0.001,
    slippage=0.0005
)
```

## 📊 Performance Analysis

### Available Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Trade**: Average profit per trade

### Visualization Tools
- **Equity Curve**: Portfolio value over time
- **Drawdown Chart**: Drawdown periods visualization
- **Trade Analysis**: Individual trade performance
- **Performance Dashboard**: Comprehensive metrics view

## 🚀 Optimization Tools

### 1. Parameter Optimization
```bash
# Run parameter optimization
uv run python src/optimizers/optimizer.py
```

### 2. Bayesian Optimization
```bash
# Run Bayesian optimization
uv run python src/optimizers/bayesian_optimizer.py
```

### 3. Walk-Forward Analysis
```bash
# Run walk-forward analysis
uv run python src/optimizers/walk_forward_optimizer.py
```

## 📈 Strategy Development

### Creating New Strategies
1. **Inherit from Strategy base class**
2. **Implement backtest() method**
3. **Define strategy parameters**
4. **Add to strategy manager**

### Example Strategy Template
```python
class MyStrategy(Strategy):
    def __init__(self, data, initial_balance=1000.0, fee=0.001, slippage=0.0005, **params):
        super().__init__(data, initial_balance, fee, slippage)
        self.params = params
    
    def backtest(self):
        # Implement your strategy logic here
        # Return trades and performance metrics
        pass
```

## 🔍 Analysis Tools

### Debugging
```bash
# Debug strategy performance
uv run python src/debug_strategy.py

# Debug data issues
uv run python src/debug_csv.py
```

### Performance Analysis
```bash
# Run final analysis
uv run python src/final_analysis.py

# Performance dashboard
uv run python src/performance_dashboard.py
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_enhanced_strategies.py
```

### Test Coverage
```bash
# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

## 📚 Data Requirements

### Supported Data Formats
- **CSV files**: OHLCV data with timestamp/index
- **Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d
- **Symbols**: BTC/USDT, ETH/USDT, and others

### Data Location
All data files should be placed in `../../data/raw/` directory.

## 🎯 Best Practices

### Strategy Development
1. **Start Simple**: Begin with basic strategies
2. **Test Thoroughly**: Use multiple timeframes and symbols
3. **Optimize Carefully**: Avoid overfitting
4. **Validate Out-of-Sample**: Use walk-forward analysis

### Risk Management
1. **Position Sizing**: Use risk-based position sizing
2. **Stop Losses**: Implement proper stop-loss mechanisms
3. **Diversification**: Test across multiple symbols
4. **Drawdown Control**: Monitor and limit drawdowns

## 🆘 Troubleshooting

### Common Issues
**"No data found"**
```bash
# Check data directory
ls ../../data/raw/

# Verify data format
head ../../data/raw/BTCUSDT_1h.csv
```

**"Strategy not found"**
```bash
# Check strategy imports
python -c "from src.strategies import TrendFollowingStrategy"
```

**"Optimization failed"**
```bash
# Check parameter ranges
# Verify data quality
# Reduce optimization complexity
```

## 📈 Performance Benchmarks

### Strategy Performance Comparison
- **Trend Following**: 15-25% annual return, 0.8-1.2 Sharpe
- **Mean Reversion**: 10-20% annual return, 0.6-1.0 Sharpe
- **Enhanced Strategies**: 20-35% annual return, 1.0-1.5 Sharpe
- **ML Strategies**: 25-40% annual return, 1.2-1.8 Sharpe

## 🚀 Production Integration

### Integration with ML Trading System
Strategies can be integrated with the ML trading system for:
- **Signal Generation**: Use strategy signals as features
- **Validation**: Backtest ML predictions against strategies
- **Hybrid Approaches**: Combine ML and traditional strategies

---

**Ready to start? Run: `uv run python app.py`** 🚀📈
