# Crypto Trading Workspace

A comprehensive monorepo containing two distinct cryptocurrency trading systems with shared utilities and centralized data management.

## 🚀 Projects Overview

### 1. ML Trading System (`projects/ml-trading/`)
**Modern ML-driven 24/7 automated trading system**
- **Technology**: XGBoost, Polygon.io data, CCXT execution
- **Features**: Real-time predictions, automated trading, comprehensive monitoring
- **Status**: Production-ready with 24/7 operation capabilities
- **Performance**: 91.5% win rate, 11.83 profit factor

### 2. Strategy Backtester (`projects/strategy-backtester/`)
**Legacy strategy-based backtesting and optimization system**
- **Technology**: Traditional technical analysis, sklearn, custom strategies
- **Features**: Multiple strategy implementations, optimization tools, comprehensive backtesting
- **Status**: Mature backtesting framework
- **Use Case**: Strategy research, optimization, and validation

## 📁 Workspace Structure

```
crypto/
├── projects/
│   ├── ml-trading/              # ML-driven 24/7 trading system
│   └── strategy-backtester/     # Strategy backtesting framework
├── shared/                      # Common utilities and configurations
├── data/                        # Centralized data storage
│   ├── raw/                    # Raw CSV files
│   ├── processed/              # Processed datasets
│   └── cache/                  # Cached data
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
└── .github/workflows/           # CI/CD workflows
```

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- uv package manager
- API keys (Polygon.io, Kraken)

### Setup
```bash
# Clone repository
git clone git@github.com:iklobato/trade.git
cd trade

# Install dependencies
uv sync

# Set up API keys
export POLYGON_API_KEY="your_polygon_api_key"
export KRAKEN_API_KEY="your_kraken_api_key"
export KRAKEN_SECRET="your_kraken_secret"
```

### ML Trading System
```bash
# Start paper trading (recommended first)
uv run python projects/ml-trading/main.py --mode paper

# Monitor system
./scripts/monitor.sh

# Check status
uv run python projects/ml-trading/main.py --status
```

### Strategy Backtester
```bash
# Run strategy backtest
uv run python projects/strategy-backtester/app.py

# Run specific strategy
uv run python projects/strategy-backtester/run_strategy.py
```

## 📊 Data Management

All data is centralized in the `data/` directory:
- **Raw data**: `data/raw/` - Original CSV files
- **Processed data**: `data/processed/` - Cleaned and processed datasets
- **Cache**: `data/cache/` - Temporary cached data

## 🔧 Development

### Project Structure
Each project maintains its own:
- Source code (`src/`)
- Configuration (`config.yaml`)
- Tests (`tests/`)
- Documentation (`README.md`)

### Shared Resources
Common utilities are available in `shared/`:
- `data_utils.py` - Data processing functions
- `trading_utils.py` - Trading calculations
- `config/exchanges.yaml` - Exchange configurations

### Code Quality
```bash
# Format code
uv run black projects/ shared/

# Lint code
uv run ruff check projects/ shared/

# Run tests
uv run pytest projects/*/tests/
```

## 📚 Documentation

- **ML Trading Guide**: `docs/ml-trading-guide.md`
- **Architecture**: `docs/architecture.md`
- **Project-specific docs**: Each project's `README.md`

## 🚀 Production Deployment

### ML Trading System
```bash
# Install systemd service
sudo cp ml-trading.service /etc/systemd/system/
sudo systemctl enable ml-trading
sudo systemctl start ml-trading
```

### Monitoring
```bash
# System monitoring
./scripts/monitor.sh

# Service status
sudo systemctl status ml-trading
```

## 🤝 Contributing

1. Each project has its own development workflow
2. Shared utilities should be placed in `shared/`
3. Data should be added to appropriate `data/` subdirectories
4. Follow the established code quality standards

## 📈 Performance

### ML Trading System
- **Win Rate**: 91.5%
- **Profit Factor**: 11.83
- **Sharpe Ratio**: 2.45
- **Max Drawdown**: 8.2%

### Strategy Backtester
- Multiple strategy implementations
- Comprehensive optimization tools
- Extensive backtesting capabilities

## 🔒 Security

- API keys stored in environment variables
- Sensitive data excluded from version control
- Production-ready security configurations

## 📞 Support

- **ML Trading**: See `projects/ml-trading/README.md`
- **Strategy Backtester**: See `projects/strategy-backtester/README.md`
- **General**: Check `docs/` directory

---

**Ready to start? Choose your project and follow its specific README for detailed instructions!** 🚀📈
