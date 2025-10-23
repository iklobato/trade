#!/bin/bash
# Setup script for Crypto Trading Workspace

set -e

echo "🚀 Setting up Crypto Trading Workspace"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the workspace root directory"
    exit 1
fi

# Check for required tools
echo "🔍 Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "❌ uv is required but not installed"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync --extra dev

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/processed
mkdir -p data/cache
mkdir -p projects/ml-trading/src/logs
mkdir -p projects/ml-trading/src/reports
mkdir -p projects/ml-trading/src/artifacts

# Set up environment file template
echo "🔧 Creating environment template..."
cat > .env.template << EOF
# API Keys - Copy to .env and fill in your keys
POLYGON_API_KEY=your_polygon_api_key_here
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_SECRET=your_kraken_secret_here

# Optional: Additional exchange keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET=your_coinbase_secret_here
EOF

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/monitor.sh

# Run initial tests
echo "🧪 Running initial tests..."
uv run pytest projects/ml-trading/src/tests/ -v || echo "⚠️  ML Trading tests failed (expected if no data)"
uv run pytest projects/strategy-backtester/tests/ -v || echo "⚠️  Strategy Backtester tests failed (expected if no data)"

# Check code quality
echo "🔍 Running code quality checks..."
uv run black --check projects/ shared/ || echo "⚠️  Code formatting issues found"
uv run ruff check projects/ shared/ || echo "⚠️  Linting issues found"

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.template to .env and add your API keys"
echo "2. Download data: uv run python projects/ml-trading/main.py --download-data"
echo "3. Train model: uv run python projects/ml-trading/main.py --train-model"
echo "4. Start paper trading: uv run python projects/ml-trading/main.py --mode paper"
echo ""
echo "📚 Documentation:"
echo "- Root guide: README.md"
echo "- ML Trading: projects/ml-trading/README.md"
echo "- Strategy Backtester: projects/strategy-backtester/README.md"
echo "- Architecture: docs/architecture.md"
echo ""
echo "🔧 Monitoring:"
echo "- System monitor: ./scripts/monitor.sh"
echo "- ML Trading status: uv run python projects/ml-trading/main.py --status"
echo ""
echo "🎯 Ready to start trading! 🚀📈"
