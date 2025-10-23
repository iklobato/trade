"""
Main entry point for ML trading system with 24/7 operation support.
"""

import argparse
import sys
import signal
from pathlib import Path
from typing import Dict, Any
import yaml
import json
from datetime import datetime

from orchestration.scheduler import TradingOrchestrator
from utils.seed import set_seed


# Configure enhanced logging for 24/7 operation
def setup_logging():
    """Setup comprehensive logging for 24/7 operation."""
    log_dir = Path("./app/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure loguru for better 24/7 logging
    from loguru import logger

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add file handler with rotation
    logger.add(
        log_dir / "trading_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )

    # Add error file handler
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="1 day",
        retention="30 days",
    )

    return logger


logger = setup_logging()

# Global orchestrator for signal handling
orchestrator_instance = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global orchestrator_instance
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

    if orchestrator_instance:
        orchestrator_instance.stop()

    logger.info("System shutdown completed")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for the trading system with 24/7 operation support."""
    global orchestrator_instance

    parser = argparse.ArgumentParser(
        description="ML-Driven Automated Crypto Trading System - 24/7 Operation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./app/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper", "backtest"],
        default="paper",
        help="Trading mode (default: paper for safety)",
    )
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Trading symbol")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without executing trades"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test exchange connection and exit",
    )
    parser.add_argument(
        "--download-data", action="store_true", help="Download historical data and exit"
    )
    parser.add_argument(
        "--train-model", action="store_true", help="Train model and exit"
    )
    parser.add_argument(
        "--backtest-only", action="store_true", help="Run backtest only and exit"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show system status and exit"
    )
    parser.add_argument(
        "--restart", action="store_true", help="Restart the trading system"
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Run as daemon (background process)"
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default="./app/trading.pid",
        help="PID file for daemon mode",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override config with command line arguments
        if args.mode != "paper":
            config["execution"]["trade_mode"] = args.mode
        if args.symbol != "BTC/USD":
            config["symbol"] = args.symbol

        # Set random seed
        set_seed(config.get("seed", 42))

        logger.info("=" * 80)
        logger.info("ML-Driven Automated Crypto Trading System - 24/7 Operation")
        logger.info("=" * 80)
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Trading Mode: {config['execution']['trade_mode']}")
        logger.info(f"Symbol: {config['symbol']}")
        logger.info(f"Exchange: {config['execution']['exchange']}")
        logger.info(f"Start Time: {datetime.now().isoformat()}")
        logger.info("=" * 80)

        # Initialize orchestrator
        orchestrator_instance = TradingOrchestrator(args.config)

        # Handle special modes
        if args.status:
            show_status(orchestrator_instance)
            return

        if args.test_connection:
            test_connection(orchestrator_instance)
            return

        if args.download_data:
            download_data(orchestrator_instance)
            return

        if args.train_model:
            train_model(orchestrator_instance)
            return

        if args.backtest_only:
            run_backtest(orchestrator_instance)
            return

        if args.restart:
            restart_system(orchestrator_instance)
            return

        # Daemon mode setup
        if args.daemon:
            setup_daemon(args.pid_file)

        # Start the trading system
        if args.dry_run:
            logger.info("DRY RUN MODE - No trades will be executed")
            config["execution"]["trade_mode"] = "paper"

        logger.info("Starting 24/7 trading system...")
        logger.info("Press Ctrl+C to stop gracefully")

        # Start the orchestrator
        success = orchestrator_instance.start()

        if not success:
            logger.error("Failed to start trading system")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def test_connection(orchestrator: TradingOrchestrator):
    """Test exchange connection."""
    logger.info("Testing exchange connection...")

    try:
        if orchestrator.gateway.test_connection():
            logger.info("✓ Exchange connection successful")

            # Get balance
            balance = orchestrator.gateway.get_balance()
            logger.info(f"Account balance: {balance}")

            # Get ticker
            ticker = orchestrator.gateway.get_ticker(orchestrator.config["symbol"])
            logger.info(f"Current price: {ticker.get('last', 'N/A')}")

        else:
            logger.error("✗ Exchange connection failed")

    except Exception as e:
        logger.error(f"Connection test error: {e}")


def download_data(orchestrator: TradingOrchestrator):
    """Download historical data."""
    logger.info("Downloading historical data...")

    try:
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days

        stats = orchestrator.downloader.download_date_range(start_date, end_date)

        logger.info("Download completed:")
        logger.info(f"  - Total files: {stats['total_files']}")
        logger.info(f"  - Downloaded: {stats['downloaded']}")
        logger.info(f"  - Skipped: {stats['skipped']}")
        logger.info(f"  - Failed: {stats['failed']}")

    except Exception as e:
        logger.error(f"Data download error: {e}")


def train_model(orchestrator: TradingOrchestrator):
    """Train the ML model."""
    logger.info("Training ML model...")

    try:
        orchestrator._daily_retrain_job()
        logger.info("Model training completed")

    except Exception as e:
        logger.error(f"Model training error: {e}")


def run_backtest(orchestrator: TradingOrchestrator):
    """Run backtest only."""
    logger.info("Running backtest...")

    try:
        from datetime import datetime, timedelta

        # Load recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        data = orchestrator.loader.load_date_range(start_date, end_date)
        symbol_data = orchestrator.loader.get_symbol_data(
            data, orchestrator.config["symbol"]
        )

        if len(symbol_data) == 0:
            logger.error("No data available for backtest")
            return

        # Generate features and labels
        features_data = orchestrator.feature_engineer.calculate_all_features(
            symbol_data
        )
        labeled_data = orchestrator.label_generator.generate_labels(features_data)

        # Train model
        feature_cols = orchestrator.feature_engineer.get_feature_names()
        X = labeled_data[feature_cols]
        y = labeled_data["label"]

        orchestrator.model.fit(X, y)

        # Run backtest
        predictions = orchestrator.model.predict(X)
        probabilities = orchestrator.model.predict_proba(X)

        backtest_results = orchestrator.backtester.run_backtest(
            labeled_data, predictions, probabilities
        )

        # Save results
        orchestrator.backtester.save_results()

        # Print results
        logger.info("Backtest Results:")
        logger.info(f"  - Total Return: {backtest_results['total_return']:.2%}")
        logger.info(f"  - Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        logger.info(f"  - Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        logger.info(f"  - Win Rate: {backtest_results['win_rate']:.2%}")
        logger.info(f"  - Number of Trades: {backtest_results['num_trades']}")

    except Exception as e:
        logger.error(f"Backtest error: {e}")


def show_status(orchestrator: TradingOrchestrator):
    """Show comprehensive system status."""
    logger.info("System Status Check")
    logger.info("=" * 50)

    try:
        status = orchestrator.get_system_status()

        logger.info(f"System Status: {status['status']}")
        logger.info(f"Model Trained: {status['model_trained']}")
        logger.info(f"Open Positions: {status['open_positions']}")
        logger.info(f"Total Trades: {status['total_trades']}")
        logger.info(f"Consecutive Errors: {status['consecutive_errors']}")
        logger.info(f"Uptime Hours: {status['uptime_hours']:.2f}")

        if status["last_prediction_time"]:
            logger.info(f"Last Prediction: {status['last_prediction_time']}")

        if status["last_retrain_time"]:
            logger.info(f"Last Retrain: {status['last_retrain_time']}")

        if status["performance_metrics"]:
            metrics = status["performance_metrics"]
            logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

        # Check health status
        health_file = Path("./app/logs/health_status.json")
        if health_file.exists():
            with open(health_file, "r") as f:
                health = json.load(f)

            logger.info(f"System Healthy: {health.get('is_healthy', 'Unknown')}")
            if health.get("issues"):
                logger.warning(f"Issues: {', '.join(health['issues'])}")

    except Exception as e:
        logger.error(f"Status check error: {e}")


def restart_system(orchestrator: TradingOrchestrator):
    """Restart the trading system."""
    logger.info("Restarting trading system...")

    try:
        success = orchestrator.restart()
        if success:
            logger.info("System restarted successfully")
        else:
            logger.error("System restart failed")
    except Exception as e:
        logger.error(f"Restart error: {e}")


def setup_daemon(pid_file: str):
    """Setup daemon mode for background operation."""
    import os

    logger.info(f"Setting up daemon mode with PID file: {pid_file}")

    try:
        # Create PID file
        pid_path = Path(pid_file)
        pid_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))

        logger.info(f"Daemon mode enabled - PID: {os.getpid()}")
        logger.info(f"PID file: {pid_file}")
        logger.info("System will run in background")

    except Exception as e:
        logger.error(f"Daemon setup error: {e}")


def create_systemd_service():
    """Create systemd service file for production deployment."""
    service_content = """[Unit]
Description=ML Trading System
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/path/to/crypto
ExecStart=/usr/bin/uv run python app/main.py --mode paper
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

    service_file = Path("./ml-trading.service")
    with open(service_file, "w") as f:
        f.write(service_content)

    logger.info(f"Systemd service file created: {service_file}")
    logger.info("To install:")
    logger.info("  sudo cp ml-trading.service /etc/systemd/system/")
    logger.info("  sudo systemctl enable ml-trading")
    logger.info("  sudo systemctl start ml-trading")


def create_monitoring_script():
    """Create monitoring script for system health."""
    script_content = """#!/bin/bash
# ML Trading System Monitoring Script

LOG_DIR="./app/logs"
STATUS_FILE="$LOG_DIR/health_status.json"

echo "=== ML Trading System Monitor ==="
echo "Timestamp: $(date)"
echo

# Check if system is running
if pgrep -f "python app/main.py" > /dev/null; then
    echo "✓ System is running"
else
    echo "✗ System is not running"
fi

# Check health status
if [ -f "$STATUS_FILE" ]; then
    echo "Health Status:"
    cat "$STATUS_FILE" | jq '.is_healthy, .issues' 2>/dev/null || echo "Health file exists but cannot parse"
else
    echo "✗ No health status file found"
fi

# Check recent logs
echo
echo "Recent Errors:"
tail -5 "$LOG_DIR/errors_$(date +%Y-%m-%d).log" 2>/dev/null || echo "No error logs found"

echo
echo "Recent Activity:"
tail -5 "$LOG_DIR/trading_$(date +%Y-%m-%d).log" 2>/dev/null || echo "No activity logs found"
"""

    script_file = Path("./monitor.sh")
    with open(script_file, "w") as f:
        f.write(script_content)

    # Make executable
    script_file.chmod(0o755)

    logger.info(f"Monitoring script created: {script_file}")
    logger.info("Run with: ./monitor.sh")


if __name__ == "__main__":
    main()
