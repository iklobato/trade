"""
Main entry point for ML trading system.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import yaml

from orchestration.scheduler import TradingOrchestrator
from utils.seed import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the trading system."""
    parser = argparse.ArgumentParser(description='ML-Driven Automated Crypto Trading System')
    parser.add_argument('--config', type=str, default='./app/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['live', 'paper', 'backtest'],
                       default='live', help='Trading mode')
    parser.add_argument('--symbol', type=str, default='BTC/USD',
                       help='Trading symbol')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without executing trades')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test exchange connection and exit')
    parser.add_argument('--download-data', action='store_true',
                       help='Download historical data and exit')
    parser.add_argument('--train-model', action='store_true',
                       help='Train model and exit')
    parser.add_argument('--backtest-only', action='store_true',
                       help='Run backtest only and exit')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.mode != 'live':
            config['execution']['trade_mode'] = args.mode
        if args.symbol != 'BTC/USD':
            config['symbol'] = args.symbol
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        logger.info("=" * 60)
        logger.info("ML-Driven Automated Crypto Trading System")
        logger.info("=" * 60)
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Trading Mode: {config['execution']['trade_mode']}")
        logger.info(f"Symbol: {config['symbol']}")
        logger.info(f"Exchange: {config['execution']['exchange']}")
        logger.info("=" * 60)
        
        # Initialize orchestrator
        orchestrator = TradingOrchestrator(args.config)
        
        # Handle special modes
        if args.test_connection:
            test_connection(orchestrator)
            return
        
        if args.download_data:
            download_data(orchestrator)
            return
        
        if args.train_model:
            train_model(orchestrator)
            return
        
        if args.backtest_only:
            run_backtest(orchestrator)
            return
        
        # Start the trading system
        if args.dry_run:
            logger.info("DRY RUN MODE - No trades will be executed")
            config['execution']['trade_mode'] = 'paper'
        
        orchestrator.start()
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
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
            ticker = orchestrator.gateway.get_ticker(orchestrator.config['symbol'])
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
        
        logger.info(f"Download completed:")
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
        symbol_data = orchestrator.loader.get_symbol_data(data, orchestrator.config['symbol'])
        
        if len(symbol_data) == 0:
            logger.error("No data available for backtest")
            return
        
        # Generate features and labels
        features_data = orchestrator.feature_engineer.calculate_all_features(symbol_data)
        labeled_data = orchestrator.label_generator.generate_labels(features_data)
        
        # Train model
        feature_cols = orchestrator.feature_engineer.get_feature_names()
        X = labeled_data[feature_cols]
        y = labeled_data['label']
        
        metrics = orchestrator.model.fit(X, y)
        
        # Run backtest
        predictions = orchestrator.model.predict(X)
        probabilities = orchestrator.model.predict_proba(X)
        
        backtest_results = orchestrator.backtester.run_backtest(
            labeled_data,
            predictions,
            probabilities
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


if __name__ == "__main__":
    main()
