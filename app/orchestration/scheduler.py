"""
APScheduler orchestration for automated trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path
import json

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from app.data.polygon_flatfiles_downloader import PolygonFlatFilesDownloader
from app.data.polygon_loader import PolygonLoader
from app.features.engineer import FeatureEngineer
from app.labels.targets import LabelGenerator
from app.models.classifier import XGBoostClassifier
from app.backtest.simulator import BacktestSimulator
from app.execution.gateway import CCXTGateway
from app.execution.trade_manager import TradeManager
from app.utils.seed import set_seed

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Orchestrate the entire trading system with scheduled tasks.
    """
    
    def __init__(self, config_path: str = "./app/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set random seed
        set_seed(self.config.get('seed', 42))
        
        # Initialize components
        self._initialize_components()
        
        # Initialize scheduler
        self.scheduler = BlockingScheduler()
        self._setup_jobs()
        
        # State tracking
        self.last_prediction_time = None
        self.last_retrain_time = None
        self.system_status = "initialized"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing trading system components...")
        
        # Data components
        self.downloader = PolygonFlatFilesDownloader(
            api_key=self.config['polygon_api_key'],
            data_root=self.config['data_root']
        )
        
        self.loader = PolygonLoader(
            data_root=self.config['data_root']
        )
        
        # ML components
        self.feature_engineer = FeatureEngineer()
        self.label_generator = LabelGenerator(
            horizon_minutes=self.config['horizon_minutes'],
            up_threshold=self.config['thresholds']['up'],
            down_threshold=self.config['thresholds']['down']
        )
        
        self.model = XGBoostClassifier(
            model_params=self.config['model']['params'],
            artifacts_dir="./app/artifacts"
        )
        
        # Backtesting
        self.backtester = BacktestSimulator(
            initial_capital=10000.0,
            fee_bps=self.config['costs']['fee_bps'],
            slippage_bps=self.config['costs']['slippage_bps'],
            risk_per_trade=self.config['risk']['risk_per_trade'],
            max_position=self.config['risk']['max_pos'],
            horizon_minutes=self.config['horizon_minutes']
        )
        
        # Execution components
        self.gateway = CCXTGateway(
            exchange=self.config['execution']['exchange'],
            trade_mode=self.config['execution']['trade_mode'],
            sandbox=False
        )
        
        self.trade_manager = TradeManager(
            gateway=self.gateway,
            initial_capital=10000.0,
            risk_per_trade=self.config['risk']['risk_per_trade'],
            max_position=self.config['risk']['max_pos']
        )
        
        logger.info("All components initialized successfully")
    
    def _setup_jobs(self):
        """Setup scheduled jobs."""
        logger.info("Setting up scheduled jobs...")
        
        # Hourly prediction job
        self.scheduler.add_job(
            func=self._hourly_prediction_job,
            trigger=IntervalTrigger(minutes=self.config['scheduler']['interval_minutes']),
            id='hourly_prediction',
            name='Hourly Prediction and Trading',
            max_instances=1,
            coalesce=True
        )
        
        # Daily retraining job
        if self.config['scheduler']['retrain_daily']:
            self.scheduler.add_job(
                func=self._daily_retrain_job,
                trigger=CronTrigger(hour=0, minute=0),  # Midnight UTC
                id='daily_retrain',
                name='Daily Model Retraining',
                max_instances=1,
                coalesce=True
            )
        
        # Daily data download job
        self.scheduler.add_job(
            func=self._daily_data_download_job,
            trigger=CronTrigger(hour=1, minute=0),  # 1 AM UTC
            id='daily_data_download',
            name='Daily Data Download',
            max_instances=1,
            coalesce=True
        )
        
        logger.info("Scheduled jobs configured")
    
    def _hourly_prediction_job(self):
        """Hourly prediction and trading job."""
        try:
            logger.info("Starting hourly prediction job...")
            
            # Get latest data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Last 7 days
            
            data = self.loader.load_date_range(start_date, end_date)
            if len(data) == 0:
                logger.warning("No data available for prediction")
                return
            
            # Get symbol data
            symbol_data = self.loader.get_symbol_data(data, self.config['symbol'])
            if len(symbol_data) == 0:
                logger.warning(f"No data available for {self.config['symbol']}")
                return
            
            # Generate features
            features_data = self.feature_engineer.calculate_all_features(symbol_data)
            
            # Get latest features for prediction
            latest_features = features_data.iloc[-1:][self.feature_engineer.get_feature_names()]
            
            # Make prediction
            if not self.model.is_trained:
                logger.warning("Model not trained, skipping prediction")
                return
            
            prediction = self.model.predict(latest_features)
            probabilities = self.model.predict_proba(latest_features)
            
            # Get current price
            current_price = symbol_data.iloc[-1]['close']
            
            # Execute trades based on prediction
            self._execute_trades(prediction[0], probabilities[0], current_price)
            
            # Update positions
            current_prices = {self.config['symbol']: current_price}
            self.trade_manager.update_positions(current_prices)
            
            # Check stop loss and take profit
            closed_trades = self.trade_manager.check_stop_loss_take_profit(current_prices)
            
            # Log status
            self._log_trading_status(prediction[0], probabilities[0], current_price)
            
            self.last_prediction_time = datetime.now()
            logger.info("Hourly prediction job completed")
            
        except Exception as e:
            logger.error(f"Error in hourly prediction job: {e}")
    
    def _daily_retrain_job(self):
        """Daily model retraining job."""
        try:
            logger.info("Starting daily retraining job...")
            
            # Download latest data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            
            # Download data
            self.downloader.download_date_range(start_date, end_date)
            
            # Load data
            data = self.loader.load_date_range(start_date, end_date)
            if len(data) == 0:
                logger.warning("No data available for retraining")
                return
            
            # Get symbol data
            symbol_data = self.loader.get_symbol_data(data, self.config['symbol'])
            if len(symbol_data) == 0:
                logger.warning(f"No data available for {self.config['symbol']}")
                return
            
            # Generate features
            features_data = self.feature_engineer.calculate_all_features(symbol_data)
            
            # Generate labels
            labeled_data = self.label_generator.generate_labels(features_data)
            
            # Prepare training data
            feature_cols = self.feature_engineer.get_feature_names()
            X = labeled_data[feature_cols]
            y = labeled_data['label']
            
            # Train model
            metrics = self.model.fit(X, y)
            
            # Save model
            model_path = self.model.save()
            
            # Run backtest
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            backtest_results = self.backtester.run_backtest(
                labeled_data,
                predictions,
                probabilities
            )
            
            # Save backtest results
            self.backtester.save_results()
            
            # Log results
            logger.info(f"Model retrained - Accuracy: {metrics['val_accuracy']:.4f}")
            logger.info(f"Backtest results - Return: {backtest_results['total_return']:.2%}")
            
            self.last_retrain_time = datetime.now()
            logger.info("Daily retraining job completed")
            
        except Exception as e:
            logger.error(f"Error in daily retraining job: {e}")
    
    def _daily_data_download_job(self):
        """Daily data download job."""
        try:
            logger.info("Starting daily data download job...")
            
            # Download yesterday's data
            yesterday = datetime.now() - timedelta(days=1)
            
            stats = self.downloader.download_date_range(
                yesterday, yesterday,
                force_redownload=False
            )
            
            logger.info(f"Data download completed: {stats['downloaded']} files downloaded")
            
        except Exception as e:
            logger.error(f"Error in daily data download job: {e}")
    
    def _execute_trades(self, prediction: int, probabilities: np.ndarray, current_price: float):
        """Execute trades based on model prediction."""
        try:
            prob_up = probabilities[2]  # P(up)
            prob_down = probabilities[0]  # P(down)
            
            # Long signal
            if prediction == 1 and prob_up >= 0.6:  # High confidence threshold
                position_size = self._calculate_position_size(current_price)
                self.trade_manager.open_position(
                    symbol=self.config['symbol'],
                    side='long',
                    size=position_size,
                    price=current_price,
                    strategy="ML"
                )
                logger.info(f"Long signal executed: {position_size:.6f} at {current_price:.2f}")
            
            # Short signal
            elif prediction == -1 and prob_down >= 0.6:  # High confidence threshold
                position_size = self._calculate_position_size(current_price)
                self.trade_manager.open_position(
                    symbol=self.config['symbol'],
                    side='short',
                    size=position_size,
                    price=current_price,
                    strategy="ML"
                )
                logger.info(f"Short signal executed: {position_size:.6f} at {current_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk parameters."""
        portfolio_value = self.trade_manager.get_portfolio_value({self.config['symbol']: price})
        max_position_value = portfolio_value * self.config['risk']['max_pos']
        position_size = max_position_value / price
        
        return position_size
    
    def _log_trading_status(self, prediction: int, probabilities: np.ndarray, current_price: float):
        """Log current trading status."""
        metrics = self.trade_manager.get_performance_metrics()
        portfolio_value = self.trade_manager.get_portfolio_value({self.config['symbol']: current_price})
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'probabilities': probabilities.tolist(),
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'open_positions': len([p for p in self.trade_manager.positions if p.status == 'open']),
            'total_return': metrics['total_return'],
            'win_rate': metrics['win_rate'],
            'num_trades': metrics['num_trades']
        }
        
        logger.info(f"Trading Status: Prediction={prediction}, Price={current_price:.2f}, "
                   f"Portfolio=${portfolio_value:.2f}, Return={metrics['total_return']:.2%}")
        
        # Save status to file
        status_file = Path("./app/reports/trading_status.json")
        status_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def start(self):
        """Start the trading orchestrator."""
        logger.info("Starting trading orchestrator...")
        
        # Test connections
        if not self.gateway.test_connection():
            logger.error("Exchange connection test failed")
            return
        
        # Initial model training if not trained
        if not self.model.is_trained:
            logger.info("Model not trained, starting initial training...")
            self._daily_retrain_job()
        
        # Start scheduler
        self.system_status = "running"
        logger.info("Trading system started successfully")
        
        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            logger.info("Trading system stopped by user")
        except Exception as e:
            logger.error(f"Error in trading system: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading orchestrator."""
        logger.info("Stopping trading orchestrator...")
        
        # Close all positions
        current_prices = {self.config['symbol']: 50000.0}  # Default price
        for position in self.trade_manager.positions:
            if position.status == 'open':
                self.trade_manager.close_position(position, current_prices[self.config['symbol']], "system_shutdown")
        
        # Save final state
        self.trade_manager.save_state("./app/reports/final_state.json")
        
        self.system_status = "stopped"
        logger.info("Trading system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'status': self.system_status,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'model_trained': self.model.is_trained,
            'open_positions': len([p for p in self.trade_manager.positions if p.status == 'open']),
            'total_trades': len(self.trade_manager.trades),
            'performance_metrics': self.trade_manager.get_performance_metrics()
        }
