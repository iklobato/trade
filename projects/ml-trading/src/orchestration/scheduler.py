"""
APScheduler orchestration for automated trading system with 24/7 operation.
"""

import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import json
import signal
import sys
import time
import traceback
from threading import Event

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from loguru import logger

from src.data.polygon_flatfiles_downloader import PolygonFlatFilesDownloader
from src.data.polygon_loader import PolygonLoader
from src.features.engineer import FeatureEngineer
from src.labels.targets import LabelGenerator
from src.models.classifier import XGBoostClassifier
from src.backtest.simulator import BacktestSimulator
from src.execution.gateway import CCXTGateway
from src.execution.trade_manager import TradeManager
from src.utils.seed import set_seed


class TradingOrchestrator:
    """
    Orchestrate the entire trading system with scheduled tasks for 24/7 operation.
    """

    def __init__(self, config_path: str = "./app/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

        # Set random seed
        set_seed(self.config.get("seed", 42))

        # Initialize components
        self._initialize_components()

        # Initialize scheduler with robust configuration
        self.scheduler = BlockingScheduler(
            timezone="UTC",
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 300,  # 5 minutes grace period
            },
        )

        # Add event listeners for monitoring
        self.scheduler.add_listener(self._job_executed_listener, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)

        self._setup_jobs()

        # State tracking
        self.last_prediction_time = None
        self.last_retrain_time = None
        self.last_health_check = None
        self.system_status = "initialized"
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

        # Graceful shutdown handling
        self.shutdown_event = Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Health monitoring
        self.health_check_interval = 300  # 5 minutes
        self.last_successful_prediction = None
        self.last_successful_retrain = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing trading system components...")

        # Data components
        self.downloader = PolygonFlatFilesDownloader(
            api_key=self.config["polygon_api_key"], data_root=self.config["data_root"]
        )

        self.loader = PolygonLoader(data_root=self.config["data_root"])

        # ML components
        self.feature_engineer = FeatureEngineer()
        self.label_generator = LabelGenerator(
            horizon_minutes=self.config["horizon_minutes"],
            up_threshold=self.config["thresholds"]["up"],
            down_threshold=self.config["thresholds"]["down"],
        )

        self.model = XGBoostClassifier(
            model_params=self.config["model"]["params"], artifacts_dir="./app/artifacts"
        )

        # Backtesting
        self.backtester = BacktestSimulator(
            initial_capital=10000.0,
            fee_bps=self.config["costs"]["fee_bps"],
            slippage_bps=self.config["costs"]["slippage_bps"],
            risk_per_trade=self.config["risk"]["risk_per_trade"],
            max_position=self.config["risk"]["max_pos"],
            horizon_minutes=self.config["horizon_minutes"],
        )

        # Execution components
        self.gateway = CCXTGateway(
            exchange=self.config["execution"]["exchange"],
            trade_mode=self.config["execution"]["trade_mode"],
            sandbox=False,
        )

        self.trade_manager = TradeManager(
            gateway=self.gateway,
            initial_capital=10000.0,
            risk_per_trade=self.config["risk"]["risk_per_trade"],
            max_position=self.config["risk"]["max_pos"],
        )

        logger.info("All components initialized successfully")

    def _setup_jobs(self):
        """Setup scheduled jobs for 24/7 operation."""
        logger.info("Setting up scheduled jobs for 24/7 operation...")

        # Hourly prediction job (every 15 minutes for more responsive trading)
        self.scheduler.add_job(
            func=self._hourly_prediction_job,
            trigger=IntervalTrigger(
                minutes=15
            ),  # More frequent for better responsiveness
            id="prediction_job",
            name="Prediction and Trading",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

        # Health check job (every 5 minutes)
        self.scheduler.add_job(
            func=self._health_check_job,
            trigger=IntervalTrigger(minutes=5),
            id="health_check",
            name="System Health Check",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

        # Daily retraining job
        if self.config["scheduler"]["retrain_daily"]:
            self.scheduler.add_job(
                func=self._daily_retrain_job,
                trigger=CronTrigger(hour=0, minute=30),  # 12:30 AM UTC
                id="daily_retrain",
                name="Daily Model Retraining",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )

        # Daily data download job
        self.scheduler.add_job(
            func=self._daily_data_download_job,
            trigger=CronTrigger(hour=1, minute=0),  # 1 AM UTC
            id="daily_data_download",
            name="Daily Data Download",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

        # Weekly system maintenance (Sundays at 2 AM)
        self.scheduler.add_job(
            func=self._weekly_maintenance_job,
            trigger=CronTrigger(day_of_week=6, hour=2, minute=0),  # Sunday 2 AM UTC
            id="weekly_maintenance",
            name="Weekly System Maintenance",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

        # Emergency recovery job (every hour)
        self.scheduler.add_job(
            func=self._emergency_recovery_job,
            trigger=IntervalTrigger(hours=1),
            id="emergency_recovery",
            name="Emergency Recovery Check",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

        logger.info("Scheduled jobs configured for 24/7 operation")

    def _job_executed_listener(self, event):
        """Listener for successful job execution."""
        logger.info(f"Job {event.job_id} executed successfully")
        self.consecutive_errors = 0  # Reset error counter on success

    def _job_error_listener(self, event):
        """Listener for job execution errors."""
        logger.error(f"Job {event.job_id} failed: {event.exception}")
        self.consecutive_errors += 1

        # Log detailed error information
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "job_id": event.job_id,
            "error": str(event.exception),
            "traceback": traceback.format_exc(),
        }

        # Save error to file
        error_file = Path("./src/logs/errors.json")
        error_file.parent.mkdir(parents=True, exist_ok=True)

        with open(error_file, "a") as f:
            f.write(json.dumps(error_info) + "\n")

        # Trigger emergency recovery if too many consecutive errors
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.critical(
                f"Too many consecutive errors ({self.consecutive_errors}), triggering emergency recovery"
            )
            self._emergency_recovery_job()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.stop()
        sys.exit(0)

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
            symbol_data = self.loader.get_symbol_data(data, self.config["symbol"])
            if len(symbol_data) == 0:
                logger.warning(f"No data available for {self.config['symbol']}")
                return

            # Generate features
            features_data = self.feature_engineer.calculate_all_features(symbol_data)

            # Get latest features for prediction
            latest_features = features_data.iloc[-1:][
                self.feature_engineer.get_feature_names()
            ]

            # Make prediction
            if not self.model.is_trained:
                logger.warning("Model not trained, skipping prediction")
                return

            prediction = self.model.predict(latest_features)
            probabilities = self.model.predict_proba(latest_features)

            # Get current price
            current_price = symbol_data.iloc[-1]["close"]

            # Execute trades based on prediction
            self._execute_trades(prediction[0], probabilities[0], current_price)

            # Update positions
            current_prices = {self.config["symbol"]: current_price}
            self.trade_manager.update_positions(current_prices)

            # Check stop loss and take profit
            self.trade_manager.check_stop_loss_take_profit(current_prices)

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
            symbol_data = self.loader.get_symbol_data(data, self.config["symbol"])
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
            y = labeled_data["label"]

            # Train model
            metrics = self.model.fit(X, y)

            # Save model
            self.model.save()

            # Run backtest
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            backtest_results = self.backtester.run_backtest(
                labeled_data, predictions, probabilities
            )

            # Save backtest results
            self.backtester.save_results()

            # Log results
            logger.info(f"Model retrained - Accuracy: {metrics['val_accuracy']:.4f}")
            logger.info(
                f"Backtest results - Return: {backtest_results['total_return']:.2%}"
            )

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
                yesterday, yesterday, force_redownload=False
            )

            logger.info(
                f"Data download completed: {stats['downloaded']} files downloaded"
            )

        except Exception as e:
            logger.error(f"Error in daily data download job: {e}")

    def _health_check_job(self):
        """System health check job."""
        try:
            logger.debug("Performing system health check...")

            health_status = {
                "timestamp": datetime.now().isoformat(),
                "system_status": self.system_status,
                "last_prediction_time": (
                    self.last_prediction_time.isoformat()
                    if self.last_prediction_time
                    else None
                ),
                "last_retrain_time": (
                    self.last_retrain_time.isoformat()
                    if self.last_retrain_time
                    else None
                ),
                "consecutive_errors": self.consecutive_errors,
                "model_trained": self.model.is_trained,
                "open_positions": len(
                    [p for p in self.trade_manager.positions if p.status == "open"]
                ),
                "total_trades": len(self.trade_manager.trades),
            }

            # Check if system is healthy
            is_healthy = True
            issues = []

            # Check if model is trained
            if not self.model.is_trained:
                is_healthy = False
                issues.append("Model not trained")

            # Check if too many consecutive errors
            if self.consecutive_errors >= self.max_consecutive_errors:
                is_healthy = False
                issues.append(f"Too many consecutive errors: {self.consecutive_errors}")

            # Check if prediction job is running
            if self.last_prediction_time:
                time_since_prediction = datetime.now() - self.last_prediction_time
                if time_since_prediction.total_seconds() > 3600:  # 1 hour
                    is_healthy = False
                    issues.append(f"No prediction for {time_since_prediction}")

            # Check exchange connection
            try:
                if not self.gateway.test_connection():
                    is_healthy = False
                    issues.append("Exchange connection failed")
            except Exception as e:
                is_healthy = False
                issues.append(f"Exchange connection error: {e}")

            health_status["is_healthy"] = is_healthy
            health_status["issues"] = issues

            # Save health status
            health_file = Path("./src/logs/health_status.json")
            health_file.parent.mkdir(parents=True, exist_ok=True)

            with open(health_file, "w") as f:
                json.dump(health_status, f, indent=2)

            if not is_healthy:
                logger.warning(f"System health check failed: {issues}")
            else:
                logger.debug("System health check passed")

            self.last_health_check = datetime.now()

        except Exception as e:
            logger.error(f"Error in health check job: {e}")

    def _weekly_maintenance_job(self):
        """Weekly system maintenance job."""
        try:
            logger.info("Starting weekly system maintenance...")

            # Clean up old log files (keep last 30 days)
            log_dir = Path("./src/logs")
            if log_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=30)
                for log_file in log_dir.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        logger.info(f"Cleaned up old log file: {log_file}")

            # Clean up old artifacts (keep last 7 models)
            artifacts_dir = Path("./app/artifacts")
            if artifacts_dir.exists():
                model_dirs = [
                    d
                    for d in artifacts_dir.iterdir()
                    if d.is_dir() and d.name.startswith("model_")
                ]
                if len(model_dirs) > 7:
                    model_dirs.sort(key=lambda x: x.stat().st_mtime)
                    for old_model in model_dirs[:-7]:
                        import shutil

                        shutil.rmtree(old_model)
                        logger.info(f"Cleaned up old model: {old_model}")

            # Generate weekly performance report
            self._generate_weekly_report()

            logger.info("Weekly system maintenance completed")

        except Exception as e:
            logger.error(f"Error in weekly maintenance job: {e}")

    def _emergency_recovery_job(self):
        """Emergency recovery job."""
        try:
            logger.warning("Starting emergency recovery procedures...")

            # Reset error counter
            self.consecutive_errors = 0

            # Test all critical components
            components_ok = True

            # Test exchange connection
            try:
                if not self.gateway.test_connection():
                    logger.error("Exchange connection failed during recovery")
                    components_ok = False
            except Exception as e:
                logger.error(f"Exchange connection error during recovery: {e}")
                components_ok = False

            # Test data loading
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)
                data = self.loader.load_date_range(start_date, end_date)
                if len(data) == 0:
                    logger.error("No data available during recovery")
                    components_ok = False
            except Exception as e:
                logger.error(f"Data loading error during recovery: {e}")
                components_ok = False

            # Test model
            try:
                if not self.model.is_trained:
                    logger.warning(
                        "Model not trained during recovery, attempting retrain..."
                    )
                    self._daily_retrain_job()
            except Exception as e:
                logger.error(f"Model error during recovery: {e}")
                components_ok = False

            if components_ok:
                logger.info("Emergency recovery completed successfully")
                self.system_status = "recovered"
            else:
                logger.critical(
                    "Emergency recovery failed - manual intervention required"
                )
                self.system_status = "critical"

        except Exception as e:
            logger.critical(f"Error in emergency recovery job: {e}")

    def _generate_weekly_report(self):
        """Generate weekly performance report."""
        try:
            logger.info("Generating weekly performance report...")

            # Get performance metrics
            metrics = self.trade_manager.get_performance_metrics()

            # Get system status
            status = self.get_system_status()

            # Create weekly report
            report = {
                "report_type": "weekly_performance",
                "generated_at": datetime.now().isoformat(),
                "period": "7_days",
                "performance_metrics": metrics,
                "system_status": status,
                "health_checks": self._get_health_summary(),
                "recommendations": self._generate_recommendations(metrics),
            }

            # Save report
            report_file = Path("./app/reports/weekly_report.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.info("Weekly performance report generated")

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")

    def _get_health_summary(self):
        """Get health check summary."""
        try:
            health_file = Path("./src/logs/health_status.json")
            if health_file.exists():
                with open(health_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def _generate_recommendations(self, metrics):
        """Generate system recommendations based on performance."""
        recommendations = []

        if metrics.get("win_rate", 0) < 0.6:
            recommendations.append("Consider retraining model - low win rate")

        if metrics.get("total_return", 0) < 0:
            recommendations.append("Review trading strategy - negative returns")

        if self.consecutive_errors > 2:
            recommendations.append("Investigate system errors - high error rate")

        if not self.model.is_trained:
            recommendations.append("Train model immediately")

        return recommendations

    def _execute_trades(
        self, prediction: int, probabilities: np.ndarray, current_price: float
    ):
        """Execute trades based on model prediction."""
        try:
            prob_up = probabilities[2]  # P(up)
            prob_down = probabilities[0]  # P(down)

            # Long signal
            if prediction == 1 and prob_up >= 0.6:  # High confidence threshold
                position_size = self._calculate_position_size(current_price)
                self.trade_manager.open_position(
                    symbol=self.config["symbol"],
                    side="long",
                    size=position_size,
                    price=current_price,
                    strategy="ML",
                )
                logger.info(
                    f"Long signal executed: {position_size:.6f} at {current_price:.2f}"
                )

            # Short signal
            elif prediction == -1 and prob_down >= 0.6:  # High confidence threshold
                position_size = self._calculate_position_size(current_price)
                self.trade_manager.open_position(
                    symbol=self.config["symbol"],
                    side="short",
                    size=position_size,
                    price=current_price,
                    strategy="ML",
                )
                logger.info(
                    f"Short signal executed: {position_size:.6f} at {current_price:.2f}"
                )

        except Exception as e:
            logger.error(f"Error executing trades: {e}")

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk parameters."""
        portfolio_value = self.trade_manager.get_portfolio_value(
            {self.config["symbol"]: price}
        )
        max_position_value = portfolio_value * self.config["risk"]["max_pos"]
        position_size = max_position_value / price

        return position_size

    def _log_trading_status(
        self, prediction: int, probabilities: np.ndarray, current_price: float
    ):
        """Log current trading status."""
        metrics = self.trade_manager.get_performance_metrics()
        portfolio_value = self.trade_manager.get_portfolio_value(
            {self.config["symbol"]: current_price}
        )

        status = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
            "current_price": current_price,
            "portfolio_value": portfolio_value,
            "open_positions": len(
                [p for p in self.trade_manager.positions if p.status == "open"]
            ),
            "total_return": metrics["total_return"],
            "win_rate": metrics["win_rate"],
            "num_trades": metrics["num_trades"],
        }

        logger.info(
            f"Trading Status: Prediction={prediction}, Price={current_price:.2f}, "
            f"Portfolio=${portfolio_value:.2f}, Return={metrics['total_return']:.2%}"
        )

        # Save status to file
        status_file = Path("./app/reports/trading_status.json")
        status_file.parent.mkdir(parents=True, exist_ok=True)

        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)

    def start(self):
        """Start the trading orchestrator for 24/7 operation."""
        logger.info("Starting trading orchestrator for 24/7 operation...")

        try:
            # Test connections
            logger.info("Testing system connections...")
            if not self.gateway.test_connection():
                logger.error("Exchange connection test failed")
                return False

            # Initial model training if not trained
            if not self.model.is_trained:
                logger.info("Model not trained, starting initial training...")
                self._daily_retrain_job()

            # Set system status
            self.system_status = "running"
            logger.info("Trading system started successfully - running 24/7")

            # Start scheduler with error handling
            try:
                self.scheduler.start()
            except KeyboardInterrupt:
                logger.info("Trading system stopped by user (Ctrl+C)")
            except Exception as e:
                logger.error(f"Critical error in trading system: {e}")
                logger.error(traceback.format_exc())
            finally:
                self.stop()

        except Exception as e:
            logger.critical(f"Failed to start trading system: {e}")
            logger.critical(traceback.format_exc())
            return False

        return True

    def stop(self):
        """Stop the trading orchestrator gracefully."""
        logger.info("Stopping trading orchestrator...")

        try:
            # Set system status
            self.system_status = "stopping"

            # Close all open positions gracefully
            logger.info("Closing all open positions...")
            current_prices = {self.config["symbol"]: 50000.0}  # Default price

            for position in self.trade_manager.positions:
                if position.status == "open":
                    try:
                        self.trade_manager.close_position(
                            position,
                            current_prices[self.config["symbol"]],
                            "system_shutdown",
                        )
                        logger.info(
                            f"Closed position: {position.symbol} {position.side}"
                        )
                    except Exception as e:
                        logger.error(f"Error closing position {position.id}: {e}")

            # Save final state
            logger.info("Saving final system state...")
            self.trade_manager.save_state("./app/reports/final_state.json")

            # Generate shutdown report
            self._generate_shutdown_report()

            # Shutdown scheduler
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)

            self.system_status = "stopped"
            logger.info("Trading system stopped gracefully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.error(traceback.format_exc())

    def _generate_shutdown_report(self):
        """Generate shutdown report."""
        try:
            shutdown_report = {
                "shutdown_time": datetime.now().isoformat(),
                "system_status": self.system_status,
                "final_metrics": self.trade_manager.get_performance_metrics(),
                "open_positions": len(
                    [p for p in self.trade_manager.positions if p.status == "open"]
                ),
                "total_trades": len(self.trade_manager.trades),
                "uptime_hours": self._calculate_uptime(),
                "shutdown_reason": "graceful_shutdown",
            }

            report_file = Path("./app/reports/shutdown_report.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, "w") as f:
                json.dump(shutdown_report, f, indent=2)

            logger.info("Shutdown report generated")

        except Exception as e:
            logger.error(f"Error generating shutdown report: {e}")

    def _calculate_uptime(self):
        """Calculate system uptime."""
        try:
            if self.last_prediction_time:
                return (
                    datetime.now() - self.last_prediction_time
                ).total_seconds() / 3600
            return 0
        except Exception:
            return 0

    def restart(self):
        """Restart the trading system."""
        logger.info("Restarting trading system...")
        self.stop()
        time.sleep(5)  # Wait 5 seconds
        return self.start()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            metrics = self.trade_manager.get_performance_metrics()

            return {
                "status": self.system_status,
                "last_prediction_time": (
                    self.last_prediction_time.isoformat()
                    if self.last_prediction_time
                    else None
                ),
                "last_retrain_time": (
                    self.last_retrain_time.isoformat()
                    if self.last_retrain_time
                    else None
                ),
                "last_health_check": (
                    self.last_health_check.isoformat()
                    if self.last_health_check
                    else None
                ),
                "consecutive_errors": self.consecutive_errors,
                "model_trained": self.model.is_trained,
                "open_positions": len(
                    [p for p in self.trade_manager.positions if p.status == "open"]
                ),
                "total_trades": len(self.trade_manager.trades),
                "performance_metrics": metrics,
                "uptime_hours": self._calculate_uptime(),
                "scheduler_running": (
                    self.scheduler.running
                    if hasattr(self.scheduler, "running")
                    else False
                ),
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"status": "error", "error": str(e)}
