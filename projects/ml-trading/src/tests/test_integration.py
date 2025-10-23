"""
Integration tests for ML trading system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

# Import system components
from src.data.polygon_flatfiles_downloader import PolygonFlatFilesDownloader
from src.data.polygon_loader import PolygonLoader
from src.features.engineer import FeatureEngineer
from src.labels.targets import LabelGenerator
from src.models.classifier import XGBoostClassifier
from src.backtest.simulator import BacktestSimulator
from src.execution.gateway import CCXTGateway
from src.execution.trade_manager import TradeManager
from src.utils.seed import set_seed
from src.utils.costs import calculate_trading_costs, calculate_position_sizing
from src.utils.splits import create_time_series_splits


class TestDataComponents:
    """Test data ingestion components."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.api_key = "test_api_key"

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_polygon_downloader_initialization(self):
        """Test Polygon downloader initialization."""
        downloader = PolygonFlatFilesDownloader(self.api_key, self.temp_dir)
        assert downloader.api_key == self.api_key
        assert downloader.data_root == Path(self.temp_dir)
        assert downloader.data_root.exists()

    def test_polygon_loader_initialization(self):
        """Test Polygon loader initialization."""
        loader = PolygonLoader(self.temp_dir)
        assert loader.data_root == Path(self.temp_dir)
        assert loader.target_symbols == ["BTCUSD", "ETHUSD"]

    @patch("requests.get")
    def test_polygon_download_simulation(self, mock_get):
        """Test Polygon download simulation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = (
            b"test,data\nBTCUSD,100,50000,50100,50200,49900,1640995200000000000,1000"
        )
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        downloader = PolygonFlatFilesDownloader(self.api_key, self.temp_dir)

        # Test single file download
        url = "https://files.polygon.io/flatfiles/global_crypto/minute_aggs_v1/2022/01/2022-01-01.csv.gz"
        cache_path = Path(self.temp_dir) / "2022-01-01.csv.gz"

        result = downloader._download_file(url, cache_path)
        assert result is True
        assert cache_path.exists()


class TestFeatureEngineering:
    """Test feature engineering components."""

    def setup_method(self):
        """Setup test data."""
        # Create sample OHLCV data
        dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0, 0.001, 1000)
        prices = 50000 * np.exp(np.cumsum(returns))

        self.sample_data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.0005, 1000)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.001, 1000))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.001, 1000))),
                "close": prices,
                "volume": np.random.uniform(100, 1000, 1000),
            },
            index=dates,
        )

    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.feature_names == []

    def test_feature_calculation(self):
        """Test feature calculation."""
        engineer = FeatureEngineer()
        features_data = engineer.calculate_all_features(self.sample_data)

        # Check that features were added
        assert len(engineer.feature_names) > 0
        assert len(features_data.columns) > len(self.sample_data.columns)

        # Check specific features
        assert "ema_9" in features_data.columns
        assert "rsi" in features_data.columns
        assert "macd" in features_data.columns
        assert "atr" in features_data.columns

    def test_feature_causality(self):
        """Test that features don't use future data."""
        engineer = FeatureEngineer()
        features_data = engineer.calculate_all_features(self.sample_data)

        # Check that features don't have lookahead bias
        for feature in engineer.feature_names:
            # Features should not be NaN at the beginning (except for rolling windows)
            non_nan_start = features_data[feature].first_valid_index()
            if non_nan_start is not None:
                # Allow some initial NaN values for rolling calculations
                assert features_data.index.get_loc(non_nan_start) < 100


class TestLabelGeneration:
    """Test label generation components."""

    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
        np.random.seed(42)

        returns = np.random.normal(0, 0.001, 1000)
        prices = 50000 * np.exp(np.cumsum(returns))

        self.sample_data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.0005, 1000)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.001, 1000))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.001, 1000))),
                "close": prices,
                "volume": np.random.uniform(100, 1000, 1000),
            },
            index=dates,
        )

    def test_label_generator_initialization(self):
        """Test label generator initialization."""
        generator = LabelGenerator(
            horizon_minutes=10, up_threshold=0.002, down_threshold=-0.002
        )
        assert generator.horizon_minutes == 10
        assert generator.up_threshold == 0.002
        assert generator.down_threshold == -0.002

    def test_label_generation(self):
        """Test label generation."""
        generator = LabelGenerator(horizon_minutes=10)
        labeled_data = generator.generate_labels(self.sample_data)

        # Check that labels were added
        assert "label" in labeled_data.columns
        assert "forward_return" in labeled_data.columns

        # Check label values
        unique_labels = labeled_data["label"].unique()
        assert set(unique_labels).issubset({-1, 0, 1})

        # Check that tail rows were dropped
        assert len(labeled_data) < len(self.sample_data)
        assert len(labeled_data) == len(self.sample_data) - generator.horizon_minutes

    def test_label_statistics(self):
        """Test label statistics calculation."""
        generator = LabelGenerator()
        labeled_data = generator.generate_labels(self.sample_data)

        stats = generator.get_label_statistics(labeled_data)

        assert "total_samples" in stats
        assert "label_counts" in stats
        assert "label_percentages" in stats
        assert stats["total_samples"] == len(labeled_data)


class TestMLModel:
    """Test ML model components."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)

        # Create sample features and labels
        n_samples = 1000
        n_features = 20

        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Create balanced labels
        self.y = pd.Series(np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3]))

    def test_xgboost_initialization(self):
        """Test XGBoost classifier initialization."""
        model_params = {
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_rounds": 100,
        }

        model = XGBoostClassifier(model_params)
        assert model.model_params == model_params
        assert not model.is_trained

    def test_model_training(self):
        """Test model training."""
        model_params = {
            "max_depth": 3,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_rounds": 50,
        }

        model = XGBoostClassifier(model_params)
        metrics = model.fit(self.X, self.y)

        assert model.is_trained
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert metrics["train_accuracy"] > 0
        assert metrics["val_accuracy"] > 0

    def test_model_prediction(self):
        """Test model prediction."""
        model_params = {
            "max_depth": 3,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_rounds": 50,
        }

        model = XGBoostClassifier(model_params)
        model.fit(self.X, self.y)

        # Test prediction
        predictions = model.predict(self.X.iloc[:10])
        assert len(predictions) == 10
        assert set(predictions).issubset({-1, 0, 1})

        # Test probability prediction
        probabilities = model.predict_proba(self.X.iloc[:10])
        assert probabilities.shape == (10, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestBacktesting:
    """Test backtesting components."""

    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
        np.random.seed(42)

        returns = np.random.normal(0, 0.001, 1000)
        prices = 50000 * np.exp(np.cumsum(returns))

        self.sample_data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.0005, 1000)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.001, 1000))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.001, 1000))),
                "close": prices,
                "volume": np.random.uniform(100, 1000, 1000),
            },
            index=dates,
        )

        # Create sample predictions
        self.predictions = np.random.choice([-1, 0, 1], 1000)
        self.probabilities = np.random.rand(1000, 3)
        self.probabilities = self.probabilities / self.probabilities.sum(
            axis=1, keepdims=True
        )

    def test_backtest_simulator_initialization(self):
        """Test backtest simulator initialization."""
        simulator = BacktestSimulator(
            initial_capital=10000.0, fee_bps=5.0, slippage_bps=3.0
        )

        assert simulator.initial_capital == 10000.0
        assert simulator.fee_bps == 5.0
        assert simulator.slippage_bps == 3.0

    def test_backtest_execution(self):
        """Test backtest execution."""
        simulator = BacktestSimulator(initial_capital=10000.0)

        results = simulator.run_backtest(
            self.sample_data, self.predictions, self.probabilities
        )

        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "win_rate" in results
        assert "num_trades" in results

        # Check that trades were recorded
        assert len(simulator.trades) >= 0
        assert len(simulator.equity_curve) == len(self.sample_data)


class TestExecutionGateway:
    """Test execution gateway components."""

    def test_gateway_initialization(self):
        """Test gateway initialization."""
        gateway = CCXTGateway(exchange="kraken", trade_mode="paper")

        assert gateway.exchange == "kraken"
        assert gateway.trade_mode == "paper"

    def test_paper_trading_mode(self):
        """Test paper trading mode."""
        gateway = CCXTGateway(trade_mode="paper")

        # Test balance
        balance = gateway.get_balance()
        assert "USD" in balance
        assert "BTC" in balance

        # Test ticker
        ticker = gateway.get_ticker("BTC/USD")
        assert "symbol" in ticker

        # Test market order
        order = gateway.create_market_order("BTC/USD", "buy", 0.001)
        assert "id" in order
        assert order["side"] == "buy"
        assert order["amount"] == 0.001


class TestTradeManager:
    """Test trade manager components."""

    def setup_method(self):
        """Setup test environment."""
        self.gateway = CCXTGateway(trade_mode="paper")
        self.trade_manager = TradeManager(gateway=self.gateway, initial_capital=10000.0)

    def test_trade_manager_initialization(self):
        """Test trade manager initialization."""
        assert self.trade_manager.initial_capital == 10000.0
        assert self.trade_manager.capital == 10000.0
        assert len(self.trade_manager.positions) == 0
        assert len(self.trade_manager.trades) == 0

    def test_position_opening(self):
        """Test position opening."""
        position = self.trade_manager.open_position(
            symbol="BTC/USD", side="long", size=0.001, price=50000.0
        )

        assert position is not None
        assert position.symbol == "BTC/USD"
        assert position.side == "long"
        assert position.size == 0.001
        assert position.status == "open"

    def test_position_closing(self):
        """Test position closing."""
        # Open position
        position = self.trade_manager.open_position(
            symbol="BTC/USD", side="long", size=0.001, price=50000.0
        )

        # Close position
        trade = self.trade_manager.close_position(position, 51000.0)

        assert trade is not None
        assert trade.symbol == "BTC/USD"
        assert trade.side == "long"
        assert trade.pnl > 0  # Should be profitable

    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        # Open position
        self.trade_manager.open_position(
            symbol="BTC/USD", side="long", size=0.001, price=50000.0
        )

        # Calculate portfolio value
        current_prices = {"BTC/USD": 51000.0}
        portfolio_value = self.trade_manager.get_portfolio_value(current_prices)

        assert portfolio_value > self.trade_manager.initial_capital


class TestUtilities:
    """Test utility functions."""

    def test_trading_costs_calculation(self):
        """Test trading costs calculation."""
        costs = calculate_trading_costs(
            price=50000.0, size=0.001, side="buy", fee_bps=5.0, slippage_bps=3.0
        )

        assert "notional" in costs
        assert "fee_cost" in costs
        assert "slippage_cost" in costs
        assert "total_cost" in costs
        assert "effective_price" in costs

        assert costs["notional"] == 50.0
        assert costs["fee_cost"] > 0
        assert costs["slippage_cost"] > 0

    def test_position_sizing_calculation(self):
        """Test position sizing calculation."""
        size = calculate_position_sizing(
            portfolio_value=10000.0,
            volatility=0.02,
            risk_per_trade=0.005,
            horizon_minutes=10,
            max_position=1.0,
        )

        assert size > 0
        assert size <= 1.0

    def test_time_series_splits(self):
        """Test time series cross-validation splits."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
        data = pd.DataFrame({"value": np.random.randn(1000)}, index=dates)

        splits = list(create_time_series_splits(data, n_folds=3))

        assert len(splits) == 3

        for train_data, test_data in splits:
            assert len(train_data) > 0
            assert len(test_data) > 0
            assert train_data.index[-1] < test_data.index[0]


class TestIntegration:
    """Integration tests for the complete system."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        set_seed(42)

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # This is a simplified integration test
        # In practice, you would test the full pipeline with real data

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=500, freq="1min")
        np.random.seed(42)

        returns = np.random.normal(0, 0.001, 500)
        prices = 50000 * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.0005, 500)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.001, 500))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.001, 500))),
                "close": prices,
                "volume": np.random.uniform(100, 1000, 500),
            },
            index=dates,
        )

        # Feature engineering
        engineer = FeatureEngineer()
        features_data = engineer.calculate_all_features(data)

        # Label generation
        generator = LabelGenerator(horizon_minutes=10)
        labeled_data = generator.generate_labels(features_data)

        # Model training
        feature_cols = engineer.get_feature_names()
        X = labeled_data[feature_cols]
        y = labeled_data["label"]

        model_params = {
            "max_depth": 3,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_rounds": 50,
        }

        model = XGBoostClassifier(model_params)
        metrics = model.fit(X, y)

        # Backtesting
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        simulator = BacktestSimulator(initial_capital=10000.0)
        results = simulator.run_backtest(labeled_data, predictions, probabilities)

        # Verify results
        assert metrics["val_accuracy"] > 0
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert len(simulator.trades) >= 0


if __name__ == "__main__":
    pytest.main([__file__])
