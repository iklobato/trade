"""
Backtesting simulator for ML trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from src.utils.costs import (
    calculate_trading_costs,
    calculate_position_sizing,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
)

logger = logging.getLogger(__name__)


class BacktestSimulator:
    """
    Simulate trading based on ML model predictions.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_bps: float = 5.0,
        slippage_bps: float = 3.0,
        risk_per_trade: float = 0.005,
        max_position: float = 1.0,
        horizon_minutes: int = 10,
    ):
        self.initial_capital = initial_capital
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position
        self.horizon_minutes = horizon_minutes

        # Trading state
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []

    def run_backtest(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        up_threshold: float = 0.5,
        down_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.

        Args:
            data: DataFrame with OHLCV data
            predictions: Model predictions (-1, 0, 1)
            probabilities: Model probabilities [P(down), P(flat), P(up)]
            up_threshold: Threshold for long positions
            down_threshold: Threshold for short positions

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with {len(data)} periods")

        # Reset state
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []

        # Process each period
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]["close"]
            current_volatility = self._estimate_volatility(
                data.iloc[max(0, i - 60) : i + 1]
            )

            # Update existing positions
            self._update_positions(current_time, current_price)

            # Check for new signals
            if i < len(predictions):
                pred = predictions[i]
                prob_up = probabilities[i][2]  # P(up)
                prob_down = probabilities[i][0]  # P(down)

                # Long signal
                if pred == 1 and prob_up >= up_threshold:
                    self._open_position(
                        "long", current_time, current_price, current_volatility
                    )

                # Short signal
                elif pred == -1 and prob_down >= down_threshold:
                    self._open_position(
                        "short", current_time, current_price, current_volatility
                    )

            # Record equity
            portfolio_value = self._calculate_portfolio_value(current_price)
            self.equity_curve.append(
                {
                    "timestamp": current_time,
                    "equity": portfolio_value,
                    "capital": self.capital,
                    "positions": len(self.positions),
                }
            )

        # Close any remaining positions
        final_price = data.iloc[-1]["close"]
        self._close_all_positions(data.index[-1], final_price)

        # Calculate final metrics
        results = self._calculate_metrics()

        logger.info(
            f"Backtest completed: {len(self.trades)} trades, {results['total_return']:.2%} return"
        )

        return results

    def _estimate_volatility(self, data_window: pd.DataFrame) -> float:
        """
        Estimate current volatility from recent price data.

        Args:
            data_window: Recent price data window

        Returns:
            Volatility estimate
        """
        if len(data_window) < 2:
            return 0.02  # Default 2% volatility

        returns = data_window["close"].pct_change().dropna()
        if len(returns) == 0:
            return 0.02

        return returns.std()

    def _open_position(
        self, side: str, timestamp: datetime, price: float, volatility: float
    ):
        """
        Open a new trading position.

        Args:
            side: 'long' or 'short'
            timestamp: Entry timestamp
            price: Entry price
            volatility: Current volatility estimate
        """
        # Calculate position size
        position_size = calculate_position_sizing(
            self.capital,
            volatility,
            self.risk_per_trade,
            self.horizon_minutes,
            self.max_position,
        )

        # Calculate costs
        costs = calculate_trading_costs(
            price, position_size, side, self.fee_bps, self.slippage_bps
        )

        # Check if we have enough capital
        required_capital = costs["notional"] + costs["total_cost"]
        if required_capital > self.capital:
            logger.warning(
                f"Insufficient capital for {side} position: {required_capital:.2f} > {self.capital:.2f}"
            )
            return

        # Create position
        position = {
            "id": len(self.positions),
            "side": side,
            "entry_time": timestamp,
            "entry_price": costs["effective_price"],
            "size": position_size,
            "costs": costs,
            "exit_time": None,
            "exit_price": None,
            "pnl": None,
            "status": "open",
        }

        # Update capital
        self.capital -= required_capital
        self.positions.append(position)

        logger.debug(
            f"Opened {side} position: {position_size:.6f} at {costs['effective_price']:.2f}"
        )

    def _update_positions(self, current_time: datetime, current_price: float):
        """
        Update existing positions and check for exits.

        Args:
            current_time: Current timestamp
            current_price: Current price
        """
        positions_to_close = []

        for position in self.positions:
            if position["status"] != "open":
                continue

            # Check if position should be closed (time-based exit)
            time_held = current_time - position["entry_time"]
            if time_held >= timedelta(minutes=self.horizon_minutes):
                positions_to_close.append(position)

        # Close positions
        for position in positions_to_close:
            self._close_position(position, current_time, current_price)

    def _close_position(self, position: Dict, exit_time: datetime, exit_price: float):
        """
        Close a trading position.

        Args:
            position: Position dictionary
            exit_time: Exit timestamp
            exit_price: Exit price
        """
        # Calculate exit costs
        exit_costs = calculate_trading_costs(
            exit_price,
            position["size"],
            "sell" if position["side"] == "long" else "buy",
            self.fee_bps,
            self.slippage_bps,
        )

        # Calculate PnL
        if position["side"] == "long":
            gross_pnl = (
                exit_costs["effective_price"] - position["entry_price"]
            ) * position["size"]
        else:
            gross_pnl = (
                position["entry_price"] - exit_costs["effective_price"]
            ) * position["size"]

        net_pnl = gross_pnl - position["costs"]["total_cost"] - exit_costs["total_cost"]

        # Update position
        position["exit_time"] = exit_time
        position["exit_price"] = exit_costs["effective_price"]
        position["pnl"] = net_pnl
        position["status"] = "closed"

        # Update capital
        self.capital += exit_costs["notional"] - exit_costs["total_cost"]

        # Record trade
        trade = {
            "position_id": position["id"],
            "side": position["side"],
            "entry_time": position["entry_time"],
            "exit_time": exit_time,
            "entry_price": position["entry_price"],
            "exit_price": exit_costs["effective_price"],
            "size": position["size"],
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "total_costs": position["costs"]["total_cost"] + exit_costs["total_cost"],
            "holding_period": (exit_time - position["entry_time"]).total_seconds() / 60,
        }

        self.trades.append(trade)

        logger.debug(f"Closed {position['side']} position: PnL {net_pnl:.2f}")

    def _close_all_positions(self, exit_time: datetime, exit_price: float):
        """
        Close all remaining open positions.

        Args:
            exit_time: Exit timestamp
            exit_price: Exit price
        """
        for position in self.positions:
            if position["status"] == "open":
                self._close_position(position, exit_time, exit_price)

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        Calculate current portfolio value.

        Args:
            current_price: Current market price

        Returns:
            Total portfolio value
        """
        total_value = self.capital

        for position in self.positions:
            if position["status"] == "open":
                if position["side"] == "long":
                    unrealized_pnl = (
                        current_price - position["entry_price"]
                    ) * position["size"]
                else:
                    unrealized_pnl = (
                        position["entry_price"] - current_price
                    ) * position["size"]

                total_value += unrealized_pnl

        return total_value

    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate backtest performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "num_trades": 0,
                "avg_trade_return": 0.0,
            }

        # Calculate returns
        equity_values = [point["equity"] for point in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]

        # Basic metrics
        total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 0 else 0.0
        max_drawdown = calculate_max_drawdown(np.array(equity_values))

        # Trade metrics
        trade_pnls = [trade["net_pnl"] for trade in self.trades]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]

        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0
        profit_factor = calculate_profit_factor(
            np.array(winning_trades), np.array([abs(pnl) for pnl in losing_trades])
        )

        avg_trade_return = np.mean(trade_pnls) if trade_pnls else 0.0

        metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": len(self.trades),
            "avg_trade_return": avg_trade_return,
            "total_pnl": sum(trade_pnls),
            "gross_profit": sum(winning_trades),
            "gross_loss": sum(losing_trades),
            "final_capital": (
                equity_values[-1] if equity_values else self.initial_capital
            ),
        }

        return metrics

    def save_results(self, results_dir: str = "./app/reports"):
        """
        Save backtest results to files.

        Args:
            results_dir: Directory to save results
        """
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Save trade log
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(results_path / "trade_log.csv", index=False)

        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(results_path / "equity_curve.csv", index=False)

        # Save summary metrics
        metrics = self._calculate_metrics()
        with open(results_path / "summary.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Backtest results saved to {results_path}")

    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades.

        Returns:
            DataFrame with trade summary
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve data.

        Returns:
            DataFrame with equity curve
        """
        if not self.equity_curve:
            return pd.DataFrame()

        return pd.DataFrame(self.equity_curve)
