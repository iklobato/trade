"""
Trade manager for order execution and position tracking.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json
from dataclasses import dataclass, asdict

from app.execution.gateway import CCXTGateway
from app.utils.costs import calculate_position_sizing, calculate_trading_costs

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position data structure."""

    id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "open"  # 'open', 'closed', 'stopped'
    order_id: Optional[str] = None


@dataclass
class Trade:
    """Completed trade data structure."""

    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    strategy: str
    order_id: Optional[str] = None


class TradeManager:
    """
    Manage trading positions and order execution.
    """

    def __init__(
        self,
        gateway: CCXTGateway,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.005,
        max_position: float = 1.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ):
        self.gateway = gateway
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # State tracking
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.capital = initial_capital
        self.position_counter = 0
        self.trade_counter = 0

        # Performance tracking
        self.equity_history = []
        self.daily_pnl = {}

        # Risk management
        self.max_daily_loss = initial_capital * 0.05  # 5% max daily loss
        self.max_positions = 5  # Maximum concurrent positions

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.

        Args:
            current_prices: Dictionary with current prices for each symbol

        Returns:
            Total portfolio value
        """
        total_value = self.capital

        for position in self.positions:
            if position.status == "open":
                current_price = current_prices.get(
                    position.symbol, position.current_price
                )
                position.current_price = current_price

                if position.side == "long":
                    position.unrealized_pnl = (
                        current_price - position.entry_price
                    ) * position.size
                else:
                    position.unrealized_pnl = (
                        position.entry_price - current_price
                    ) * position.size

                total_value += position.unrealized_pnl

        return total_value

    def can_open_position(self, symbol: str, size: float, price: float) -> bool:
        """
        Check if we can open a new position.

        Args:
            symbol: Trading symbol
            size: Position size
            price: Entry price

        Returns:
            True if position can be opened, False otherwise
        """
        # Check capital requirements
        required_capital = size * price
        if required_capital > self.capital:
            logger.warning(
                f"Insufficient capital for {symbol}: {required_capital:.2f} > {self.capital:.2f}"
            )
            return False

        # Check maximum positions
        open_positions = len([p for p in self.positions if p.status == "open"])
        if open_positions >= self.max_positions:
            logger.warning(f"Maximum positions reached: {open_positions}")
            return False

        # Check daily loss limit
        today = datetime.now().date()
        daily_pnl = self.daily_pnl.get(today, 0)
        if daily_pnl < -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {daily_pnl:.2f}")
            return False

        return True

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = "ML",
    ) -> Optional[Position]:
        """
        Open a new trading position.

        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            price: Entry price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            strategy: Strategy name

        Returns:
            Position object if successful, None otherwise
        """
        if not self.can_open_position(symbol, size, price):
            return None

        # Calculate position size based on risk
        volatility = self._estimate_volatility(symbol)
        risk_adjusted_size = calculate_position_sizing(
            self.capital,
            volatility,
            self.risk_per_trade,
            10,  # horizon_minutes
            self.max_position,
        )

        # Use smaller of requested size and risk-adjusted size
        final_size = min(size, risk_adjusted_size)

        # Calculate costs
        costs = calculate_trading_costs(
            price, final_size, side, 5.0, 3.0
        )  # 5bps fee, 3bps slippage

        # Check if we still have enough capital after costs
        total_cost = costs["notional"] + costs["total_cost"]
        if total_cost > self.capital:
            logger.warning(
                f"Insufficient capital after costs: {total_cost:.2f} > {self.capital:.2f}"
            )
            return None

        # Create position
        position = Position(
            id=f"pos_{self.position_counter}",
            symbol=symbol,
            side=side,
            size=final_size,
            entry_price=costs["effective_price"],
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=costs["effective_price"],
            status="open",
        )

        # Execute order
        try:
            order = self.gateway.create_market_order(symbol, side, final_size)
            position.order_id = order.get("id")

            # Update capital
            self.capital -= total_cost

            # Add position
            self.positions.append(position)
            self.position_counter += 1

            logger.info(
                f"Opened {side} position: {final_size:.6f} {symbol} at {costs['effective_price']:.2f}"
            )

            return position

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return None

    def close_position(
        self, position: Position, exit_price: float, reason: str = "manual"
    ) -> Optional[Trade]:
        """
        Close a trading position.

        Args:
            position: Position to close
            exit_price: Exit price
            reason: Reason for closing

        Returns:
            Trade object if successful, None otherwise
        """
        if position.status != "open":
            logger.warning(f"Position {position.id} is not open")
            return None

        try:
            # Calculate exit costs
            exit_costs = calculate_trading_costs(
                exit_price,
                position.size,
                "sell" if position.side == "long" else "buy",
                5.0,
                3.0,
            )

            # Calculate PnL
            if position.side == "long":
                gross_pnl = (
                    exit_costs["effective_price"] - position.entry_price
                ) * position.size
            else:
                gross_pnl = (
                    position.entry_price - exit_costs["effective_price"]
                ) * position.size

            net_pnl = gross_pnl - exit_costs["total_cost"]
            pnl_pct = net_pnl / (position.entry_price * position.size)

            # Execute exit order
            exit_order = self.gateway.create_market_order(
                position.symbol,
                "sell" if position.side == "long" else "buy",
                position.size,
            )

            # Create trade record
            trade = Trade(
                id=f"trade_{self.trade_counter}",
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=exit_costs["effective_price"],
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                strategy="ML",
                order_id=exit_order.get("id"),
            )

            # Update capital
            self.capital += exit_costs["notional"] - exit_costs["total_cost"]

            # Update position
            position.status = "closed"
            position.current_price = exit_costs["effective_price"]
            position.unrealized_pnl = net_pnl

            # Add trade
            self.trades.append(trade)
            self.trade_counter += 1

            # Update daily PnL
            today = datetime.now().date()
            self.daily_pnl[today] = self.daily_pnl.get(today, 0) + net_pnl

            logger.info(
                f"Closed {position.side} position: PnL {net_pnl:.2f} ({pnl_pct:.2%}) - {reason}"
            )

            return trade

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None

    def check_stop_loss_take_profit(
        self, current_prices: Dict[str, float]
    ) -> List[Trade]:
        """
        Check stop loss and take profit levels for all open positions.

        Args:
            current_prices: Dictionary with current prices

        Returns:
            List of closed trades
        """
        closed_trades = []

        for position in self.positions:
            if position.status != "open":
                continue

            current_price = current_prices.get(position.symbol, position.current_price)

            # Check stop loss
            if position.stop_loss is not None:
                if position.side == "long" and current_price <= position.stop_loss:
                    trade = self.close_position(position, current_price, "stop_loss")
                    if trade:
                        closed_trades.append(trade)
                elif position.side == "short" and current_price >= position.stop_loss:
                    trade = self.close_position(position, current_price, "stop_loss")
                    if trade:
                        closed_trades.append(trade)

            # Check take profit
            if position.take_profit is not None:
                if position.side == "long" and current_price >= position.take_profit:
                    trade = self.close_position(position, current_price, "take_profit")
                    if trade:
                        closed_trades.append(trade)
                elif position.side == "short" and current_price <= position.take_profit:
                    trade = self.close_position(position, current_price, "take_profit")
                    if trade:
                        closed_trades.append(trade)

        return closed_trades

    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update all open positions with current prices.

        Args:
            current_prices: Dictionary with current prices
        """
        for position in self.positions:
            if position.status == "open":
                current_price = current_prices.get(
                    position.symbol, position.current_price
                )
                position.current_price = current_price

                if position.side == "long":
                    position.unrealized_pnl = (
                        current_price - position.entry_price
                    ) * position.size
                else:
                    position.unrealized_pnl = (
                        position.entry_price - current_price
                    ) * position.size

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "num_trades": 0,
                "avg_trade_return": 0.0,
                "total_pnl": 0.0,
            }

        # Calculate metrics
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_return = total_pnl / self.initial_capital

        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades)

        total_profits = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else np.inf

        avg_trade_return = np.mean([t.pnl_pct for t in self.trades])

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": len(self.trades),
            "avg_trade_return": avg_trade_return,
            "total_pnl": total_pnl,
            "gross_profit": total_profits,
            "gross_loss": total_losses,
        }

    def _estimate_volatility(self, symbol: str) -> float:
        """
        Estimate volatility for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Volatility estimate
        """
        # This is a simplified implementation
        # In practice, you would calculate volatility from recent price data
        return 0.02  # Default 2% volatility

    def save_state(self, filepath: str):
        """
        Save current state to file.

        Args:
            filepath: Path to save state
        """
        state = {
            "capital": self.capital,
            "position_counter": self.position_counter,
            "trade_counter": self.trade_counter,
            "daily_pnl": {str(k): v for k, v in self.daily_pnl.items()},
            "positions": [asdict(p) for p in self.positions],
            "trades": [asdict(t) for t in self.trades],
            "equity_history": self.equity_history,
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"State saved to {filepath}")

    def load_state(self, filepath: str):
        """
        Load state from file.

        Args:
            filepath: Path to load state from
        """
        with open(filepath, "r") as f:
            state = json.load(f)

        self.capital = state["capital"]
        self.position_counter = state["position_counter"]
        self.trade_counter = state["trade_counter"]
        self.daily_pnl = {
            datetime.fromisoformat(k).date(): v for k, v in state["daily_pnl"].items()
        }

        # Reconstruct positions and trades
        self.positions = []
        for pos_data in state["positions"]:
            pos_data["entry_time"] = datetime.fromisoformat(pos_data["entry_time"])
            self.positions.append(Position(**pos_data))

        self.trades = []
        for trade_data in state["trades"]:
            trade_data["entry_time"] = datetime.fromisoformat(trade_data["entry_time"])
            trade_data["exit_time"] = datetime.fromisoformat(trade_data["exit_time"])
            self.trades.append(Trade(**trade_data))

        self.equity_history = state["equity_history"]

        logger.info(f"State loaded from {filepath}")
