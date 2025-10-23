"""
Trading cost calculations including fees and slippage.
"""

import numpy as np
from typing import Dict


def calculate_trading_costs(
    price: float,
    size: float,
    side: str,
    fee_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> Dict[str, float]:
    """
    Calculate trading costs for a given trade.

    Args:
        price: Trade price
        size: Trade size
        side: 'buy' or 'sell'
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points

    Returns:
        Dictionary with cost breakdown
    """
    notional = price * size

    # Calculate fees
    fee_rate = fee_bps / 10000
    fee_cost = notional * fee_rate

    # Calculate slippage cost
    slippage_rate = slippage_bps / 10000
    slippage_cost = notional * slippage_rate

    # Total cost
    total_cost = fee_cost + slippage_cost

    return {
        "notional": notional,
        "fee_cost": fee_cost,
        "slippage_cost": slippage_cost,
        "total_cost": total_cost,
        "effective_price": (
            price * (1 + slippage_rate)
            if side == "buy"
            else price * (1 - slippage_rate)
        ),
    }


def calculate_position_sizing(
    portfolio_value: float,
    volatility: float,
    risk_per_trade: float = 0.005,
    horizon_minutes: int = 10,
    max_position: float = 1.0,
) -> float:
    """
    Calculate position size based on volatility and risk parameters.

    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Risk per trade as fraction of portfolio
        volatility: Current volatility estimate
        horizon_minutes: Holding period in minutes
        max_position: Maximum position size as fraction of portfolio

    Returns:
        Position size as fraction of portfolio
    """
    # Kelly-style position sizing adjusted for volatility and horizon
    horizon_hours = horizon_minutes / 60
    volatility_adjusted = volatility * np.sqrt(horizon_hours)

    # Calculate position size
    position_size = min(risk_per_trade / volatility_adjusted, max_position)

    # Ensure minimum viable position
    min_position = 0.001
    position_size = max(position_size, min_position)

    return position_size


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a series of returns.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Array of portfolio values over time

    Returns:
        Maximum drawdown as negative percentage
    """
    if len(equity_curve) == 0:
        return 0.0

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    return np.min(drawdown)


def calculate_profit_factor(profits: np.ndarray, losses: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        profits: Array of profitable trade returns
        losses: Array of losing trade returns (positive values)

    Returns:
        Profit factor
    """
    total_profit = np.sum(profits)
    total_loss = np.sum(losses)

    if total_loss == 0:
        return np.inf if total_profit > 0 else 0.0

    return total_profit / total_loss
