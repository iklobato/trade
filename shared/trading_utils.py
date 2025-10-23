"""
Common trading utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class Trade:
    """Trade data structure."""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float = 0.0
    pnl: float = 0.0


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0


def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    Calculate position size based on risk management.
    
    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Risk percentage per trade (0.01 = 1%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
        
    Returns:
        Position size in units
    """
    risk_amount = portfolio_value * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0.0
    
    return risk_amount / price_risk


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
    fee_rate: float = 0.001
) -> float:
    """
    Calculate profit/loss for a trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Trade quantity
        side: 'buy' or 'sell'
        fee_rate: Trading fee rate
        
    Returns:
        PnL amount
    """
    if side == 'buy':
        pnl = (exit_price - entry_price) * quantity
    else:  # sell
        pnl = (entry_price - exit_price) * quantity
    
    # Subtract fees
    fees = (entry_price + exit_price) * quantity * fee_rate
    return pnl - fees


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        Maximum drawdown as percentage
    """
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def calculate_win_rate(trades: List[Trade]) -> float:
    """
    Calculate win rate from trades.
    
    Args:
        trades: List of Trade objects
        
    Returns:
        Win rate as percentage
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.pnl > 0)
    return winning_trades / len(trades)
