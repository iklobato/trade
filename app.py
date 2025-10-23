"""
Cryptocurrency Trading App - Sandbox Environment
================================================

A comprehensive trading application that runs enhanced strategies
in a sandbox environment with real market data and simulated trading.

Features:
- Real-time market data from multiple exchanges
- Sandbox trading simulation
- Enhanced strategy execution
- Portfolio management
- Risk management
- Performance monitoring
"""

import pandas as pd
import numpy as np
import ccxt
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Trading configuration."""
    exchange: str = 'binance'
    symbol: str = 'BTC/USDT'
    timeframe: str = '1h'
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    sandbox_mode: bool = True
    strategy_name: str = 'Enhanced Trend Following'
    update_interval: int = 3600  # 1 hour in seconds


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'stopped'


@dataclass
class Trade:
    """Completed trade."""
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


class ExchangeConnector:
    """Connect to cryptocurrency exchanges."""
    
    def __init__(self, exchange_name: str = 'binance', sandbox: bool = True):
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize exchange connection."""
        try:
            if self.exchange_name == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': os.getenv('BINANCE_API_KEY', ''),
                    'secret': os.getenv('BINANCE_SECRET', ''),
                    'enableRateLimit': True,
                    'sandbox': self.sandbox,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
            elif self.exchange_name == 'coinbase':
                self.exchange = ccxt.coinbasepro({
                    'apiKey': os.getenv('COINBASE_API_KEY', ''),
                    'secret': os.getenv('COINBASE_SECRET', ''),
                    'passphrase': os.getenv('COINBASE_PASSPHRASE', ''),
                    'enableRateLimit': True,
                    'sandbox': self.sandbox
                })
            elif self.exchange_name == 'kraken':
                self.exchange = ccxt.kraken({
                    'apiKey': os.getenv('KRAKEN_API_KEY', ''),
                    'secret': os.getenv('KRAKEN_SECRET', ''),
                    'enableRateLimit': True,
                    'sandbox': self.sandbox
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange_name}")
            
            logger.info(f"Connected to {self.exchange_name} (sandbox: {self.sandbox})")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """Get historical market data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            raise
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance."""
        try:
            if self.sandbox:
                # Simulate balance for sandbox
                return {'USDT': 10000.0, 'BTC': 0.0}
            else:
                balance = self.exchange.fetch_balance()
                return {k: v['free'] for k, v in balance.items() if v['free'] > 0}
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {'USDT': 10000.0, 'BTC': 0.0}
    
    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a trading order."""
        try:
            if self.sandbox:
                # Simulate order placement
                current_price = self.get_current_price(symbol)
                order = {
                    'id': f"sim_{int(time.time())}",
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price or current_price,
                    'status': 'filled',
                    'timestamp': int(time.time() * 1000)
                }
                logger.info(f"Simulated {side} order: {amount} {symbol} at {order['price']}")
                return order
            else:
                if price:
                    order = self.exchange.create_limit_order(symbol, side, amount, price)
                else:
                    order = self.exchange.create_market_order(symbol, side, amount)
                logger.info(f"Placed {side} order: {amount} {symbol}")
                return order
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise


class TechnicalIndicators:
    """Technical indicators for strategy signals."""
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # Momentum indicators
        df['RSI'] = TechnicalIndicators._calculate_rsi(df['close'], 14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = TechnicalIndicators._calculate_macd(df['close'])
        df['STOCH_K'], df['STOCH_D'] = TechnicalIndicators._calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = TechnicalIndicators._calculate_bollinger_bands(df['close'])
        df['ATR'] = TechnicalIndicators._calculate_atr(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['OBV'] = TechnicalIndicators._calculate_obv(df['close'], df['volume'])
        
        # Trend indicators
        df['ADX'] = TechnicalIndicators._calculate_adx(df['high'], df['low'], df['close'])
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close_prev = (high - close.shift()).abs()
        low_close_prev = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    @staticmethod
    def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)."""
        tr = TechnicalIndicators._calculate_atr(high, low, close, period)
        plus_dm = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        minus_dm = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / tr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx


class EnhancedTrendFollowingStrategy:
    """Enhanced trend following strategy."""
    
    def __init__(self, name: str = "Enhanced Trend Following"):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals."""
        df = data.copy()
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        if len(df) < 200:
            return {'signal': 'hold', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Trend conditions
        trend_up = latest['SMA_50'] > latest['SMA_200']
        price_above_sma = latest['close'] > latest['SMA_50']
        
        # Momentum conditions
        rsi_ok = 30 < latest['RSI'] < 70
        macd_bullish = latest['MACD'] > latest['MACD_signal']
        
        # Volatility conditions
        atr_normal = latest['ATR'] < df['ATR'].rolling(20).mean().iloc[-1] * 2
        
        # Trend strength
        adx_strong = latest['ADX'] > 25
        
        # Calculate signal strength
        conditions = [trend_up, price_above_sma, rsi_ok, macd_bullish, atr_normal, adx_strong]
        confidence = sum(conditions) / len(conditions)
        
        # Generate signal
        if confidence >= 0.8:
            signal = 'buy'
            reason = f"Strong trend signal (confidence: {confidence:.2f})"
        elif confidence <= 0.2:
            signal = 'sell'
            reason = f"Weak trend signal (confidence: {confidence:.2f})"
        else:
            signal = 'hold'
            reason = f"Neutral signal (confidence: {confidence:.2f})"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'indicators': {
                'sma_50': latest['SMA_50'],
                'sma_200': latest['SMA_200'],
                'rsi': latest['RSI'],
                'macd': latest['MACD'],
                'macd_signal': latest['MACD_signal'],
                'atr': latest['ATR'],
                'adx': latest['ADX']
            }
        }


class PortfolioManager:
    """Manage trading portfolio."""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_history = []
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total_value = self.balance
        
        for position in self.positions:
            if position.status == 'open':
                current_price = current_prices.get(position.symbol, position.current_price)
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
                total_value += position.unrealized_pnl
        
        return total_value
    
    def can_open_position(self, symbol: str, size: float, price: float) -> bool:
        """Check if we can open a new position."""
        required_balance = size * price
        return self.balance >= required_balance
    
    def open_position(self, symbol: str, side: str, size: float, price: float, 
                     stop_loss: float, take_profit: float) -> bool:
        """Open a new position."""
        if not self.can_open_position(symbol, size, price):
            return False
        
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=price
        )
        
        self.positions.append(position)
        self.balance -= size * price
        
        logger.info(f"Opened {side} position: {size} {symbol} at {price}")
        return True
    
    def close_position(self, position: Position, exit_price: float) -> Trade:
        """Close a position."""
        pnl = (exit_price - position.entry_price) * position.size
        pnl_pct = (exit_price - position.entry_price) / position.entry_price
        
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy="Enhanced Trend Following"
        )
        
        self.trades.append(trade)
        self.balance += position.size * exit_price
        position.status = 'closed'
        
        logger.info(f"Closed position: {trade.pnl:.2f} ({trade.pnl_pct:.2%})")
        return trade
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[Trade]:
        """Check stop loss and take profit levels."""
        closed_trades = []
        
        for position in self.positions:
            if position.status != 'open':
                continue
            
            current_price = current_prices.get(position.symbol, position.current_price)
            
            # Check stop loss
            if position.side == 'long' and current_price <= position.stop_loss:
                trade = self.close_position(position, current_price)
                closed_trades.append(trade)
                logger.info(f"Stop loss triggered for {position.symbol}")
            
            # Check take profit
            elif position.side == 'long' and current_price >= position.take_profit:
                trade = self.close_position(position, current_price)
                closed_trades.append(trade)
                logger.info(f"Take profit triggered for {position.symbol}")
        
        return closed_trades
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'num_trades': 0,
                'avg_trade_return': 0.0
            }
        
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_return = total_pnl / self.initial_balance
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades)
        
        total_profits = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else np.inf
        
        avg_trade_return = np.mean([t.pnl_pct for t in self.trades])
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(self.trades),
            'avg_trade_return': avg_trade_return
        }


class TradingApp:
    """Main trading application."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange = ExchangeConnector(config.exchange, config.sandbox_mode)
        self.strategy = EnhancedTrendFollowingStrategy(config.strategy_name)
        self.portfolio = PortfolioManager(config.initial_balance)
        self.running = False
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        config_file = Path('trading_config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                    logger.info("Loaded saved configuration")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            config_dict = asdict(self.config)
            with open('trading_config.json', 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def run_strategy_cycle(self):
        """Run one cycle of the trading strategy."""
        try:
            # Get market data
            logger.info(f"Fetching market data for {self.config.symbol}")
            data = self.exchange.get_market_data(
                self.config.symbol, 
                self.config.timeframe, 
                limit=200
            )
            
            # Generate signals
            signal_data = self.strategy.generate_signals(data)
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            reason = signal_data['reason']
            
            logger.info(f"Signal: {signal} (confidence: {confidence:.2f}) - {reason}")
            
            # Get current price
            current_price = self.exchange.get_current_price(self.config.symbol)
            
            # Check existing positions
            current_prices = {self.config.symbol: current_price}
            closed_trades = self.portfolio.check_stop_loss_take_profit(current_prices)
            
            # Process new signals
            if signal == 'buy' and confidence > 0.7:
                self._process_buy_signal(current_price, confidence)
            elif signal == 'sell' and confidence > 0.7:
                self._process_sell_signal(current_price, confidence)
            
            # Update portfolio value
            portfolio_value = self.portfolio.get_portfolio_value(current_prices)
            self.portfolio.equity_history.append({
                'timestamp': datetime.now(),
                'value': portfolio_value,
                'balance': self.portfolio.balance
            })
            
            # Log status
            self._log_status(current_price, portfolio_value, signal_data)
            
        except Exception as e:
            logger.error(f"Error in strategy cycle: {e}")
    
    def _process_buy_signal(self, current_price: float, confidence: float):
        """Process buy signal."""
        # Calculate position size
        portfolio_value = self.portfolio.get_portfolio_value({self.config.symbol: current_price})
        position_size = min(
            portfolio_value * self.config.max_position_size / current_price,
            self.portfolio.balance / current_price
        )
        
        if position_size > 0.001:  # Minimum position size
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 - self.config.stop_loss_pct)
            take_profit = current_price * (1 + self.config.take_profit_pct)
            
            # Open position
            success = self.portfolio.open_position(
                self.config.symbol,
                'long',
                position_size,
                current_price,
                stop_loss,
                take_profit
            )
            
            if success:
                logger.info(f"Opened long position: {position_size:.6f} {self.config.symbol}")
    
    def _process_sell_signal(self, current_price: float, confidence: float):
        """Process sell signal."""
        # Close existing long positions
        for position in self.portfolio.positions:
            if position.status == 'open' and position.side == 'long':
                self.portfolio.close_position(position, current_price)
                logger.info(f"Closed long position due to sell signal")
    
    def _log_status(self, current_price: float, portfolio_value: float, signal_data: Dict):
        """Log current status."""
        metrics = self.portfolio.get_performance_metrics()
        
        logger.info("=" * 60)
        logger.info(f"Trading Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Current Price: ${current_price:.2f}")
        logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
        logger.info(f"Balance: ${self.portfolio.balance:.2f}")
        logger.info(f"Open Positions: {len([p for p in self.portfolio.positions if p.status == 'open'])}")
        logger.info(f"Signal: {signal_data['signal']} (confidence: {signal_data['confidence']:.2f})")
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Win Rate: {metrics['win_rate']:.1%}")
        logger.info(f"Number of Trades: {metrics['num_trades']}")
        logger.info("=" * 60)
    
    def start_trading(self):
        """Start the trading loop."""
        logger.info("Starting trading application...")
        logger.info(f"Exchange: {self.config.exchange}")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Strategy: {self.config.strategy_name}")
        logger.info(f"Sandbox Mode: {self.config.sandbox_mode}")
        
        self.running = True
        
        try:
            while self.running:
                self.run_strategy_cycle()
                
                # Wait for next cycle
                logger.info(f"Waiting {self.config.update_interval} seconds for next cycle...")
                time.sleep(self.config.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Trading error: {e}")
        finally:
            self.stop_trading()
    
    def stop_trading(self):
        """Stop trading and close all positions."""
        logger.info("Stopping trading application...")
        self.running = False
        
        # Close all open positions
        current_price = self.exchange.get_current_price(self.config.symbol)
        for position in self.portfolio.positions:
            if position.status == 'open':
                self.portfolio.close_position(position, current_price)
        
        # Final performance report
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate final performance report."""
        metrics = self.portfolio.get_performance_metrics()
        
        logger.info("=" * 60)
        logger.info("FINAL TRADING REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Win Rate: {metrics['win_rate']:.1%}")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Number of Trades: {metrics['num_trades']}")
        logger.info(f"Average Trade Return: {metrics['avg_trade_return']:.2%}")
        logger.info(f"Final Portfolio Value: ${self.portfolio.get_portfolio_value({self.config.symbol: self.exchange.get_current_price(self.config.symbol)}):.2f}")
        logger.info("=" * 60)


def main():
    """Main function to run the trading app."""
    print("🚀 Cryptocurrency Trading App - Sandbox Environment")
    print("=" * 60)
    
    # Create configuration
    config = TradingConfig(
        exchange='binance',
        symbol='BTC/USDT',
        timeframe='1h',
        initial_balance=10000.0,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        sandbox_mode=True,
        strategy_name='Enhanced Trend Following',
        update_interval=3600
    )
    
    # Create and start trading app
    app = TradingApp(config)
    
    try:
        app.start_trading()
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        app.stop_trading()


if __name__ == "__main__":
    main()
