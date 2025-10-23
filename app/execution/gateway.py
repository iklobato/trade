"""
CCXT execution gateway for live trading.
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
import os
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CCXTGateway:
    """
    Unified execution gateway using CCXT for multiple exchanges.
    """
    
    def __init__(self, 
                 exchange: str = "kraken",
                 trade_mode: str = "live",
                 sandbox: bool = False):
        self.exchange = exchange
        self.trade_mode = trade_mode
        self.sandbox = sandbox
        
        self.exchange_instance = None
        self.logs_dir = Path("./app/execution/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchange
        self._initialize_exchange()
        
    def _initialize_exchange(self):
        """Initialize exchange connection."""
        try:
            if self.exchange == "kraken":
                self.exchange_instance = ccxt.kraken({
                    'apiKey': os.getenv('KRAKEN_API_KEY', ''),
                    'secret': os.getenv('KRAKEN_SECRET', ''),
                    'enableRateLimit': True,
                    'sandbox': self.sandbox
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange}")
            
            logger.info(f"Initialized {self.exchange} exchange")
            logger.info(f"Trade mode: {self.trade_mode}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange}: {e}")
            raise
    
    def _rate_limit(self):
        """Apply rate limiting."""
        if self.exchange_instance:
            self.exchange_instance.sleep()
    
    def _log_trade(self, action: str, data: Dict[str, Any]):
        """Log trade activity."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'exchange': self.exchange,
            'data': data
        }
        
        log_file = self.logs_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    def get_balance(self) -> Dict[str, float]:
        """
        Get account balance.
        
        Returns:
            Dictionary with balance information
        """
        try:
            if self.trade_mode == "paper":
                # Return simulated balance
                return {'USD': 10000.0, 'BTC': 0.0, 'ETH': 0.0}
            
            self._rate_limit()
            balance = self.exchange_instance.fetch_balance()
            
            # Extract free balances
            free_balance = {k: v['free'] for k, v in balance.items() if v['free'] > 0}
            
            logger.info(f"Account balance: {free_balance}")
            return free_balance
            
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            
        Returns:
            Dictionary with ticker data
        """
        try:
            self._rate_limit()
            ticker = self.exchange_instance.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return {}
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get order book data.
        
        Args:
            symbol: Trading symbol
            limit: Number of orders to fetch
            
        Returns:
            Dictionary with order book data
        """
        try:
            self._rate_limit()
            orderbook = self.exchange_instance.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'bids': orderbook['bids'],
                'asks': orderbook['asks'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            return {}
    
    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """
        Create a market order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            
        Returns:
            Dictionary with order information
        """
        try:
            if self.trade_mode == "paper":
                # Simulate market order
                ticker = self.get_ticker(symbol)
                price = ticker.get('last', 0)
                
                order = {
                    'id': f"paper_{int(time.time())}",
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'status': 'filled',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'market'
                }
                
                self._log_trade('market_order', order)
                logger.info(f"Paper market {side} order: {amount} {symbol} at {price}")
                return order
            
            self._rate_limit()
            order = self.exchange_instance.create_market_order(symbol, side, amount)
            
            self._log_trade('market_order', order)
            logger.info(f"Market {side} order created: {amount} {symbol}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to create market order: {e}")
            raise
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict[str, Any]:
        """
        Create a limit order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price
            
        Returns:
            Dictionary with order information
        """
        try:
            if self.trade_mode == "paper":
                # Simulate limit order
                order = {
                    'id': f"paper_{int(time.time())}",
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'status': 'open',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'limit'
                }
                
                self._log_trade('limit_order', order)
                logger.info(f"Paper limit {side} order: {amount} {symbol} at {price}")
                return order
            
            self._rate_limit()
            order = self.exchange_instance.create_limit_order(symbol, side, amount, price)
            
            self._log_trade('limit_order', order)
            logger.info(f"Limit {side} order created: {amount} {symbol} at {price}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to create limit order: {e}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.trade_mode == "paper":
                # Simulate order cancellation
                self._log_trade('cancel_order', {'order_id': order_id, 'symbol': symbol})
                logger.info(f"Paper order cancelled: {order_id}")
                return True
            
            self._rate_limit()
            result = self.exchange_instance.cancel_order(order_id, symbol)
            
            self._log_trade('cancel_order', {'order_id': order_id, 'symbol': symbol, 'result': result})
            logger.info(f"Order cancelled: {order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Fetch order status.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            
        Returns:
            Dictionary with order information
        """
        try:
            if self.trade_mode == "paper":
                # Return simulated order status
                return {
                    'id': order_id,
                    'symbol': symbol,
                    'status': 'filled',
                    'timestamp': datetime.now().isoformat()
                }
            
            self._rate_limit()
            order = self.exchange_instance.fetch_order(order_id, symbol)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            return {}
    
    def fetch_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of open orders
        """
        try:
            if self.trade_mode == "paper":
                # Return empty list for paper trading
                return []
            
            self._rate_limit()
            orders = self.exchange_instance.fetch_open_orders(symbol)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []
    
    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """
        Get trading fees for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with fee information
        """
        try:
            if self.trade_mode == "paper":
                # Return simulated fees
                return {'maker': 0.001, 'taker': 0.001}
            
            self._rate_limit()
            fees = self.exchange_instance.fetch_trading_fees(symbol)
            
            return fees
            
        except Exception as e:
            logger.error(f"Failed to fetch trading fees: {e}")
            return {'maker': 0.001, 'taker': 0.001}  # Default fees
    
    def switch_exchange(self, exchange_name: str) -> bool:
        """
        Switch to a different exchange.
        
        Args:
            exchange_name: Name of exchange to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange {exchange_name} not available")
            return False
        
        self.exchange_instance = self.exchanges[exchange_name]
        logger.info(f"Switched to exchange: {exchange_name}")
        
        return True
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about available exchanges.
        
        Returns:
            Dictionary with exchange information
        """
        info = {
            'exchange': self.exchange,
            'trade_mode': self.trade_mode,
            'sandbox': self.sandbox,
            'available_exchanges': [self.exchange]
        }
        
        return info
    
    def test_connection(self) -> bool:
        """
        Test exchange connection.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            if self.trade_mode == "paper":
                return True
            
            # Try to fetch balance
            balance = self.get_balance()
            return len(balance) > 0
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
