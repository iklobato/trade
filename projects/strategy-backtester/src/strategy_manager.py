"""
Strategy Manager - Organize and Manage Multiple Trading Strategies
================================================================

A comprehensive strategy management system that organizes code,
manages multiple strategies, and provides tools for optimization
and live trading implementation.
"""

import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyMetadata:
    """Metadata for trading strategies."""
    name: str
    description: str
    author: str
    version: str
    created_date: str
    last_updated: str
    performance_score: float
    risk_level: str
    recommended_capital: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    tags: List[str]


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    price: float
    quantity: float
    confidence: float
    strategy_name: str
    metadata: Dict[str, Any]


class StrategyRegistry:
    """Registry for managing trading strategies."""
    
    def __init__(self):
        self.strategies: Dict[str, Any] = {}
        self.metadata: Dict[str, StrategyMetadata] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
    
    def register_strategy(self, strategy_class: Any, metadata: StrategyMetadata):
        """Register a new strategy."""
        self.strategies[metadata.name] = strategy_class
        self.metadata[metadata.name] = metadata
        logger.info(f"Registered strategy: {metadata.name}")
    
    def get_strategy(self, name: str) -> Any:
        """Get a strategy by name."""
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found")
        return self.strategies[name]
    
    def list_strategies(self) -> List[str]:
        """List all registered strategies."""
        return list(self.strategies.keys())
    
    def get_strategy_info(self, name: str) -> StrategyMetadata:
        """Get strategy metadata."""
        if name not in self.metadata:
            raise ValueError(f"Strategy '{name}' not found")
        return self.metadata[name]
    
    def update_performance(self, name: str, performance: Dict[str, Any]):
        """Update strategy performance history."""
        if name not in self.performance_history:
            self.performance_history[name] = []
        
        performance['timestamp'] = datetime.now().isoformat()
        self.performance_history[name].append(performance)
        
        # Update metadata with latest performance
        if name in self.metadata:
            self.metadata[name].performance_score = performance.get('sharpe_ratio', 0)
            self.metadata[name].max_drawdown = performance.get('max_drawdown', 0)
            self.metadata[name].sharpe_ratio = performance.get('sharpe_ratio', 0)
            self.metadata[name].win_rate = performance.get('win_rate', 0)


class StrategyOptimizer:
    """Advanced strategy optimization using multiple techniques."""
    
    def __init__(self, registry: StrategyRegistry):
        self.registry = registry
        self.optimization_results: Dict[str, Dict] = {}
    
    def optimize_strategy(self, strategy_name: str, data: pd.DataFrame, 
                         optimization_type: str = 'grid_search') -> Dict[str, Any]:
        """Optimize a strategy using various techniques."""
        strategy_class = self.registry.get_strategy(strategy_name)
        
        if optimization_type == 'grid_search':
            return self._grid_search_optimization(strategy_class, data)
        elif optimization_type == 'genetic_algorithm':
            return self._genetic_algorithm_optimization(strategy_class, data)
        elif optimization_type == 'bayesian':
            return self._bayesian_optimization(strategy_class, data)
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
    
    def _grid_search_optimization(self, strategy_class: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Grid search optimization."""
        # Define parameter grids
        param_grids = {
            'slow_ma': [50, 100, 150, 200, 250],
            'entry_threshold': [0.01, 0.015, 0.02, 0.025, 0.03],
            'atr_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0],
            'profit_target_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0]
        }
        
        best_score = -np.inf
        best_params = {}
        
        # Simple grid search implementation
        for slow_ma in param_grids['slow_ma']:
            for entry_threshold in param_grids['entry_threshold']:
                for atr_multiplier in param_grids['atr_multiplier']:
                    for profit_target in param_grids['profit_target_multiplier']:
                        try:
                            # Create strategy with parameters
                            params = {
                                'slow_ma': slow_ma,
                                'entry_threshold': entry_threshold,
                                'atr_multiplier': atr_multiplier,
                                'profit_target_multiplier': profit_target
                            }
                            
                            # Test strategy (simplified)
                            score = self._evaluate_strategy(strategy_class, params, data)
                            
                            if score > best_score:
                                best_score = score
                                best_params = params
                                
                        except Exception as e:
                            logger.warning(f"Error testing parameters {params}: {e}")
                            continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_type': 'grid_search'
        }
    
    def _genetic_algorithm_optimization(self, strategy_class: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Genetic algorithm optimization (simplified)."""
        # Simplified genetic algorithm implementation
        population_size = 20
        generations = 10
        
        # Initialize population
        population = self._initialize_population(population_size)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    score = self._evaluate_strategy(strategy_class, individual, data)
                    fitness_scores.append(score)
                except:
                    fitness_scores.append(-1000)
            
            # Select best individuals
            best_indices = np.argsort(fitness_scores)[-5:]  # Top 5
            best_individuals = [population[i] for i in best_indices]
            
            # Create new generation
            new_population = []
            for _ in range(population_size):
                parent1 = np.random.choice(best_indices)
                parent2 = np.random.choice(best_indices)
                child = self._crossover(population[parent1], population[parent2])
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Return best individual
        final_scores = []
        for individual in population:
            try:
                score = self._evaluate_strategy(strategy_class, individual, data)
                final_scores.append(score)
            except:
                final_scores.append(-1000)
        
        best_idx = np.argmax(final_scores)
        return {
            'best_params': population[best_idx],
            'best_score': final_scores[best_idx],
            'optimization_type': 'genetic_algorithm'
        }
    
    def _bayesian_optimization(self, strategy_class: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Bayesian optimization (simplified)."""
        # Simplified Bayesian optimization
        n_trials = 50
        best_score = -np.inf
        best_params = {}
        
        for trial in range(n_trials):
            # Sample parameters
            params = {
                'slow_ma': np.random.randint(50, 300),
                'entry_threshold': np.random.uniform(0.005, 0.05),
                'atr_multiplier': np.random.uniform(1.5, 5.0),
                'profit_target_multiplier': np.random.uniform(1.5, 5.0)
            }
            
            try:
                score = self._evaluate_strategy(strategy_class, params, data)
                if score > best_score:
                    best_score = score
                    best_params = params
            except:
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_type': 'bayesian'
        }
    
    def _initialize_population(self, size: int) -> List[Dict]:
        """Initialize population for genetic algorithm."""
        population = []
        for _ in range(size):
            individual = {
                'slow_ma': np.random.randint(50, 300),
                'entry_threshold': np.random.uniform(0.005, 0.05),
                'atr_multiplier': np.random.uniform(1.5, 5.0),
                'profit_target_multiplier': np.random.uniform(1.5, 5.0)
            }
            population.append(individual)
        return population
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover operation for genetic algorithm."""
        child = {}
        for key in parent1.keys():
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def _mutate(self, individual: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        for key in mutated.keys():
            if np.random.random() < mutation_rate:
                if key == 'slow_ma':
                    mutated[key] = np.random.randint(50, 300)
                else:
                    mutated[key] = mutated[key] * np.random.uniform(0.9, 1.1)
        return mutated
    
    def _evaluate_strategy(self, strategy_class: Any, params: Dict, data: pd.DataFrame) -> float:
        """Evaluate strategy performance (simplified)."""
        # This is a simplified evaluation - in practice, you'd run full backtesting
        try:
            # Simple scoring based on parameter ranges
            score = 0
            
            # Prefer moderate parameters
            if 100 <= params['slow_ma'] <= 200:
                score += 1
            if 0.01 <= params['entry_threshold'] <= 0.02:
                score += 1
            if 2.0 <= params['atr_multiplier'] <= 3.0:
                score += 1
            if 2.0 <= params['profit_target_multiplier'] <= 3.0:
                score += 1
            
            return score
        except:
            return -1000


class PortfolioManager:
    """Manage portfolio of multiple strategies."""
    
    def __init__(self, registry: StrategyRegistry):
        self.registry = registry
        self.portfolio_weights: Dict[str, float] = {}
        self.portfolio_performance: Dict[str, Any] = {}
    
    def set_portfolio_weights(self, weights: Dict[str, float]):
        """Set portfolio weights for strategies."""
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError("Portfolio weights must sum to 1.0")
        
        self.portfolio_weights = weights
        logger.info(f"Set portfolio weights: {weights}")
    
    def optimize_portfolio_weights(self, data: pd.DataFrame, 
                                 optimization_method: str = 'equal_weight') -> Dict[str, float]:
        """Optimize portfolio weights."""
        if optimization_method == 'equal_weight':
            strategies = list(self.registry.list_strategies())
            weights = {strategy: 1.0 / len(strategies) for strategy in strategies}
            self.portfolio_weights = weights
            return weights
        
        elif optimization_method == 'performance_based':
            # Weight strategies based on their performance
            weights = {}
            total_score = 0
            
            for strategy_name in self.registry.list_strategies():
                metadata = self.registry.get_strategy_info(strategy_name)
                score = metadata.sharpe_ratio
                weights[strategy_name] = max(score, 0)  # Ensure non-negative
                total_score += weights[strategy_name]
            
            # Normalize weights
            if total_score > 0:
                weights = {k: v / total_score for k, v in weights.items()}
            else:
                # Equal weights if no positive scores
                strategies = list(self.registry.list_strategies())
                weights = {strategy: 1.0 / len(strategies) for strategy in strategies}
            
            self.portfolio_weights = weights
            return weights
        
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def calculate_portfolio_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio performance."""
        if not self.portfolio_weights:
            raise ValueError("Portfolio weights not set")
        
        # Simplified portfolio performance calculation
        total_return = 0
        total_sharpe = 0
        total_trades = 0
        
        for strategy_name, weight in self.portfolio_weights.items():
            metadata = self.registry.get_strategy_info(strategy_name)
            total_return += metadata.sharpe_ratio * weight
            total_sharpe += metadata.sharpe_ratio * weight
            total_trades += 100 * weight  # Simplified
        
        portfolio_performance = {
            'total_return': total_return,
            'sharpe_ratio': total_sharpe,
            'num_trades': int(total_trades),
            'weights': self.portfolio_weights
        }
        
        self.portfolio_performance = portfolio_performance
        return portfolio_performance


class StrategyManager:
    """Main strategy management system."""
    
    def __init__(self):
        self.registry = StrategyRegistry()
        self.optimizer = StrategyOptimizer(self.registry)
        self.portfolio_manager = PortfolioManager(self.registry)
        self.config_file = Path("strategy_config.yaml")
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info("Loaded configuration from file")
            except Exception as e:
                logger.warning(f"Error loading config: {e}")
                config = {}
        else:
            config = {}
        
        self.config = config
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info("Saved configuration to file")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def add_strategy(self, strategy_class: Any, metadata: StrategyMetadata):
        """Add a strategy to the manager."""
        self.registry.register_strategy(strategy_class, metadata)
        self.save_config()
    
    def optimize_strategy(self, strategy_name: str, data: pd.DataFrame, 
                         method: str = 'grid_search') -> Dict[str, Any]:
        """Optimize a strategy."""
        result = self.optimizer.optimize_strategy(strategy_name, data, method)
        logger.info(f"Optimized {strategy_name} using {method}")
        return result
    
    def create_portfolio(self, strategy_names: List[str], 
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Create a portfolio of strategies."""
        if weights is None:
            weights = {name: 1.0 / len(strategy_names) for name in strategy_names}
        
        self.portfolio_manager.set_portfolio_weights(weights)
        return self.portfolio_manager.portfolio_weights
    
    def get_strategy_report(self) -> Dict[str, Any]:
        """Generate comprehensive strategy report."""
        report = {
            'total_strategies': len(self.registry.list_strategies()),
            'strategies': {},
            'portfolio_performance': self.portfolio_manager.portfolio_performance,
            'recommendations': []
        }
        
        for strategy_name in self.registry.list_strategies():
            metadata = self.registry.get_strategy_info(strategy_name)
            report['strategies'][strategy_name] = {
                'performance_score': metadata.performance_score,
                'risk_level': metadata.risk_level,
                'sharpe_ratio': metadata.sharpe_ratio,
                'max_drawdown': metadata.max_drawdown,
                'win_rate': metadata.win_rate
            }
        
        # Generate recommendations
        best_strategy = max(report['strategies'].items(), 
                          key=lambda x: x[1]['sharpe_ratio'])
        report['recommendations'].append(f"Best performing strategy: {best_strategy[0]}")
        
        low_risk_strategies = [name for name, info in report['strategies'].items() 
                              if info['risk_level'] == 'LOW']
        if low_risk_strategies:
            report['recommendations'].append(f"Low risk strategies: {', '.join(low_risk_strategies)}")
        
        return report


def create_sample_strategies():
    """Create sample strategies for demonstration."""
    from best_strategy import SimpleTrendFollowingStrategyV2
    
    # Create strategy manager
    manager = StrategyManager()
    
    # Add Simple Trend Following V2 strategy
    metadata = StrategyMetadata(
        name="Simple Trend Following V2",
        description="Best performing trend following strategy",
        author="Trading Bot",
        version="1.0",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
        performance_score=2.268,
        risk_level="MODERATE",
        recommended_capital=10000.0,
        max_drawdown=-0.1692,
        sharpe_ratio=2.268,
        win_rate=0.464,
        tags=["trend_following", "long_term", "proven"]
    )
    
    manager.add_strategy(SimpleTrendFollowingStrategyV2, metadata)
    
    return manager


def run_strategy_manager_demo():
    """Demonstrate the strategy manager."""
    print("🎯 Strategy Manager Demo")
    print("=" * 50)
    
    # Create strategy manager
    manager = create_sample_strategies()
    
    # List strategies
    strategies = manager.registry.list_strategies()
    print(f"📊 Registered strategies: {strategies}")
    
    # Get strategy info
    for strategy_name in strategies:
        info = manager.registry.get_strategy_info(strategy_name)
        print(f"\n{strategy_name}:")
        print(f"  Description: {info.description}")
        print(f"  Performance Score: {info.performance_score}")
        print(f"  Risk Level: {info.risk_level}")
        print(f"  Sharpe Ratio: {info.sharpe_ratio}")
        print(f"  Max Drawdown: {info.max_drawdown*100:.2f}%")
    
    # Create portfolio
    portfolio = manager.create_portfolio(strategies)
    print(f"\n🎯 Portfolio weights: {portfolio}")
    
    # Generate report
    report = manager.get_strategy_report()
    print(f"\n📈 Strategy Report:")
    print(f"  Total strategies: {report['total_strategies']}")
    print(f"  Recommendations: {report['recommendations']}")
    
    print("\n✅ Strategy Manager demo complete!")


if __name__ == "__main__":
    run_strategy_manager_demo()

