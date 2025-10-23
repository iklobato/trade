# Architecture Documentation

## System Overview

This workspace contains two distinct cryptocurrency trading systems designed for different purposes:

1. **ML Trading System**: Modern, production-ready automated trading with ML predictions
2. **Strategy Backtester**: Comprehensive backtesting framework for strategy development

## Architecture Principles

### 1. Separation of Concerns
- Each project has its own namespace and responsibilities
- Shared utilities are centralized in `shared/` module
- Data is managed centrally with clear provenance

### 2. Scalability
- Monorepo structure allows easy addition of new projects
- Shared resources reduce duplication
- Independent CI/CD workflows per project

### 3. Maintainability
- Clear directory structure
- Comprehensive documentation
- Consistent coding standards

## Project Architecture

### ML Trading System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Feature Layer  │    │   Model Layer   │
│                 │    │                 │    │                 │
│ • Polygon.io    │───▶│ • Technical     │───▶│ • XGBoost       │
│ • Data Cache    │    │   Indicators    │    │ • Training      │
│ • OHLCV Data    │    │ • Causal Design │    │ • Prediction   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Execution Layer │    │ Backtest Layer  │    │ Orchestration  │
│                 │    │                 │    │                 │
│ • CCXT Gateway  │    │ • Simulation    │    │ • APScheduler   │
│ • Order Mgmt    │    │ • Risk Mgmt     │    │ • Health Check  │
│ • Position Mgmt│    │ • Performance   │    │ • Recovery     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Strategy Backtester Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │ Strategy Layer  │    │ Analysis Layer  │
│                 │    │                 │    │                 │
│ • CSV Files     │───▶│ • Trend Follow  │───▶│ • Performance  │
│ • OHLCV Data    │    │ • Mean Revert   │    │ • Visualization │
│ • Data Utils    │    │ • ML Strategies │    │ • Reporting     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Backtest Layer  │    │ Optimization    │    │ Management      │
│                 │    │                 │    │                 │
│ • Simulation    │    │ • Parameter Opt │    │ • Strategy Mgmt │
│ • Risk Mgmt     │    │ • Bayesian Opt   │    │ • Configuration │
│ • Performance   │    │ • Walk Forward   │    │ • Testing       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Data Flow Architecture

### Centralized Data Management

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                           │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Raw Data      │ Processed Data  │      Cache              │
│                 │                 │                         │
│ • CSV Files     │ • Clean Data    │ • Model Cache           │
│ • Historical    │ • Features      │ • API Cache             │
│ • External APIs │ • Labels        │ • Temp Data             │
└─────────────────┴─────────────────┴─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Shared Utilities                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Data Utils     │ Trading Utils   │    Config              │
│                 │                 │                         │
│ • Load/Process  │ • Calculations  │ • Exchange Configs     │
│ • Resampling    │ • Risk Mgmt     │ • Trading Params       │
│ • Validation    │ • Performance   │ • Common Settings      │
└─────────────────┴─────────────────┴─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Projects                               │
├─────────────────┬─────────────────┬─────────────────────────┤
│ ML Trading      │ Strategy Backt. │    Future Projects      │
│                 │                 │                         │
│ • Real-time     │ • Backtesting   │ • Portfolio Manager     │
│ • 24/7 Operation│ • Optimization  │ • Risk Manager          │
│ • Live Trading  │ • Analysis      │ • Data Pipeline         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Technology Stack

### ML Trading System
- **Language**: Python 3.8+
- **ML Framework**: XGBoost, scikit-learn
- **Data**: Polygon.io API, pandas
- **Execution**: CCXT (Kraken)
- **Scheduling**: APScheduler
- **Logging**: loguru
- **Configuration**: YAML

### Strategy Backtester
- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **Data**: CSV files, pandas
- **Visualization**: matplotlib, seaborn
- **Optimization**: scipy, optuna
- **Testing**: pytest

### Shared Infrastructure
- **Package Management**: uv
- **Code Quality**: black, ruff, mypy
- **Testing**: pytest
- **Documentation**: Markdown
- **Version Control**: Git

## Security Architecture

### API Key Management
```
┌─────────────────────────────────────────────────────────────┐
│                    Environment Variables                     │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Polygon.io    │     Kraken      │    Other APIs           │
│                 │                 │                         │
│ • POLYGON_API_KEY│ • KRAKEN_API_KEY│ • Additional Keys       │
│ • Rate Limits   │ • KRAKEN_SECRET │ • Future Integrations   │
└─────────────────┴─────────────────┴─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│ ML Trading      │ Strategy Backt. │    Shared Config        │
│                 │                 │                         │
│ • Secure Load   │ • Local Data    │ • Exchange Configs      │
│ • Validation    │ • No External  │ • Common Settings       │
│ • Error Handle  │ • Safe Testing  │ • Default Values        │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Data Security
- **Sensitive Data**: Excluded from version control
- **API Keys**: Environment variables only
- **Logs**: Sanitized, no sensitive information
- **Cache**: Local storage, encrypted if needed

## Deployment Architecture

### Development Environment
```
┌─────────────────────────────────────────────────────────────┐
│                    Development Setup                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Local Dev     │   Testing       │    CI/CD                │
│                 │                 │                         │
│ • uv sync       │ • pytest        │ • GitHub Actions        │
│ • Local config  │ • Coverage      │ • Automated Testing     │
│ • Debug mode    │ • Mock data     │ • Code Quality Checks   │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Production Environment
```
┌─────────────────────────────────────────────────────────────┐
│                    Production Deployment                     │
├─────────────────┬─────────────────┬─────────────────────────┤
│   ML Trading    │   Monitoring    │    Maintenance          │
│                 │                 │                         │
│ • systemd       │ • Health Checks │ • Log Rotation          │
│ • Paper Trading │ • Performance   │ • Model Updates         │
│ • Live Trading  │ • Alerts        │ • System Updates        │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Performance Considerations

### ML Trading System
- **Prediction Frequency**: Every 15 minutes
- **Health Checks**: Every 5 minutes
- **Model Retraining**: Daily
- **Data Updates**: Hourly
- **Memory Usage**: Optimized for 24/7 operation

### Strategy Backtester
- **Backtesting Speed**: Optimized for large datasets
- **Memory Management**: Efficient data processing
- **Parallel Processing**: Multi-core optimization
- **Caching**: Intelligent data caching

## Monitoring and Observability

### System Health Monitoring
```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Application   │   System        │    Business             │
│                 │                 │                         │
│ • Health Checks │ • CPU/Memory    │ • Trading Performance   │
│ • Error Rates   │ • Disk Usage    │ • Risk Metrics          │
│ • Response Time │ • Network       │ • P&L Tracking          │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Logging Strategy
- **Structured Logging**: JSON format for machine parsing
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Rotation**: Daily rotation with compression
- **Retention**: 30 days for operational logs
- **Alerting**: Critical errors and performance issues

## Future Architecture Considerations

### Scalability
- **Microservices**: Potential split into independent services
- **Containerization**: Docker for consistent deployments
- **Orchestration**: Kubernetes for production scaling
- **Message Queues**: Redis/RabbitMQ for async processing

### Data Pipeline
- **Stream Processing**: Real-time data processing
- **Data Lake**: Centralized data storage
- **ETL Pipelines**: Automated data transformation
- **Data Quality**: Automated validation and monitoring

### Integration
- **API Gateway**: Centralized API management
- **Service Mesh**: Inter-service communication
- **Event Streaming**: Kafka for event-driven architecture
- **External Integrations**: Additional exchanges and data sources

## Design Decisions

### Monorepo vs Multi-repo
**Decision**: Monorepo
**Rationale**: 
- Shared utilities and data
- Easier dependency management
- Unified CI/CD
- Simplified development workflow

### Technology Choices
**ML Framework**: XGBoost
**Rationale**: 
- Excellent performance on tabular data
- Built-in feature importance
- Robust to overfitting
- Production-ready

**Execution Framework**: CCXT
**Rationale**: 
- Unified API across exchanges
- Mature and well-tested
- Active community
- Comprehensive exchange support

### Data Management
**Decision**: Centralized data directory
**Rationale**: 
- Clear data provenance
- Reduced duplication
- Easier backup and management
- Shared access across projects

---

This architecture provides a solid foundation for both current projects and future expansion while maintaining clear separation of concerns and scalability.
