# config.py
from datetime import datetime, timedelta

# Portfolio Configuration
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Start with 4 stocks
START_DATE = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 years ago
END_DATE = datetime.now().strftime('%Y-%m-%d')  # Today

# Simulation Parameters
NUM_SIMULATIONS = 10000
INITIAL_INVESTMENT = 100000  # $100,000
TIME_HORIZON = 252  # Trading days in 1 year
RISK_FREE_RATE = 0.02  # 2% risk-free rate
CONFIDENCE_LEVEL = 0.95  # For VaR calculation

# Visualization Settings
CHART_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300

print("✓ Configuration loaded successfully")
