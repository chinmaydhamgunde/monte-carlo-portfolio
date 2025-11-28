**Advanced portfolio optimization using Monte Carlo simulation for data-driven investment decisions**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## 🎯 What It Does

This project implements a **professional-grade portfolio optimizer** that:
- Runs **10,000+ Monte Carlo simulations** to find optimal asset allocation
- Calculates **comprehensive risk metrics** (VaR, CVaR, Maximum Drawdown)
- **Stress tests** portfolios against historical market crises
- Generates **6 publication-quality visualizations**
- Provides **actionable investment insights** backed by quantitative analysis

**Built with modern Python 3.12, type hints, and production-ready code architecture.**

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎲 **Monte Carlo Engine** | 10,000+ simulations with efficient frontier optimization |
| 📊 **Risk Analytics** | VaR (95%, 99%), CVaR, Sortino Ratio, Maximum Drawdown |
| ⚠️ **Stress Testing** | Tests against 2008 Crisis, COVID-19, Black Monday scenarios |
| 📈 **Smart Optimization** | Maximizes Sharpe ratio for best risk-adjusted returns |
| 🎨 **Pro Visualizations** | 6 high-resolution charts (300 DPI, print-ready) |
| 💾 **Efficient Caching** | Parquet format for instant data reloads |
| 🔧 **Type-Safe Code** | Full type hints with modern Python features |

---

## 🚀 Quick Start

### Installation
```
Clone repository
git clone https://github.com/chinmaydhamgunde/monte-carlo-portfolio
cd monte-carlo-portfolio
```

Setup environment
```
python -m venv venv
source venv/Scripts/activate # Windows: Git Bash
pip install -r requirements.txt

```

### Run Demo

Interactive demo (5 minutes)
```
python demo.py
```
Full analysis with default portfolio
```
python main.py
```
Custom portfolio
```
python main.py --tickers AAPL GOOGL MSFT TSLA --investment 250000
```

---

## 💡 Sample Output
```
🏆 OPTIMAL PORTFOLIO FOUND:
Expected Return: 42.5%
Volatility: 18.2%
Sharpe Ratio: 2.336

📊 Asset Allocation:
AAPL ████████████ 25.3%
GOOGL ██████████████████ 35.7%
MSFT ████████████████ 32.1%
AMZN ███ 6.9%

💰 Performance (1 Year Horizon):
Initial Investment: $100,000
Expected Value: $142,500
Median Outcome: $138,200
Probability of Loss: 3.7%

📉 Risk Metrics:
VaR (95%): -$8,450
CVaR (95%): -$12,230
Max Drawdown: -15.7%

⚠️ Stress Test Results:
2008 Financial Crisis: -38.2%
COVID-19 Crash (2020): -44.6%
Black Monday (1987): -49.1%

```

---

## 🏗️ Project Architecture
```
monte-carlo-portfolio/
├── src/
│ ├── data_fetcher.py # Market data retrieval & caching
│ ├── monte_carlo_engine.py # Portfolio simulation engine
│ ├── risk_calculator.py # Risk metrics & stress testing
│ └── visualizer.py # Chart generation
├── outputs/
│ ├── charts/ # Generated visualizations
│ ├── reports/ # JSON/CSV analysis reports
│ └── monte_carlo/ # Simulation results
├── data/
│ ├── raw/ # Cached market data (.parquet)
│ └── processed/ # Calculated statistics
├── main.py # Complete application
├── demo.py # Interactive demo
└── requirements.txt

```


---

## 🛠️ Technical Implementation

### Monte Carlo Simulation
```
Generate 10,000 random portfolios using Dirichlet distribution
for i in range(10000):
weights = np.random.dirichlet(np.ones(num_assets))
portfolio_return = np.sum(weights * mean_returns) * 252
portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol

```

### Risk Calculation
- **VaR (Value at Risk):** Percentile-based loss estimation
- **CVaR (Conditional VaR):** Average of losses beyond VaR threshold
- **Maximum Drawdown:** Peak-to-trough portfolio decline
- **Stress Testing:** Simulates historical crisis scenarios

### Performance Optimization
- Vectorized NumPy operations for speed
- Cholesky decomposition for correlated returns
- Parquet caching (10x faster than CSV)

---

## 📊 Generated Outputs

| File | Description |
|------|-------------|
| `efficient_frontier.png` | Scatter plot of 10,000 portfolios with optimal points |
| `portfolio_distributions.png` | Monte Carlo paths and final value distribution |
| `asset_allocation.png` | Pie chart and bar chart of portfolio weights |
| `risk_metrics.png` | VaR and CVaR comparison visualization |
| `stress_test_results.png` | Historical crisis scenario analysis |
| `drawdown_analysis.png` | Underwater periods and recovery visualization |
| `comprehensive_report.json` | Complete analysis data in JSON format |
| `all_simulations.csv` | Raw data from all 10,000 simulations |

---

## 🎓 Skills Demonstrated

- **Quantitative Finance:** Modern Portfolio Theory, Sharpe Ratio, Efficient Frontier
- **Statistical Modeling:** Monte Carlo Methods, Probability Distributions
- **Risk Management:** VaR, CVaR, Stress Testing, Drawdown Analysis
- **Python Programming:** OOP, Type Hints, Decorators, Context Managers
- **Data Science:** NumPy, Pandas, Statistical Analysis
- **Visualization:** Matplotlib, Seaborn (publication-quality charts)
- **Software Engineering:** Modular Design, Caching, Logging, Error Handling
- **Version Control:** Git with meaningful commits

---

## 📈 Use Cases

### For Individual Investors
- Optimize personal portfolio allocation
- Understand risk-return tradeoffs
- Test portfolio resilience against market crashes

### For Financial Analysts
- Generate client portfolio recommendations
- Perform scenario analysis
- Create professional investment reports

### For Learning & Research
- Study Modern Portfolio Theory implementation
- Experiment with Monte Carlo methods
- Analyze correlation effects on diversification

---

## 🔧 Advanced Usage

### Command-Line Arguments
```
Specify custom tickers
python main.py --tickers TSLA NVDA AMD INTC GS

Set investment amount
python main.py --investment 500000

Increase simulation accuracy
python main.py --simulations 50000

Custom date range
python main.py --start-date 2020-01-01 --end-date 2024-12-31

Adjust risk-free rate
python main.py --risk-free-rate 0.03

```

### Python API
```
from src.data_fetcher import DataFetcher
from src.monte_carlo_engine import MonteCarloSimulator

Fetch data
fetcher = DataFetcher(['AAPL', 'GOOGL', 'MSFT'], '2023-01-01', '2024-12-31')
market_data = fetcher.fetch_historical_data()
stats = fetcher.calculate_statistics()

Run simulation
simulator = MonteCarloSimulator(
mean_returns=stats.mean_returns,
cov_matrix=stats.cov_matrix,
num_simulations=10000
)
results = simulator.generate_random_portfolios()

Access results
optimal = results.optimal_portfolio
print(f"Sharpe Ratio: {optimal.sharpe_ratio:.3f}")
print(f"Allocation: {optimal.weights}")

```

---

## 🏆 Performance Benchmarks

| Operation | Time |
|-----------|------|
| Data Fetch (cached) | < 1 second |
| 10,000 Simulations | ~2 seconds |
| Risk Analysis | ~3 seconds |
| 6 Visualizations | ~5 seconds |
| **Total Pipeline** | **~15 seconds** |

*Tested on: Intel Core i5, 8GB RAM*

---

## 📝 Mathematical Foundation

### Portfolio Return
```
$$R_p = \sum_{i=1}^{n} w_i \times r_i$$
```
### Portfolio Volatility
```
$$\sigma_p = \sqrt{w^T \Sigma w}$$
```
### Sharpe Ratio
```
$$S = \frac{R_p - R_f}{\sigma_p}$$

Where:
- $w$ = portfolio weights
- $r$ = asset returns
- $\Sigma$ = covariance matrix
- $R_f$ = risk-free rate
```
---








## ⭐ Star This Project

If you find this project useful, please consider giving it a star! It helps others discover it.

---

<div align="center">

**Built with ❤️ using Python, NumPy, Pandas & Matplotlib**

*Making investment decisions data-driven, one simulation at a time.*