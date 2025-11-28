"""
Quick Demo - Monte Carlo Portfolio Optimizer
5-minute demonstration of key features
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add color to terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

def print_step(step_num, text):
    print(f"{Colors.BOLD}{Colors.YELLOW}[STEP {step_num}]{Colors.END} {Colors.BOLD}{text}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def demo():
    """Run quick portfolio optimization demo"""
    
    print_header("MONTE CARLO PORTFOLIO OPTIMIZER - LIVE DEMO")
    
    print(f"{Colors.BOLD}Demo Configuration:{Colors.END}")
    print(f"  • Portfolio: AAPL, GOOGL, MSFT, AMZN")
    print(f"  • Investment: $100,000")
    print(f"  • Simulations: 10,000")
    print(f"  • Time Period: Last 1 year")
    
    input(f"\n{Colors.YELLOW}Press ENTER to start analysis...{Colors.END}")
    
    # Import modules
    from src.data_fetcher import DataFetcher
    from src.monte_carlo_engine import MonteCarloSimulator
    from src.risk_calculator import RiskCalculator
    from src.visualizer import PortfolioVisualizer
    
    # Configuration
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    initial_investment = 100000
    
    # Step 1: Fetch Data
    print_step(1, "Fetching Historical Market Data")
    print_info("Downloading stock prices from Yahoo Finance...")
    
    fetcher = DataFetcher(tickers, start_date, end_date)
    market_data = fetcher.fetch_historical_data(use_cache=True)
    stats = fetcher.calculate_statistics()
    
    print_success("Historical data loaded successfully!")
    print(f"  Trading days: {market_data.trading_days}")
    print(f"  Best performer: {stats.annualized_returns.idxmax()} "
          f"({stats.annualized_returns.max()*100:.2f}% annual return)")
    
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.END}")
    
    # Step 2: Monte Carlo Simulation
    print_step(2, "Running Monte Carlo Simulation")
    print_info("Generating 10,000 random portfolio allocations...")
    
    simulator = MonteCarloSimulator(
        mean_returns=stats.mean_returns,
        cov_matrix=stats.cov_matrix,
        num_simulations=10000,
        random_seed=42
    )
    
    results = simulator.generate_random_portfolios()
    
    print_success("Simulation complete!")
    print(f"\n{Colors.BOLD}🏆 OPTIMAL PORTFOLIO FOUND:{Colors.END}")
    print(f"  Expected Return:  {Colors.GREEN}{results.optimal_portfolio.expected_return*100:>7.2f}%{Colors.END}")
    print(f"  Volatility:       {Colors.YELLOW}{results.optimal_portfolio.volatility*100:>7.2f}%{Colors.END}")
    print(f"  Sharpe Ratio:     {Colors.CYAN}{results.optimal_portfolio.sharpe_ratio:>7.3f}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Asset Allocation:{Colors.END}")
    for ticker, weight in zip(tickers, results.optimal_portfolio.weights):
        if weight > 0.01:
            bar = '█' * int(weight * 50)
            print(f"  {ticker:6s} {bar} {weight*100:>6.2f}%")
    
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.END}")
    
    # Step 3: Risk Analysis
    print_step(3, "Calculating Risk Metrics")
    print_info("Computing VaR, CVaR, and stress tests...")
    
    portfolio_path = simulator.simulate_portfolio_paths(
        weights=results.optimal_portfolio.weights,
        initial_investment=initial_investment,
        time_horizon=252,
        num_paths=1000
    )
    
    portfolio_returns = (market_data.returns * results.optimal_portfolio.weights).sum(axis=1)
    
    risk_calc = RiskCalculator(
        portfolio_returns=portfolio_returns,
        portfolio_paths=portfolio_path.paths,
        initial_investment=initial_investment
    )
    
    risk_report = risk_calc.generate_comprehensive_report(
        weights=results.optimal_portfolio.weights,
        mean_returns=stats.mean_returns,
        cov_matrix=stats.cov_matrix
    )
    
    print_success("Risk analysis complete!")
    print(f"\n{Colors.BOLD}Risk Metrics:{Colors.END}")
    print(f"  VaR (95%):        {Colors.RED}-${abs(risk_report.var_95.var_value):>10,.2f}{Colors.END}")
    print(f"  CVaR (95%):       {Colors.RED}-${abs(risk_report.cvar_95.cvar_value):>10,.2f}{Colors.END}")
    print(f"  Max Drawdown:     {Colors.RED}{abs(risk_report.drawdown_analysis.max_drawdown)*100:>7.2f}%{Colors.END}")
    
    print(f"\n{Colors.BOLD}Worst-Case Scenarios (Stress Tests):{Colors.END}")
    for i, stress in enumerate(risk_report.stress_test_results[:3], 1):
        print(f"  {i}. {stress.scenario_name:25s} "
              f"{Colors.RED}-${abs(stress.portfolio_loss):>10,.2f}{Colors.END} "
              f"({abs(stress.portfolio_loss_percentage)*100:.1f}%)")
    
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.END}")
    
    # Step 4: Visualizations
    print_step(4, "Generating Professional Visualizations")
    print_info("Creating 6 high-resolution charts...")
    
    viz = PortfolioVisualizer(style='professional', dpi=300)
    
    # Create all charts
    charts_created = []
    
    print("  Creating efficient frontier...", end=' ')
    viz.plot_efficient_frontier(
        all_returns=results.all_returns,
        all_volatilities=results.all_volatilities,
        all_sharpe_ratios=results.all_sharpe_ratios,
        optimal_portfolio=results.optimal_portfolio.to_dict(),
        min_vol_portfolio=results.min_volatility_portfolio.to_dict(),
        tickers=tickers
    )
    print(f"{Colors.GREEN}✓{Colors.END}")
    charts_created.append("efficient_frontier.png")
    
    print("  Creating portfolio distributions...", end=' ')
    viz.plot_portfolio_distributions(
        portfolio_paths=portfolio_path.paths,
        initial_investment=initial_investment,
        percentiles=portfolio_path.percentiles
    )
    print(f"{Colors.GREEN}✓{Colors.END}")
    charts_created.append("portfolio_distributions.png")
    
    print("  Creating asset allocation...", end=' ')
    viz.plot_asset_allocation(
        weights=results.optimal_portfolio.weights,
        tickers=tickers
    )
    print(f"{Colors.GREEN}✓{Colors.END}")
    charts_created.append("asset_allocation.png")
    
    print("  Creating risk metrics...", end=' ')
    viz.plot_risk_metrics(
        var_95={'value': risk_report.var_95.var_value, 'percentage': risk_report.var_95.var_percentage},
        var_99={'value': risk_report.var_99.var_value, 'percentage': risk_report.var_99.var_percentage},
        cvar_95={'value': risk_report.cvar_95.cvar_value, 'percentage': risk_report.cvar_95.cvar_percentage},
        cvar_99={'value': risk_report.cvar_99.cvar_value, 'percentage': risk_report.cvar_99.cvar_percentage},
        initial_investment=initial_investment
    )
    print(f"{Colors.GREEN}✓{Colors.END}")
    charts_created.append("risk_metrics.png")
    
    print("  Creating stress test results...", end=' ')
    stress_data = [
        {'scenario': st.scenario_name, 'loss': st.portfolio_loss, 'loss_percentage': st.portfolio_loss_percentage}
        for st in risk_report.stress_test_results
    ]
    viz.plot_stress_test_results(stress_data)
    print(f"{Colors.GREEN}✓{Colors.END}")
    charts_created.append("stress_test_results.png")
    
    print("  Creating drawdown analysis...", end=' ')
    import pandas as pd
    import numpy as np
    portfolio_values = pd.Series(np.median(portfolio_path.paths, axis=1))
    viz.plot_drawdown_analysis(
        drawdown_series=risk_report.drawdown_analysis.drawdown_series,
        portfolio_values=portfolio_values
    )
    print(f"{Colors.GREEN}✓{Colors.END}")
    charts_created.append("drawdown_analysis.png")
    
    print_success("All visualizations created!")
    
    # Summary
    print_header("DEMO COMPLETE - SUMMARY")
    
    print(f"{Colors.BOLD}📊 Analysis Results:{Colors.END}")
    print(f"  Initial Investment:    ${initial_investment:>12,.2f}")
    print(f"  Expected Final Value:  ${portfolio_path.statistics['mean_final_value']:>12,.2f}")
    print(f"  Expected Return:       {portfolio_path.statistics['expected_return']*100:>7.2f}%")
    print(f"  Probability of Loss:   {portfolio_path.statistics['probability_of_loss']*100:>7.2f}%")
    
    print(f"\n{Colors.BOLD}📁 Generated Files:{Colors.END}")
    print(f"  Charts: outputs/charts/ ({len(charts_created)} files)")
    print(f"  Reports: outputs/reports/")
    print(f"  Data: data/processed/")
    
    print(f"\n{Colors.BOLD}🎯 Key Features Demonstrated:{Colors.END}")
    features = [
        "Real-time market data fetching",
        "10,000 Monte Carlo simulations",
        "Portfolio optimization (Sharpe ratio)",
        "Risk metrics (VaR, CVaR, Drawdown)",
        "Stress testing (6 scenarios)",
        "Professional visualizations (6 charts)",
        "Comprehensive reporting"
    ]
    for feature in features:
        print(f"  {Colors.GREEN}✓{Colors.END} {feature}")
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}Thank you for watching the demo!{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    print(f"{Colors.YELLOW}💡 Tip: Open 'outputs/charts/' to view all generated visualizations!{Colors.END}\n")

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo interrupted. Goodbye!{Colors.END}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error during demo: {e}{Colors.END}\n")
        raise
