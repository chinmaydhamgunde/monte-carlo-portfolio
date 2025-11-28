"""
Monte Carlo Portfolio Optimization - Main Application
Complete integration of all modules for portfolio analysis
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import argparse
import json

import numpy as np
import pandas as pd
from loguru import logger

from src.data_fetcher import DataFetcher
from src.monte_carlo_engine import MonteCarloSimulator
from src.risk_calculator import RiskCalculator
from src.visualizer import PortfolioVisualizer

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("data/logs/main_application.log", rotation="10 MB", level="DEBUG")


class MonteCarloPortfolioAnalyzer:
    """
    Main application for Monte Carlo Portfolio Analysis
    
    Integrates all modules: data fetching, simulation, risk analysis, and visualization
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_investment: float = 100000,
        num_simulations: int = 10000,
        risk_free_rate: float = 0.02,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the portfolio analyzer
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            initial_investment: Initial portfolio value
            num_simulations: Number of Monte Carlo simulations
            risk_free_rate: Annual risk-free rate
            random_seed: Random seed for reproducibility
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_investment = initial_investment
        self.num_simulations = num_simulations
        self.risk_free_rate = risk_free_rate
        self.random_seed = random_seed
        
        # Initialize modules
        self.data_fetcher = None
        self.simulator = None
        self.risk_calculator = None
        self.visualizer = None
        
        # Results storage
        self.market_data = None
        self.statistics = None
        self.monte_carlo_results = None
        self.portfolio_path = None
        self.risk_report = None
        
        logger.info("="*80)
        logger.info("MONTE CARLO PORTFOLIO ANALYZER INITIALIZED")
        logger.info("="*80)
        logger.info(f"Tickers: {', '.join(tickers)}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Investment: ${initial_investment:,.2f}")
        logger.info(f"Simulations: {num_simulations:,}")
        logger.info("="*80)
    
    def run_complete_analysis(self) -> dict:
        """
        Run complete portfolio analysis pipeline
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("\n🚀 Starting Complete Portfolio Analysis Pipeline...\n")
        
        try:
            # Step 1: Fetch and process data
            logger.info("STEP 1/5: Fetching Historical Market Data")
            self._fetch_data()
            
            # Step 2: Run Monte Carlo simulation
            logger.info("\nSTEP 2/5: Running Monte Carlo Simulation")
            self._run_monte_carlo()
            
            # Step 3: Simulate portfolio paths
            logger.info("\nSTEP 3/5: Simulating Portfolio Paths")
            self._simulate_paths()
            
            # Step 4: Calculate risk metrics
            logger.info("\nSTEP 4/5: Calculating Risk Metrics")
            self._calculate_risk()
            
            # Step 5: Generate visualizations
            logger.info("\nSTEP 5/5: Generating Visualizations")
            self._create_visualizations()
            
            # Export comprehensive report
            logger.info("\n📊 Exporting Comprehensive Report")
            self._export_comprehensive_report()
            
            logger.success("\n" + "="*80)
            logger.success("✅ COMPLETE ANALYSIS FINISHED SUCCESSFULLY!")
            logger.success("="*80)
            
            return self._get_summary_results()
            
        except Exception as e:
            logger.exception(f"❌ Analysis failed: {e}")
            raise
    
    def _fetch_data(self) -> None:
        """Fetch and process market data"""
        self.data_fetcher = DataFetcher(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            risk_free_rate=self.risk_free_rate
        )
        
        self.market_data = self.data_fetcher.fetch_historical_data(use_cache=True)
        self.statistics = self.data_fetcher.calculate_statistics()
        self.data_fetcher.export_data()
        
        logger.success("✓ Data fetching complete")
    
    def _run_monte_carlo(self) -> None:
        """Run Monte Carlo simulation"""
        self.simulator = MonteCarloSimulator(
            mean_returns=self.statistics.mean_returns,
            cov_matrix=self.statistics.cov_matrix,
            risk_free_rate=self.risk_free_rate,
            num_simulations=self.num_simulations,
            random_seed=self.random_seed
        )
        
        self.monte_carlo_results = self.simulator.generate_random_portfolios()
        self.simulator.calculate_efficient_frontier(num_points=100)
        self.simulator.export_results()
        
        logger.success("✓ Monte Carlo simulation complete")
    
    def _simulate_paths(self) -> None:
        """Simulate portfolio value paths over time"""
        self.portfolio_path = self.simulator.simulate_portfolio_paths(
            weights=self.monte_carlo_results.optimal_portfolio.weights,
            initial_investment=self.initial_investment,
            time_horizon=252,  # 1 year
            num_paths=1000
        )
        
        logger.success("✓ Portfolio path simulation complete")
    
    def _calculate_risk(self) -> None:
        """Calculate comprehensive risk metrics"""
        # Calculate portfolio returns from market data
        portfolio_returns = (
            self.market_data.returns * 
            self.monte_carlo_results.optimal_portfolio.weights
        ).sum(axis=1)
        
        self.risk_calculator = RiskCalculator(
            portfolio_returns=portfolio_returns,
            portfolio_paths=self.portfolio_path.paths,
            initial_investment=self.initial_investment,
            risk_free_rate=self.risk_free_rate
        )
        
        self.risk_report = self.risk_calculator.generate_comprehensive_report(
            weights=self.monte_carlo_results.optimal_portfolio.weights,
            mean_returns=self.statistics.mean_returns,
            cov_matrix=self.statistics.cov_matrix
        )
        
        self.risk_calculator.export_risk_report(self.risk_report)
        
        logger.success("✓ Risk analysis complete")
    
    def _create_visualizations(self) -> None:
        """Generate all visualizations"""
        self.visualizer = PortfolioVisualizer(
            output_dir='outputs/charts',
            style='professional',
            dpi=300
        )
        
        # 1. Efficient Frontier
        self.visualizer.plot_efficient_frontier(
            all_returns=self.monte_carlo_results.all_returns,
            all_volatilities=self.monte_carlo_results.all_volatilities,
            all_sharpe_ratios=self.monte_carlo_results.all_sharpe_ratios,
            optimal_portfolio=self.monte_carlo_results.optimal_portfolio.to_dict(),
            min_vol_portfolio=self.monte_carlo_results.min_volatility_portfolio.to_dict(),
            tickers=self.tickers
        )
        
        # 2. Portfolio Distributions
        self.visualizer.plot_portfolio_distributions(
            portfolio_paths=self.portfolio_path.paths,
            initial_investment=self.initial_investment,
            percentiles=self.portfolio_path.percentiles
        )
        
        # 3. Asset Allocation
        self.visualizer.plot_asset_allocation(
            weights=self.monte_carlo_results.optimal_portfolio.weights,
            tickers=self.tickers,
            portfolio_name='Optimal Portfolio (Max Sharpe Ratio)'
        )
        
        # 4. Risk Metrics
        self.visualizer.plot_risk_metrics(
            var_95={'value': self.risk_report.var_95.var_value, 
                    'percentage': self.risk_report.var_95.var_percentage},
            var_99={'value': self.risk_report.var_99.var_value, 
                    'percentage': self.risk_report.var_99.var_percentage},
            cvar_95={'value': self.risk_report.cvar_95.cvar_value, 
                     'percentage': self.risk_report.cvar_95.cvar_percentage},
            cvar_99={'value': self.risk_report.cvar_99.cvar_value, 
                     'percentage': self.risk_report.cvar_99.cvar_percentage},
            initial_investment=self.initial_investment
        )
        
        # 5. Stress Test Results
        stress_data = [
            {
                'scenario': st.scenario_name,
                'loss': st.portfolio_loss,
                'loss_percentage': st.portfolio_loss_percentage
            }
            for st in self.risk_report.stress_test_results
        ]
        self.visualizer.plot_stress_test_results(stress_data)
        
        # 6. Drawdown Analysis
        portfolio_values = pd.Series(np.median(self.portfolio_path.paths, axis=1))
        self.visualizer.plot_drawdown_analysis(
            drawdown_series=self.risk_report.drawdown_analysis.drawdown_series,
            portfolio_values=portfolio_values
        )
        
        logger.success("✓ All visualizations created")
    
    def _export_comprehensive_report(self) -> None:
        """Export comprehensive analysis report"""
        report = {
            'analysis_metadata': {
                'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tickers': self.tickers,
                'period': f"{self.start_date} to {self.end_date}",
                'initial_investment': self.initial_investment,
                'num_simulations': self.num_simulations,
                'risk_free_rate': self.risk_free_rate
            },
            'optimal_portfolio': {
                'weights': {
                    ticker: float(weight)
                    for ticker, weight in zip(
                        self.tickers,
                        self.monte_carlo_results.optimal_portfolio.weights
                    )
                },
                'expected_return': float(self.monte_carlo_results.optimal_portfolio.expected_return),
                'volatility': float(self.monte_carlo_results.optimal_portfolio.volatility),
                'sharpe_ratio': float(self.monte_carlo_results.optimal_portfolio.sharpe_ratio)
            },
            'risk_metrics': self.risk_report.to_dict(),
            'performance_summary': {
                'mean_final_value': float(self.portfolio_path.statistics['mean_final_value']),
                'median_final_value': float(self.portfolio_path.statistics['median_final_value']),
                'expected_return': float(self.portfolio_path.statistics['expected_return']),
                'probability_of_loss': float(self.portfolio_path.statistics['probability_of_loss'])
            }
        }
        
        output_path = Path('outputs/reports/comprehensive_analysis_report.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.success(f"✓ Comprehensive report exported: {output_path}")
    
    def _get_summary_results(self) -> dict:
        """Get summary of key results"""
        return {
            'optimal_portfolio': {
                'weights': dict(zip(
                    self.tickers,
                    self.monte_carlo_results.optimal_portfolio.weights
                )),
                'expected_return': self.monte_carlo_results.optimal_portfolio.expected_return,
                'volatility': self.monte_carlo_results.optimal_portfolio.volatility,
                'sharpe_ratio': self.monte_carlo_results.optimal_portfolio.sharpe_ratio
            },
            'performance': {
                'expected_final_value': self.portfolio_path.statistics['mean_final_value'],
                'probability_of_loss': self.portfolio_path.statistics['probability_of_loss']
            },
            'risk': {
                'var_95': self.risk_report.var_95.var_value,
                'cvar_95': self.risk_report.cvar_95.cvar_value,
                'max_drawdown': self.risk_report.drawdown_analysis.max_drawdown
            }
        }


def main():
    """Main entry point with CLI support"""
    parser = argparse.ArgumentParser(
        description='Monte Carlo Portfolio Optimization and Risk Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        help='Stock ticker symbols (default: AAPL GOOGL MSFT AMZN)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD, default: 2 years ago)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD, default: today)'
    )
    
    parser.add_argument(
        '--investment',
        type=float,
        default=100000,
        help='Initial investment amount (default: 100000)'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations (default: 10000)'
    )
    
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.02,
        help='Annual risk-free rate (default: 0.02)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = MonteCarloPortfolioAnalyzer(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_investment=args.investment,
        num_simulations=args.simulations,
        risk_free_rate=args.risk_free_rate
    )
    
    results = analyzer.run_complete_analysis()
    
        # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("\n📊 Optimal Portfolio Allocation:")
    
    # Show ALL assets, not just > 1%
    for ticker, weight in results['optimal_portfolio']['weights'].items():
        # Only filter out truly zero weights
        if weight > 0.001:  # Show anything above 0.1%
            print(f"  {ticker:10s}: {weight*100:>6.2f}%")
    
    print(f"\n📈 Expected Return:  {results['optimal_portfolio']['expected_return']*100:.2f}%")
    print(f"📉 Volatility:       {results['optimal_portfolio']['volatility']*100:.2f}%")
    print(f"⚡ Sharpe Ratio:     {results['optimal_portfolio']['sharpe_ratio']:.3f}")
    print(f"\n💰 Expected Value:   ${results['performance']['expected_final_value']:,.2f}")
    print(f"⚠️  Probability Loss: {results['performance']['probability_of_loss']*100:.2f}%")
    print(f"\n📉 VaR (95%):        ${abs(results['risk']['var_95']):,.2f}")
    print(f"📉 CVaR (95%):       ${abs(results['risk']['cvar_95']):,.2f}")
    print(f"📉 Max Drawdown:     {abs(results['risk']['max_drawdown'])*100:.2f}%")
    print("\n" + "="*80)
    print("📁 All results saved to outputs/ directory")
    print("="*80)



if __name__ == "__main__":
    main()
