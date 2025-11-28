"""
Monte Carlo Simulation Engine - Portfolio Optimization
Advanced implementation with modern Python 3.12 features and vectorized operations
"""

from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime
import json


@dataclass
class PortfolioResult:
    """Container for individual portfolio simulation result"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'weights': self.weights.tolist(),
            'expected_return': float(self.expected_return),
            'volatility': float(self.volatility),
            'sharpe_ratio': float(self.sharpe_ratio)
        }


@dataclass
class SimulationResults:
    """Container for all simulation results"""
    all_weights: np.ndarray
    all_returns: np.ndarray
    all_volatilities: np.ndarray
    all_sharpe_ratios: np.ndarray
    optimal_portfolio: PortfolioResult
    min_volatility_portfolio: PortfolioResult
    max_sharpe_portfolio: PortfolioResult
    num_simulations: int
    tickers: list[str]
    execution_time: float
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics of all simulations"""
        return {
            'mean_return': float(np.mean(self.all_returns)),
            'std_return': float(np.std(self.all_returns)),
            'mean_volatility': float(np.mean(self.all_volatilities)),
            'std_volatility': float(np.std(self.all_volatilities)),
            'mean_sharpe': float(np.mean(self.all_sharpe_ratios)),
            'max_sharpe': float(np.max(self.all_sharpe_ratios)),
            'min_sharpe': float(np.min(self.all_sharpe_ratios))
        }


@dataclass
class PortfolioPath:
    """Container for portfolio value over time simulation"""
    paths: np.ndarray  # Shape: (time_steps, num_simulations)
    timestamps: np.ndarray
    percentiles: dict[int, np.ndarray]
    statistics: dict[str, float]
    
    def get_final_values(self) -> np.ndarray:
        """Get final portfolio values across all simulations"""
        return self.paths[-1, :]


class MonteCarloSimulator:
    """
    Advanced Monte Carlo Portfolio Simulator
    
    Features:
    - Vectorized operations for performance
    - Multiple optimization objectives
    - Cholesky decomposition for correlated returns
    - Portfolio path simulation over time
    - Comprehensive risk metrics
    """
    
    def __init__(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        num_simulations: int = 10000,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initialize Monte Carlo Simulator
        
        Args:
            mean_returns: Mean daily returns for each asset
            cov_matrix: Covariance matrix of returns
            risk_free_rate: Annual risk-free rate (default 2%)
            num_simulations: Number of portfolio simulations
            random_seed: Seed for reproducibility
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.num_simulations = num_simulations
        self.tickers = list(mean_returns.index)
        self.num_assets = len(self.tickers)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Results storage
        self.results: Optional[SimulationResults] = None
        
        logger.info(f"Initialized Monte Carlo Simulator")
        logger.info(f"Assets: {self.num_assets}, Simulations: {self.num_simulations:,}")
    
    def generate_random_portfolios(self) -> SimulationResults:
        """
        Generate random portfolio allocations and calculate metrics
        
        Returns:
            SimulationResults object containing all simulation data
        """
        logger.info(f"Starting Monte Carlo simulation with {self.num_simulations:,} iterations...")
        start_time = datetime.now()
        
        # Pre-allocate arrays for efficiency
        all_weights = np.zeros((self.num_simulations, self.num_assets))
        all_returns = np.zeros(self.num_simulations)
        all_volatilities = np.zeros(self.num_simulations)
        all_sharpe_ratios = np.zeros(self.num_simulations)
        
        # Vectorized generation of random portfolios
        for i in range(self.num_simulations):
            # Generate random weights using Dirichlet distribution
            # This ensures weights are positive and sum to 1
            weights = np.random.dirichlet(np.ones(self.num_assets))
            
            # Calculate portfolio metrics
            portfolio_return, portfolio_vol = self._calculate_portfolio_metrics(weights)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_return, portfolio_vol)
            
            # Store results
            all_weights[i] = weights
            all_returns[i] = portfolio_return
            all_volatilities[i] = portfolio_vol
            all_sharpe_ratios[i] = sharpe_ratio
            
            # Progress logging
            if (i + 1) % 2000 == 0:
                logger.info(f"  Progress: {i+1:,}/{self.num_simulations:,} ({(i+1)/self.num_simulations*100:.1f}%)")
        
        # Find optimal portfolios
        max_sharpe_idx = np.argmax(all_sharpe_ratios)
        min_vol_idx = np.argmin(all_volatilities)
        
        # Create portfolio result objects
        optimal_portfolio = PortfolioResult(
            weights=all_weights[max_sharpe_idx],
            expected_return=all_returns[max_sharpe_idx],
            volatility=all_volatilities[max_sharpe_idx],
            sharpe_ratio=all_sharpe_ratios[max_sharpe_idx]
        )
        
        min_vol_portfolio = PortfolioResult(
            weights=all_weights[min_vol_idx],
            expected_return=all_returns[min_vol_idx],
            volatility=all_volatilities[min_vol_idx],
            sharpe_ratio=all_sharpe_ratios[min_vol_idx]
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create results object
        self.results = SimulationResults(
            all_weights=all_weights,
            all_returns=all_returns,
            all_volatilities=all_volatilities,
            all_sharpe_ratios=all_sharpe_ratios,
            optimal_portfolio=optimal_portfolio,
            min_volatility_portfolio=min_vol_portfolio,
            max_sharpe_portfolio=optimal_portfolio,  # Same as optimal
            num_simulations=self.num_simulations,
            tickers=self.tickers,
            execution_time=execution_time
        )
        
        logger.success(f"✓ Simulation complete in {execution_time:.2f} seconds")
        self._display_optimal_portfolios()
        
        return self.results
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio return and volatility
        
        Args:
            weights: Portfolio weights array
            
        Returns:
            Tuple of (annualized_return, annualized_volatility)
        """
        # Portfolio return (annualized)
        portfolio_return = np.sum(weights * self.mean_returns) * 252
        
        # Portfolio volatility (annualized)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)
        
        return portfolio_return, portfolio_volatility
    
    def _calculate_sharpe_ratio(self, portfolio_return: float, portfolio_vol: float) -> float:
        """Calculate Sharpe ratio"""
        if portfolio_vol == 0:
            return 0.0
        return (portfolio_return - self.risk_free_rate) / portfolio_vol
    
    def simulate_portfolio_paths(
        self,
        weights: np.ndarray,
        initial_investment: float = 100000,
        time_horizon: int = 252,
        num_paths: int = 1000
    ) -> PortfolioPath:
        """
        Simulate portfolio value over time using Geometric Brownian Motion
        
        Args:
            weights: Portfolio weights
            initial_investment: Starting capital
            time_horizon: Number of trading days to simulate
            num_paths: Number of simulation paths
            
        Returns:
            PortfolioPath object with simulation results
        """
        logger.info(f"Simulating {num_paths:,} portfolio paths over {time_horizon} days...")
        
        # Calculate portfolio statistics
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Cholesky decomposition for correlated random variables
        try:
            L = np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix not positive definite, using eigenvalue method")
            L = self._modified_cholesky(self.cov_matrix)
        
        # Pre-allocate paths array
        paths = np.zeros((time_horizon, num_paths))
        paths[0, :] = initial_investment
        
        # Generate correlated random returns
        for t in range(1, time_horizon):
            # Generate uncorrelated random numbers
            Z = np.random.standard_normal((self.num_assets, num_paths))
            
            # Create correlated returns using Cholesky decomposition
            correlated_returns = np.dot(L, Z)
            
            # Calculate portfolio returns for this time step
            daily_returns = self.mean_returns.values.reshape(-1, 1) + correlated_returns
            portfolio_daily_returns = np.dot(weights, daily_returns)
            
            # Update portfolio values
            paths[t, :] = paths[t-1, :] * (1 + portfolio_daily_returns)
        
        # Calculate percentiles
        percentiles = {
            5: np.percentile(paths, 5, axis=1),
            25: np.percentile(paths, 25, axis=1),
            50: np.percentile(paths, 50, axis=1),
            75: np.percentile(paths, 75, axis=1),
            95: np.percentile(paths, 95, axis=1)
        }
        
        # Calculate statistics
        final_values = paths[-1, :]
        statistics = {
            'mean_final_value': float(np.mean(final_values)),
            'median_final_value': float(np.median(final_values)),
            'std_final_value': float(np.std(final_values)),
            'min_final_value': float(np.min(final_values)),
            'max_final_value': float(np.max(final_values)),
            'probability_of_loss': float(np.sum(final_values < initial_investment) / num_paths),
            'expected_return': float((np.mean(final_values) - initial_investment) / initial_investment)
        }
        
        portfolio_path = PortfolioPath(
            paths=paths,
            timestamps=np.arange(time_horizon),
            percentiles=percentiles,
            statistics=statistics
        )
        
        logger.success(f"✓ Portfolio paths simulated successfully")
        self._display_path_statistics(portfolio_path, initial_investment)
        
        return portfolio_path
    
    def _modified_cholesky(self, matrix: pd.DataFrame) -> np.ndarray:
        """Modified Cholesky decomposition for near-positive-definite matrices"""
        # Add small positive value to diagonal
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    def calculate_efficient_frontier(
        self,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the efficient frontier
        
        Args:
            num_points: Number of points on the frontier
            
        Returns:
            Tuple of (returns, volatilities, sharpe_ratios)
        """
        logger.info(f"Calculating efficient frontier with {num_points} points...")
        
        if self.results is None:
            raise ValueError("Run generate_random_portfolios() first")
        
        # Sort by volatility
        sorted_indices = np.argsort(self.results.all_volatilities)
        
        # Create bins for volatility ranges
        vol_bins = np.linspace(
            self.results.all_volatilities.min(),
            self.results.all_volatilities.max(),
            num_points
        )
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpes = []
        
        for i in range(len(vol_bins) - 1):
            # Find portfolios in this volatility range
            mask = (self.results.all_volatilities >= vol_bins[i]) & \
                   (self.results.all_volatilities < vol_bins[i+1])
            
            if np.any(mask):
                # Get portfolio with max return in this vol range
                max_return_idx = np.argmax(self.results.all_returns[mask])
                indices = np.where(mask)[0]
                idx = indices[max_return_idx]
                
                frontier_returns.append(self.results.all_returns[idx])
                frontier_volatilities.append(self.results.all_volatilities[idx])
                frontier_sharpes.append(self.results.all_sharpe_ratios[idx])
        
        logger.success(f"✓ Efficient frontier calculated with {len(frontier_returns)} points")
        
        return (
            np.array(frontier_returns),
            np.array(frontier_volatilities),
            np.array(frontier_sharpes)
        )
    
    def _display_optimal_portfolios(self) -> None:
        """Display optimal portfolio allocations"""
        if self.results is None:
            return
        
        print("\n" + "="*80)
        print("OPTIMAL PORTFOLIOS")
        print("="*80)
        
        print("\n🏆 Maximum Sharpe Ratio Portfolio:")
        print(f"  Expected Return: {self.results.optimal_portfolio.expected_return*100:>8.2f}%")
        print(f"  Volatility:      {self.results.optimal_portfolio.volatility*100:>8.2f}%")
        print(f"  Sharpe Ratio:    {self.results.optimal_portfolio.sharpe_ratio:>8.3f}")
        print("\n  Asset Allocation:")
        for ticker, weight in zip(self.tickers, self.results.optimal_portfolio.weights):
            print(f"    {ticker:10s}: {weight*100:>6.2f}%")
        
        print("\n📉 Minimum Volatility Portfolio:")
        print(f"  Expected Return: {self.results.min_volatility_portfolio.expected_return*100:>8.2f}%")
        print(f"  Volatility:      {self.results.min_volatility_portfolio.volatility*100:>8.2f}%")
        print(f"  Sharpe Ratio:    {self.results.min_volatility_portfolio.sharpe_ratio:>8.3f}")
        print("\n  Asset Allocation:")
        for ticker, weight in zip(self.tickers, self.results.min_volatility_portfolio.weights):
            print(f"    {ticker:10s}: {weight*100:>6.2f}%")
        
        print("="*80)
    
    def _display_path_statistics(self, path: PortfolioPath, initial: float) -> None:
        """Display portfolio path statistics"""
        stats = path.statistics
        
        print("\n" + "="*80)
        print("PORTFOLIO PATH SIMULATION STATISTICS")
        print("="*80)
        print(f"Initial Investment:     ${initial:,.2f}")
        print(f"Mean Final Value:       ${stats['mean_final_value']:,.2f}")
        print(f"Median Final Value:     ${stats['median_final_value']:,.2f}")
        print(f"Expected Return:        {stats['expected_return']*100:.2f}%")
        print(f"Standard Deviation:     ${stats['std_final_value']:,.2f}")
        print(f"Best Case:              ${stats['max_final_value']:,.2f}")
        print(f"Worst Case:             ${stats['min_final_value']:,.2f}")
        print(f"Probability of Loss:    {stats['probability_of_loss']*100:.2f}%")
        print("="*80)
    
    def export_results(self, output_dir: str = 'outputs/monte_carlo') -> None:
        """Export simulation results to files"""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.results is None:
            logger.warning("No results to export")
            return
        
        # Export all simulations
        simulations_df = pd.DataFrame({
            'Return': self.results.all_returns,
            'Volatility': self.results.all_volatilities,
            'Sharpe_Ratio': self.results.all_sharpe_ratios
        })
        
        # Add weight columns
        for i, ticker in enumerate(self.tickers):
            simulations_df[f'Weight_{ticker}'] = self.results.all_weights[:, i]
        
        simulations_df.to_csv(output_path / 'all_simulations.csv', index=False)
        
        # Export optimal portfolios
        optimal_data = {
            'max_sharpe': self.results.optimal_portfolio.to_dict(),
            'min_volatility': self.results.min_volatility_portfolio.to_dict(),
            'summary_stats': self.results.get_summary_stats(),
            'execution_time_seconds': self.results.execution_time
        }
        
        with open(output_path / 'optimal_portfolios.json', 'w') as f:
            json.dump(optimal_data, f, indent=2)
        
        logger.success(f"✓ Results exported to {output_dir}/")


# Testing and example usage
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from datetime import datetime, timedelta
    
    logger.add("data/logs/monte_carlo.log", rotation="10 MB")
    
    print("\n" + "="*80)
    print("TESTING MONTE CARLO ENGINE - LATEST TECH STACK 2025")
    print("="*80)
    
    # Fetch data first
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    fetcher = DataFetcher(tickers, start_date, end_date)
    market_data = fetcher.fetch_historical_data(use_cache=True)
    stats = fetcher.calculate_statistics()
    
    # Initialize Monte Carlo Simulator
    simulator = MonteCarloSimulator(
        mean_returns=stats.mean_returns,
        cov_matrix=stats.cov_matrix,
        risk_free_rate=0.02,
        num_simulations=10000,
        random_seed=42
    )
    
    # Run simulation
    results = simulator.generate_random_portfolios()
    
    # Calculate efficient frontier
    frontier = simulator.calculate_efficient_frontier(num_points=50)
    
    # Simulate portfolio paths
    portfolio_path = simulator.simulate_portfolio_paths(
        weights=results.optimal_portfolio.weights,
        initial_investment=100000,
        time_horizon=252,
        num_paths=1000
    )
    
    # Export results
    simulator.export_results()
    
    print("\n" + "="*80)
    print("✅ MONTE CARLO ENGINE TEST SUCCESSFUL!")
    print("="*80)
