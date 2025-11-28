"""
Risk Calculator Module - Comprehensive Portfolio Risk Analysis
Advanced risk metrics including VaR, CVaR, drawdowns, and stress testing
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
import json
from pathlib import Path


class RiskMetric(Enum):
    """Enumeration of risk metrics"""
    VAR = "Value at Risk"
    CVAR = "Conditional Value at Risk"
    MAX_DRAWDOWN = "Maximum Drawdown"
    SEMI_DEVIATION = "Semi-Deviation"
    SORTINO_RATIO = "Sortino Ratio"


@dataclass
class VaRResult:
    """Container for Value at Risk calculations"""
    confidence_level: float
    var_value: float
    var_percentage: float
    method: str
    
    def __str__(self) -> str:
        return (f"VaR ({self.confidence_level*100:.0f}% confidence): "
                f"${abs(self.var_value):,.2f} ({abs(self.var_percentage)*100:.2f}%)")


@dataclass
class CVaRResult:
    """Container for Conditional Value at Risk"""
    confidence_level: float
    cvar_value: float
    cvar_percentage: float
    var_value: float
    
    def __str__(self) -> str:
        return (f"CVaR ({self.confidence_level*100:.0f}% confidence): "
                f"${abs(self.cvar_value):,.2f} ({abs(self.cvar_percentage)*100:.2f}%)")


@dataclass
class DrawdownAnalysis:
    """Container for drawdown analysis"""
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    recovery_time: int
    drawdown_series: pd.Series
    
    def __str__(self) -> str:
        return f"Max Drawdown: {abs(self.max_drawdown)*100:.2f}% (Duration: {self.max_drawdown_duration} days)"


@dataclass
class StressTestResult:
    """Container for stress test scenario result"""
    scenario_name: str
    portfolio_loss: float
    portfolio_loss_percentage: float
    new_portfolio_value: float
    scenario_parameters: dict
    
    def __str__(self) -> str:
        return (f"{self.scenario_name}: Loss ${abs(self.portfolio_loss):,.2f} "
                f"({abs(self.portfolio_loss_percentage)*100:.2f}%)")


@dataclass
class ComprehensiveRiskReport:
    """Complete risk analysis report"""
    var_95: VaRResult
    var_99: VaRResult
    cvar_95: CVaRResult
    cvar_99: CVaRResult
    drawdown_analysis: DrawdownAnalysis
    stress_test_results: List[StressTestResult]
    portfolio_statistics: dict
    risk_adjusted_metrics: dict
    
    def to_dict(self) -> dict:
        """Convert report to dictionary"""
        return {
            'var_95': {
                'value': self.var_95.var_value,
                'percentage': self.var_95.var_percentage
            },
            'var_99': {
                'value': self.var_99.var_value,
                'percentage': self.var_99.var_percentage
            },
            'cvar_95': {
                'value': self.cvar_95.cvar_value,
                'percentage': self.cvar_95.cvar_percentage
            },
            'cvar_99': {
                'value': self.cvar_99.cvar_value,
                'percentage': self.cvar_99.cvar_percentage
            },
            'max_drawdown': self.drawdown_analysis.max_drawdown,
            'stress_tests': [
                {
                    'scenario': st.scenario_name,
                    'loss': st.portfolio_loss,
                    'loss_percentage': st.portfolio_loss_percentage
                }
                for st in self.stress_test_results
            ],
            'portfolio_statistics': self.portfolio_statistics,
            'risk_adjusted_metrics': self.risk_adjusted_metrics
        }


class RiskCalculator:
    """
    Advanced Risk Calculator for Portfolio Analysis
    
    Features:
    - Multiple VaR calculation methods (historical, parametric, Monte Carlo)
    - Conditional Value at Risk (Expected Shortfall)
    - Maximum Drawdown analysis
    - Stress testing scenarios
    - Risk-adjusted performance metrics
    - Tail risk analysis
    """
    
    def __init__(
        self,
        portfolio_returns: Optional[pd.Series] = None,
        portfolio_paths: Optional[np.ndarray] = None,
        initial_investment: float = 100000,
        risk_free_rate: float = 0.02
    ) -> None:
        """
        Initialize Risk Calculator
        
        Args:
            portfolio_returns: Series of historical portfolio returns
            portfolio_paths: Array of simulated portfolio paths (time x simulations)
            initial_investment: Initial portfolio value
            risk_free_rate: Annual risk-free rate
        """
        self.portfolio_returns = portfolio_returns
        self.portfolio_paths = portfolio_paths
        self.initial_investment = initial_investment
        self.risk_free_rate = risk_free_rate
        
        logger.info("Initialized Risk Calculator")
    
    def calculate_var(
        self,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> VaRResult:
        """
        Calculate Value at Risk
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaRResult object
        """
        if method == 'historical':
            return self._var_historical(confidence_level)
        elif method == 'parametric':
            return self._var_parametric(confidence_level)
        elif method == 'monte_carlo':
            return self._var_monte_carlo(confidence_level)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _var_historical(self, confidence_level: float) -> VaRResult:
        """Calculate VaR using historical simulation method"""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns required for historical VaR")
        
        # Calculate VaR at given confidence level
        var_percentage = np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
        var_value = self.initial_investment * var_percentage
        
        return VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            var_percentage=var_percentage,
            method='historical'
        )
    
    def _var_parametric(self, confidence_level: float) -> VaRResult:
        """Calculate VaR using parametric (variance-covariance) method"""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns required for parametric VaR")
        
        # Assume normal distribution
        mean_return = self.portfolio_returns.mean()
        std_return = self.portfolio_returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation
        var_percentage = mean_return + z_score * std_return
        var_value = self.initial_investment * var_percentage
        
        return VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            var_percentage=var_percentage,
            method='parametric'
        )
    
    def _var_monte_carlo(self, confidence_level: float) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation"""
        if self.portfolio_paths is None:
            raise ValueError("Portfolio paths required for Monte Carlo VaR")
        
        # Get final values from all simulations
        final_values = self.portfolio_paths[-1, :]
        
        # Calculate returns
        returns = (final_values - self.initial_investment) / self.initial_investment
        
        # VaR at confidence level
        var_percentage = np.percentile(returns, (1 - confidence_level) * 100)
        var_value = self.initial_investment * var_percentage
        
        return VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            var_percentage=var_percentage,
            method='monte_carlo'
        )
    
    def calculate_cvar(
        self,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> CVaRResult:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            confidence_level: Confidence level
            method: Calculation method
            
        Returns:
            CVaRResult object
        """
        # First calculate VaR
        var_result = self.calculate_var(confidence_level, method)
        
        if method == 'monte_carlo' and self.portfolio_paths is not None:
            final_values = self.portfolio_paths[-1, :]
            returns = (final_values - self.initial_investment) / self.initial_investment
        elif self.portfolio_returns is not None:
            returns = self.portfolio_returns
        else:
            raise ValueError("No data available for CVaR calculation")
        
        # CVaR: average of losses beyond VaR threshold
        threshold = var_result.var_percentage
        tail_losses = returns[returns <= threshold]
        
        if len(tail_losses) == 0:
            cvar_percentage = threshold
        else:
            cvar_percentage = tail_losses.mean()
        
        cvar_value = self.initial_investment * cvar_percentage
        
        return CVaRResult(
            confidence_level=confidence_level,
            cvar_value=cvar_value,
            cvar_percentage=cvar_percentage,
            var_value=var_result.var_value
        )
    
    def calculate_maximum_drawdown(
        self,
        portfolio_values: Optional[pd.Series] = None
    ) -> DrawdownAnalysis:
        """
        Calculate maximum drawdown and related metrics
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            DrawdownAnalysis object
        """
        if portfolio_values is None and self.portfolio_paths is not None:
            # Use median path from Monte Carlo simulation
            portfolio_values = pd.Series(np.median(self.portfolio_paths, axis=1))
        elif portfolio_values is None:
            raise ValueError("Portfolio values required for drawdown calculation")
        
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown series
        drawdown = (portfolio_values - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Calculate drawdown duration
        if isinstance(max_dd_idx, (int, np.integer)):
            # Find when drawdown started
            pre_dd = running_max[:max_dd_idx]
            if len(pre_dd) > 0:
                dd_start = pre_dd.idxmax()
                max_drawdown_duration = max_dd_idx - dd_start
            else:
                max_drawdown_duration = max_dd_idx
        else:
            max_drawdown_duration = 0
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0
        
        # Recovery time (days since max drawdown)
        recovery_time = len(drawdown) - max_dd_idx if max_dd_idx < len(drawdown) else 0
        
        return DrawdownAnalysis(
            max_drawdown=max_drawdown,
            max_drawdown_duration=int(max_drawdown_duration),
            current_drawdown=current_drawdown,
            recovery_time=int(recovery_time),
            drawdown_series=drawdown
        )
    
    def calculate_semi_deviation(
        self,
        returns: Optional[pd.Series] = None,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate semi-deviation (downside deviation)
        
        Args:
            returns: Return series
            target_return: Minimum acceptable return (default 0)
            
        Returns:
            Semi-deviation value
        """
        if returns is None:
            returns = self.portfolio_returns
        
        if returns is None:
            raise ValueError("Returns required for semi-deviation calculation")
        
        # Only consider returns below target
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate semi-deviation
        semi_dev = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        return float(semi_dev)
    
    def calculate_sortino_ratio(
        self,
        returns: Optional[pd.Series] = None,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio (risk-adjusted return using downside deviation)
        
        Args:
            returns: Return series
            target_return: Minimum acceptable return
            
        Returns:
            Sortino ratio
        """
        if returns is None:
            returns = self.portfolio_returns
        
        if returns is None:
            raise ValueError("Returns required for Sortino ratio calculation")
        
        # Annualized return
        annualized_return = returns.mean() * 252
        
        # Semi-deviation (annualized)
        semi_dev = self.calculate_semi_deviation(returns, target_return) * np.sqrt(252)
        
        if semi_dev == 0:
            return 0.0
        
        # Sortino ratio
        sortino = (annualized_return - self.risk_free_rate) / semi_dev
        
        return float(sortino)
    
    def stress_test(
        self,
        weights: np.ndarray,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        scenarios: Optional[Dict[str, dict]] = None
    ) -> List[StressTestResult]:
        """
        Perform stress testing under various market scenarios
        
        Args:
            weights: Portfolio weights
            mean_returns: Mean returns of assets
            cov_matrix: Covariance matrix
            scenarios: Dict of scenario definitions
            
        Returns:
            List of StressTestResult objects
        """
        logger.info("Running stress tests...")
        
        # Default stress scenarios
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        results = []
        
        for scenario_name, params in scenarios.items():
            # Apply shocks to returns and volatility
            shocked_returns = mean_returns * params['return_shock']
            shocked_cov = cov_matrix * params['volatility_shock']
            
            # Calculate portfolio return under stress
            portfolio_return = np.sum(weights * shocked_returns) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(shocked_cov * 252, weights)))
            
            # Calculate loss
            expected_value = self.initial_investment * (1 + portfolio_return)
            portfolio_loss = expected_value - self.initial_investment
            loss_percentage = portfolio_loss / self.initial_investment
            
            result = StressTestResult(
                scenario_name=scenario_name,
                portfolio_loss=portfolio_loss,
                portfolio_loss_percentage=loss_percentage,
                new_portfolio_value=expected_value,
                scenario_parameters=params
            )
            
            results.append(result)
        
        logger.success(f"✓ Completed {len(results)} stress test scenarios")
        
        return results
    
    def _get_default_stress_scenarios(self) -> Dict[str, dict]:
        """Get default stress test scenarios based on historical crises"""
        return {
            '2008 Financial Crisis': {
                'return_shock': 0.60,  # 40% decline
                'volatility_shock': 2.5,  # Volatility increases 2.5x
                'description': 'Similar to 2008-2009 financial crisis'
            },
            'COVID-19 Crash (2020)': {
                'return_shock': 0.70,  # 30% decline
                'volatility_shock': 3.0,  # Extreme volatility
                'description': 'March 2020 pandemic crash'
            },
            'Dot-com Bubble (2000)': {
                'return_shock': 0.50,  # 50% decline for tech
                'volatility_shock': 2.0,
                'description': 'Tech bubble burst'
            },
            'Moderate Recession': {
                'return_shock': 0.80,  # 20% decline
                'volatility_shock': 1.5,
                'description': 'Typical recession scenario'
            },
            'Black Monday (1987)': {
                'return_shock': 0.77,  # 23% single-day crash
                'volatility_shock': 4.0,  # Extreme volatility
                'description': 'October 1987 market crash'
            },
            'Stagflation Scenario': {
                'return_shock': 0.85,  # 15% decline
                'volatility_shock': 1.8,
                'description': 'High inflation, low growth'
            }
        }
    
    def generate_comprehensive_report(
        self,
        weights: np.ndarray,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        portfolio_values: Optional[pd.Series] = None
    ) -> ComprehensiveRiskReport:
        """
        Generate comprehensive risk analysis report
        
        Args:
            weights: Portfolio weights
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            portfolio_values: Time series of portfolio values
            
        Returns:
            ComprehensiveRiskReport object
        """
        logger.info("Generating comprehensive risk report...")
        
        # Calculate VaR at different confidence levels
        var_95 = self.calculate_var(0.95, method='monte_carlo' if self.portfolio_paths is not None else 'historical')
        var_99 = self.calculate_var(0.99, method='monte_carlo' if self.portfolio_paths is not None else 'historical')
        
        # Calculate CVaR
        cvar_95 = self.calculate_cvar(0.95, method='monte_carlo' if self.portfolio_paths is not None else 'historical')
        cvar_99 = self.calculate_cvar(0.99, method='monte_carlo' if self.portfolio_paths is not None else 'historical')
        
        # Drawdown analysis
        drawdown = self.calculate_maximum_drawdown(portfolio_values)
        
        # Stress tests
        stress_results = self.stress_test(weights, mean_returns, cov_matrix)
        
        # Portfolio statistics
        if self.portfolio_returns is not None:
            portfolio_stats = {
                'mean_return': float(self.portfolio_returns.mean() * 252),
                'volatility': float(self.portfolio_returns.std() * np.sqrt(252)),
                'skewness': float(self.portfolio_returns.skew()),
                'kurtosis': float(self.portfolio_returns.kurtosis()),
                'best_day': float(self.portfolio_returns.max()),
                'worst_day': float(self.portfolio_returns.min())
            }
        else:
            portfolio_stats = {}
        
        # Risk-adjusted metrics
        risk_adjusted = {
            'sortino_ratio': self.calculate_sortino_ratio() if self.portfolio_returns is not None else 0.0,
            'semi_deviation': self.calculate_semi_deviation() if self.portfolio_returns is not None else 0.0,
            'calmar_ratio': (portfolio_stats.get('mean_return', 0) / abs(drawdown.max_drawdown)) if drawdown.max_drawdown != 0 else 0.0
        }
        
        report = ComprehensiveRiskReport(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            drawdown_analysis=drawdown,
            stress_test_results=stress_results,
            portfolio_statistics=portfolio_stats,
            risk_adjusted_metrics=risk_adjusted
        )
        
        self._display_risk_report(report)
        
        return report
    
    def _display_risk_report(self, report: ComprehensiveRiskReport) -> None:
        """Display formatted risk report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE RISK ANALYSIS REPORT")
        print("="*80)
        
        print("\n📊 VALUE AT RISK (VaR):")
        print(f"  95% Confidence: ${abs(report.var_95.var_value):,.2f} ({abs(report.var_95.var_percentage)*100:.2f}%)")
        print(f"  99% Confidence: ${abs(report.var_99.var_value):,.2f} ({abs(report.var_99.var_percentage)*100:.2f}%)")
        
        print("\n📉 CONDITIONAL VALUE AT RISK (CVaR/Expected Shortfall):")
        print(f"  95% Confidence: ${abs(report.cvar_95.cvar_value):,.2f} ({abs(report.cvar_95.cvar_percentage)*100:.2f}%)")
        print(f"  99% Confidence: ${abs(report.cvar_99.cvar_value):,.2f} ({abs(report.cvar_99.cvar_percentage)*100:.2f}%)")
        
        print("\n📉 DRAWDOWN ANALYSIS:")
        print(f"  Maximum Drawdown: {abs(report.drawdown_analysis.max_drawdown)*100:.2f}%")
        print(f"  Drawdown Duration: {report.drawdown_analysis.max_drawdown_duration} days")
        print(f"  Current Drawdown: {abs(report.drawdown_analysis.current_drawdown)*100:.2f}%")
        
        print("\n⚠️  STRESS TEST SCENARIOS:")
        for stress in report.stress_test_results:
            print(f"  {stress.scenario_name:30s}: Loss ${abs(stress.portfolio_loss):>12,.2f} ({abs(stress.portfolio_loss_percentage)*100:>6.2f}%)")
        
        print("\n📈 RISK-ADJUSTED METRICS:")
        for metric, value in report.risk_adjusted_metrics.items():
            print(f"  {metric.replace('_', ' ').title():25s}: {value:>8.3f}")
        
        if report.portfolio_statistics:
            print("\n📊 PORTFOLIO STATISTICS:")
            print(f"  Skewness:      {report.portfolio_statistics.get('skewness', 0):>8.3f}")
            print(f"  Kurtosis:      {report.portfolio_statistics.get('kurtosis', 0):>8.3f}")
            print(f"  Best Day:      {report.portfolio_statistics.get('best_day', 0)*100:>8.2f}%")
            print(f"  Worst Day:     {report.portfolio_statistics.get('worst_day', 0)*100:>8.2f}%")
        
        print("="*80)
    
    def export_risk_report(
        self,
        report: ComprehensiveRiskReport,
        output_dir: str = 'outputs/risk_analysis'
    ) -> None:
        """Export risk report to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export JSON report
        with open(output_path / 'risk_report.json', 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Export stress test results to CSV
        stress_df = pd.DataFrame([
            {
                'Scenario': st.scenario_name,
                'Portfolio_Loss_$': st.portfolio_loss,
                'Loss_Percentage_%': st.portfolio_loss_percentage * 100,
                'New_Portfolio_Value_$': st.new_portfolio_value
            }
            for st in report.stress_test_results
        ])
        stress_df.to_csv(output_path / 'stress_test_results.csv', index=False)
        
        # Export drawdown series
        if report.drawdown_analysis.drawdown_series is not None:
            report.drawdown_analysis.drawdown_series.to_csv(output_path / 'drawdown_series.csv')
        
        logger.success(f"✓ Risk report exported to {output_dir}/")


# Testing
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from monte_carlo_engine import MonteCarloSimulator
    from datetime import datetime, timedelta
    
    logger.add("data/logs/risk_calculator.log", rotation="10 MB")
    
    print("\n" + "="*80)
    print("TESTING RISK CALCULATOR - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Fetch data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    fetcher = DataFetcher(tickers, start_date, end_date)
    market_data = fetcher.fetch_historical_data(use_cache=True)
    stats = fetcher.calculate_statistics()
    
    # Run Monte Carlo simulation
    simulator = MonteCarloSimulator(
        mean_returns=stats.mean_returns,
        cov_matrix=stats.cov_matrix,
        num_simulations=10000,
        random_seed=42
    )
    results = simulator.generate_random_portfolios()
    portfolio_path = simulator.simulate_portfolio_paths(
        weights=results.optimal_portfolio.weights,
        initial_investment=100000,
        time_horizon=252,
        num_paths=1000
    )
    
    # Initialize Risk Calculator
    # Calculate portfolio returns from market data
    portfolio_returns = (market_data.returns * results.optimal_portfolio.weights).sum(axis=1)
    
    risk_calc = RiskCalculator(
        portfolio_returns=portfolio_returns,
        portfolio_paths=portfolio_path.paths,
        initial_investment=100000,
        risk_free_rate=0.02
    )
    
    # Generate comprehensive risk report
    risk_report = risk_calc.generate_comprehensive_report(
        weights=results.optimal_portfolio.weights,
        mean_returns=stats.mean_returns,
        cov_matrix=stats.cov_matrix
    )
    
    # Export report
    risk_calc.export_risk_report(risk_report)
    
    print("\n" + "="*80)
    print("✅ RISK CALCULATOR TEST SUCCESSFUL!")
    print("="*80)
