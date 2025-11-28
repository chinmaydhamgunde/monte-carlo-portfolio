"""
Portfolio Visualization Module - Professional Charts & Dashboards
Modern visualization with matplotlib, seaborn, and plotly
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# Modern style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")  # Colorblind-friendly palette


class PortfolioVisualizer:
    """
    Professional portfolio visualization with modern design principles
    
    Features:
    - Efficient frontier plots
    - Risk distribution charts
    - Drawdown analysis
    - Portfolio paths simulation
    - Asset allocation visualizations
    - Stress test comparisons
    - Multi-panel dashboards
    """
    
    def __init__(
        self,
        output_dir: str = 'outputs/charts',
        style: str = 'professional',
        dpi: int = 300
    ) -> None:
        """
        Initialize Portfolio Visualizer
        
        Args:
            output_dir: Directory to save charts
            style: Visualization style ('professional', 'minimal', 'dark')
            dpi: Resolution for saved images (300 for print quality)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Apply style
        self._apply_style(style)
        
        logger.info(f"Initialized Visualizer (style: {style}, dpi: {dpi})")
    
    def _apply_style(self, style: str) -> None:
        """Apply visualization style"""
        if style == 'professional':
            sns.set_theme(style='darkgrid', palette='colorblind')
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': '#f8f9fa',
                'axes.edgecolor': '#2c3e50',
                'axes.labelcolor': '#2c3e50',
                'text.color': '#2c3e50',
                'xtick.color': '#2c3e50',
                'ytick.color': '#2c3e50',
                'grid.color': '#bdc3c7',
                'grid.linestyle': '--',
                'grid.alpha': 0.4,
                'font.family': 'sans-serif',
                'font.size': 10
            })
        elif style == 'dark':
            plt.style.use('dark_background')
        
    def plot_efficient_frontier(
        self,
        all_returns: np.ndarray,
        all_volatilities: np.ndarray,
        all_sharpe_ratios: np.ndarray,
        optimal_portfolio: dict,
        min_vol_portfolio: dict,
        tickers: List[str],
        save_name: str = 'efficient_frontier.png'
    ) -> None:
        """
        Plot the efficient frontier with optimal portfolios highlighted
        
        Args:
            all_returns: Array of portfolio returns
            all_volatilities: Array of portfolio volatilities
            all_sharpe_ratios: Array of Sharpe ratios
            optimal_portfolio: Dict with optimal portfolio info
            min_vol_portfolio: Dict with minimum volatility portfolio
            tickers: List of asset tickers
            save_name: Filename to save
        """
        logger.info("Creating efficient frontier plot...")
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Scatter plot of all portfolios (colored by Sharpe ratio)
        scatter = ax.scatter(
            all_volatilities * 100,
            all_returns * 100,
            c=all_sharpe_ratios,
            cmap='viridis',
            alpha=0.4,
            s=15,
            edgecolors='none'
        )
        
        # Color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, fontsize=11)
        
        # Highlight optimal portfolio (max Sharpe ratio)
        ax.scatter(
            optimal_portfolio['volatility'] * 100,
            optimal_portfolio['expected_return'] * 100,
            marker='*',
            s=800,
            color='gold',
            edgecolors='darkred',
            linewidths=2,
            zorder=5,
            label=f'Optimal Portfolio (Sharpe: {optimal_portfolio["sharpe_ratio"]:.3f})'
        )
        
        # Highlight minimum volatility portfolio
        ax.scatter(
            min_vol_portfolio['volatility'] * 100,
            min_vol_portfolio['expected_return'] * 100,
            marker='D',
            s=300,
            color='lightgreen',
            edgecolors='darkgreen',
            linewidths=2,
            zorder=5,
            label=f'Min Volatility (Vol: {min_vol_portfolio["volatility"]*100:.2f}%)'
        )
        
        # Labels and title
        ax.set_xlabel('Volatility (Standard Deviation) %', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Annual Return %', fontsize=12, fontweight='bold')
        ax.set_title(
            'Efficient Frontier - Portfolio Optimization\n' +
            f'Monte Carlo Simulation ({len(all_returns):,} portfolios)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotation with portfolio composition
        composition_text = "Optimal Portfolio Allocation:\n"
        for ticker, weight in zip(tickers, optimal_portfolio['weights']):
            if weight > 0.01:  # Only show weights > 1%
                composition_text += f"{ticker}: {weight*100:.1f}%  "
        
        ax.text(
            0.98, 0.02, composition_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logger.success(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_portfolio_distributions(
        self,
        portfolio_paths: np.ndarray,
        initial_investment: float,
        percentiles: Dict[int, np.ndarray],
        save_name: str = 'portfolio_distributions.png'
    ) -> None:
        """
        Plot portfolio value distributions and paths
        
        Args:
            portfolio_paths: Array of simulated paths (time x simulations)
            initial_investment: Initial portfolio value
            percentiles: Dict of percentile arrays
            save_name: Filename to save
        """
        logger.info("Creating portfolio distribution plots...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        final_values = portfolio_paths[-1, :]
        days = np.arange(portfolio_paths.shape[0])
        
        # 1. Portfolio paths over time
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot sample paths (100 random simulations)
        sample_indices = np.random.choice(portfolio_paths.shape[1], 100, replace=False)
        for idx in sample_indices:
            ax1.plot(days, portfolio_paths[:, idx], alpha=0.1, color='steelblue', linewidth=0.5)
        
        # Plot percentiles
        ax1.plot(days, percentiles[50], color='darkblue', linewidth=2.5, label='Median (50th percentile)')
        ax1.plot(days, percentiles[5], color='red', linewidth=2, linestyle='--', label='5th percentile')
        ax1.plot(days, percentiles[95], color='green', linewidth=2, linestyle='--', label='95th percentile')
        
        # Fill between 25th and 75th percentiles
        ax1.fill_between(
            days, percentiles[25], percentiles[75],
            alpha=0.3, color='skyblue', label='25th-75th percentile range'
        )
        
        # Initial investment line
        ax1.axhline(y=initial_investment, color='black', linestyle=':', linewidth=2, label='Initial Investment')
        
        ax1.set_xlabel('Trading Days', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
        ax1.set_title('Portfolio Value Simulation - Monte Carlo Paths', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Distribution of final values (histogram)
        ax2 = fig.add_subplot(gs[1, 0])
        
        n, bins, patches = ax2.hist(final_values, bins=60, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Color bars based on profit/loss
        for i, patch in enumerate(patches):
            if bins[i] < initial_investment:
                patch.set_facecolor('salmon')
            else:
                patch.set_facecolor('lightgreen')
        
        # Add vertical lines for key statistics
        ax2.axvline(initial_investment, color='black', linestyle=':', linewidth=2, label='Initial Investment')
        ax2.axvline(np.median(final_values), color='blue', linestyle='--', linewidth=2, label='Median')
        ax2.axvline(np.mean(final_values), color='orange', linestyle='--', linewidth=2, label='Mean')
        
        ax2.set_xlabel('Final Portfolio Value ($)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Distribution of Final Portfolio Values', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 3. Box plot with statistics
        ax3 = fig.add_subplot(gs[1, 1])
        
        box = ax3.boxplot(
            final_values,
            vert=True,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
        )
        
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')
        
        # Add statistics text
        stats_text = (
            f"Mean: ${np.mean(final_values):,.0f}\n"
            f"Median: ${np.median(final_values):,.0f}\n"
            f"Std Dev: ${np.std(final_values):,.0f}\n"
            f"Min: ${np.min(final_values):,.0f}\n"
            f"Max: ${np.max(final_values):,.0f}\n"
            f"Prob(Loss): {np.sum(final_values < initial_investment)/len(final_values)*100:.1f}%"
        )
        
        ax3.text(
            1.15, 0.5, stats_text,
            transform=ax3.transData,
            fontsize=9,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        ax3.set_ylabel('Final Portfolio Value ($)', fontsize=11, fontweight='bold')
        ax3.set_title('Statistical Summary', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax3.set_xticklabels(['Portfolio'])
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logger.success(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_drawdown_analysis(
        self,
        drawdown_series: pd.Series,
        portfolio_values: Optional[pd.Series] = None,
        save_name: str = 'drawdown_analysis.png'
    ) -> None:
        """
        Plot drawdown analysis charts [web:50]
        
        Args:
            drawdown_series: Series of drawdown values
            portfolio_values: Portfolio value over time
            save_name: Filename to save
        """
        logger.info("Creating drawdown analysis plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Portfolio value over time
        if portfolio_values is not None:
            ax1.plot(portfolio_values.index, portfolio_values.values, 
                    linewidth=2, color='steelblue', label='Portfolio Value')
            
            # Running maximum
            running_max = portfolio_values.expanding().max()
            ax1.plot(running_max.index, running_max.values,
                    linewidth=1.5, color='green', linestyle='--', 
                    alpha=0.7, label='Peak Value')
            
            # Fill underwater area
            ax1.fill_between(portfolio_values.index, portfolio_values.values, 
                            running_max.values, where=(portfolio_values < running_max),
                            color='red', alpha=0.2, label='Drawdown Period')
            
            ax1.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
            ax1.set_title('Portfolio Value and Underwater Periods', fontsize=13, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Drawdown over time
        ax2.fill_between(drawdown_series.index, drawdown_series.values * 100, 0,
                        color='crimson', alpha=0.5, label='Drawdown')
        ax2.plot(drawdown_series.index, drawdown_series.values * 100,
                color='darkred', linewidth=1.5)
        
        # Highlight maximum drawdown
        max_dd_idx = drawdown_series.idxmin()
        max_dd_val = drawdown_series.min()
        ax2.scatter([max_dd_idx], [max_dd_val * 100], 
                   color='red', s=200, marker='v', zorder=5,
                   label=f'Max Drawdown: {max_dd_val*100:.2f}%')
        
        ax2.set_xlabel('Time Period', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Drawdown Analysis Over Time', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logger.success(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_asset_allocation(
        self,
        weights: np.ndarray,
        tickers: List[str],
        portfolio_name: str = 'Optimal Portfolio',
        save_name: str = 'asset_allocation.png'
    ) -> None:
        """
        Plot asset allocation pie chart
        
        Args:
            weights: Portfolio weights array
            tickers: List of asset tickers
            portfolio_name: Name for the chart title
            save_name: Filename to save
        """
        logger.info(f"Creating asset allocation chart for {portfolio_name}...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Filter out very small weights for clarity
        significant_mask = weights > 0.01
        display_weights = weights[significant_mask]
        display_tickers = [t for t, m in zip(tickers, significant_mask) if m]
        
        # Add "Other" category if needed
        if len(display_weights) < len(weights):
            other_weight = weights[~significant_mask].sum()
            if other_weight > 0:
                display_weights = np.append(display_weights, other_weight)
                display_tickers.append('Other (<1%)')
        
        # Pie chart
        colors = sns.color_palette("Set2", len(display_weights))
        wedges, texts, autotexts = ax1.pie(
            display_weights,
            labels=display_tickers,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=[0.05 if w == max(display_weights) else 0 for w in display_weights],
            shadow=True,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        ax1.set_title(f'{portfolio_name}\nAsset Allocation', 
                     fontsize=13, fontweight='bold', pad=20)
        
        # Bar chart
        y_pos = np.arange(len(display_tickers))
        bars = ax2.barh(y_pos, display_weights * 100, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, display_weights)):
            ax2.text(weight * 100 + 1, i, f'{weight*100:.1f}%',
                    va='center', fontsize=9, fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(display_tickers, fontsize=10)
        ax2.set_xlabel('Allocation (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Portfolio Weights', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, max(display_weights) * 110)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logger.success(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_risk_metrics(
        self,
        var_95: dict,
        var_99: dict,
        cvar_95: dict,
        cvar_99: dict,
        initial_investment: float,
        save_name: str = 'risk_metrics.png'
    ) -> None:
        """
        Plot comprehensive risk metrics comparison
        
        Args:
            var_95: VaR at 95% confidence
            var_99: VaR at 99% confidence
            cvar_95: CVaR at 95% confidence
            cvar_99: CVaR at 99% confidence
            initial_investment: Initial portfolio value
            save_name: Filename to save
        """
        logger.info("Creating risk metrics visualization...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metrics = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
        values = [
            abs(var_95['value']),
            abs(var_99['value']),
            abs(cvar_95['value']),
            abs(cvar_99['value'])
        ]
        percentages = [
            abs(var_95['percentage']) * 100,
            abs(var_99['percentage']) * 100,
            abs(cvar_95['percentage']) * 100,
            abs(cvar_99['percentage']) * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.6
        
        # Create bars
        bars = ax.bar(x, values, width, color=['#3498db', '#2980b9', '#e74c3c', '#c0392b'],
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val:,.0f}\n({pct:.2f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Potential Loss ($)', fontsize=12, fontweight='bold')
        ax.set_title('Risk Metrics Comparison\nValue at Risk (VaR) & Conditional Value at Risk (CVaR)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Add reference line for initial investment
        ax.axhline(y=initial_investment * 0.1, color='green', linestyle='--',
                  linewidth=2, alpha=0.7, label='10% of Initial Investment')
        ax.legend(fontsize=10)
        
        # Add explanation text
        explanation = (
            "VaR: Maximum expected loss at given confidence level\n"
            "CVaR: Average loss beyond VaR threshold (worst-case scenarios)"
        )
        ax.text(0.02, 0.98, explanation,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logger.success(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_stress_test_results(
        self,
        stress_results: List[dict],
        save_name: str = 'stress_test_results.png'
    ) -> None:
        """
        Plot stress test scenario results
        
        Args:
            stress_results: List of stress test result dicts
            save_name: Filename to save
        """
        logger.info("Creating stress test visualization...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios = [r['scenario'] for r in stress_results]
        losses = [abs(r['loss']) for r in stress_results]
        loss_pcts = [abs(r['loss_percentage']) * 100 for r in stress_results]
        
        # Sort by loss magnitude
        sorted_indices = np.argsort(losses)[::-1]
        scenarios = [scenarios[i] for i in sorted_indices]
        losses = [losses[i] for i in sorted_indices]
        loss_pcts = [loss_pcts[i] for i in sorted_indices]
        
        y_pos = np.arange(len(scenarios))
        
        # Color bars by severity
        colors = ['#c0392b' if pct > 30 else '#e67e22' if pct > 20 else '#f39c12' 
                 for pct in loss_pcts]
        
        bars = ax.barh(y_pos, losses, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add percentage labels
        for i, (bar, loss, pct) in enumerate(zip(bars, losses, loss_pcts)):
            ax.text(loss + max(losses)*0.02, i,
                   f'${loss:,.0f} ({pct:.1f}%)',
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(scenarios, fontsize=10)
        ax.set_xlabel('Potential Loss ($)', fontsize=12, fontweight='bold')
        ax.set_title('Stress Test Scenarios - Portfolio Resilience Analysis',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Add severity legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#c0392b', label='Severe (>30% loss)'),
            Patch(facecolor='#e67e22', label='Moderate (20-30% loss)'),
            Patch(facecolor='#f39c12', label='Mild (<20% loss)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logger.success(f"✓ Saved: {save_path}")
        plt.close()
    
    def create_comprehensive_dashboard(
        self,
        monte_carlo_results: dict,
        risk_report: dict,
        portfolio_path: dict,
        save_name: str = 'comprehensive_dashboard.png'
    ) -> None:
        """
        Create a comprehensive multi-panel dashboard [web:46][web:52]
        
        Args:
            monte_carlo_results: Results from Monte Carlo simulation
            risk_report: Risk analysis report
            portfolio_path: Portfolio path simulation data
            save_name: Filename to save
        """
        logger.info("Creating comprehensive dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Title
        fig.suptitle('Portfolio Analysis Dashboard - Monte Carlo Simulation',
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Add 6 panels with key visualizations
        # This is a simplified version - you can expand with actual data
        
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.text(0.5, 0.5, 'Efficient Frontier\n(Add scatter plot here)',
                ha='center', va='center', fontsize=12)
        ax1.set_title('Risk-Return Tradeoff', fontweight='bold')
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.text(0.5, 0.5, 'Asset\nAllocation\n(Add pie chart)',
                ha='center', va='center', fontsize=12)
        ax2.set_title('Portfolio Weights', fontweight='bold')
        
        ax3 = fig.add_subplot(gs[1, :])
        ax3.text(0.5, 0.5, 'Portfolio Paths Over Time\n(Add path simulation)',
                ha='center', va='center', fontsize=12)
        ax3.set_title('Monte Carlo Simulation Paths', fontweight='bold')
        
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.text(0.5, 0.5, 'Return\nDistribution',
                ha='center', va='center', fontsize=12)
        ax4.set_title('Final Values', fontweight='bold')
        
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.text(0.5, 0.5, 'Risk Metrics\nVaR & CVaR',
                ha='center', va='center', fontsize=12)
        ax5.set_title('Risk Analysis', fontweight='bold')
        
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.text(0.5, 0.5, 'Key Statistics',
                ha='center', va='center', fontsize=10)
        ax6.set_title('Summary', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logger.success(f"✓ Saved: {save_path}")
        plt.close()


# Testing
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from monte_carlo_engine import MonteCarloSimulator
    from risk_calculator import RiskCalculator
    from datetime import datetime, timedelta
    
    logger.add("data/logs/visualizer.log", rotation="10 MB")
    
    print("\n" + "="*80)
    print("TESTING VISUALIZATION MODULE - CREATING BEAUTIFUL CHARTS")
    print("="*80)
    
    # Load all data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch data
    fetcher = DataFetcher(tickers, start_date, end_date)
    market_data = fetcher.fetch_historical_data(use_cache=True)
    stats = fetcher.calculate_statistics()
    
    # Monte Carlo simulation
    simulator = MonteCarloSimulator(
        mean_returns=stats.mean_returns,
        cov_matrix=stats.cov_matrix,
        num_simulations=10000,
        random_seed=42
    )
    mc_results = simulator.generate_random_portfolios()
    portfolio_path = simulator.simulate_portfolio_paths(
        weights=mc_results.optimal_portfolio.weights,
        initial_investment=100000,
        time_horizon=252,
        num_paths=1000
    )
    
    # Risk analysis
    portfolio_returns = (market_data.returns * mc_results.optimal_portfolio.weights).sum(axis=1)
    risk_calc = RiskCalculator(
        portfolio_returns=portfolio_returns,
        portfolio_paths=portfolio_path.paths,
        initial_investment=100000
    )
    risk_report = risk_calc.generate_comprehensive_report(
        weights=mc_results.optimal_portfolio.weights,
        mean_returns=stats.mean_returns,
        cov_matrix=stats.cov_matrix
    )
    
    # Create visualizations
    viz = PortfolioVisualizer(style='professional', dpi=300)
    
    print("\n📊 Creating visualizations...")
    
    # 1. Efficient Frontier
    viz.plot_efficient_frontier(
        all_returns=mc_results.all_returns,
        all_volatilities=mc_results.all_volatilities,
        all_sharpe_ratios=mc_results.all_sharpe_ratios,
        optimal_portfolio=mc_results.optimal_portfolio.to_dict(),
        min_vol_portfolio=mc_results.min_volatility_portfolio.to_dict(),
        tickers=tickers
    )
    
    # 2. Portfolio Distributions
    viz.plot_portfolio_distributions(
        portfolio_paths=portfolio_path.paths,
        initial_investment=100000,
        percentiles=portfolio_path.percentiles
    )
    
    # 3. Asset Allocation
    viz.plot_asset_allocation(
        weights=mc_results.optimal_portfolio.weights,
        tickers=tickers,
        portfolio_name='Optimal Portfolio (Max Sharpe Ratio)'
    )
    
    # 4. Risk Metrics
    viz.plot_risk_metrics(
        var_95={'value': risk_report.var_95.var_value, 'percentage': risk_report.var_95.var_percentage},
        var_99={'value': risk_report.var_99.var_value, 'percentage': risk_report.var_99.var_percentage},
        cvar_95={'value': risk_report.cvar_95.cvar_value, 'percentage': risk_report.cvar_95.cvar_percentage},
        cvar_99={'value': risk_report.cvar_99.cvar_value, 'percentage': risk_report.cvar_99.cvar_percentage},
        initial_investment=100000
    )
    
    # 5. Stress Test Results
    stress_data = [
        {'scenario': st.scenario_name, 'loss': st.portfolio_loss, 'loss_percentage': st.portfolio_loss_percentage}
        for st in risk_report.stress_test_results
    ]
    viz.plot_stress_test_results(stress_data)
    
    # 6. Drawdown Analysis
    portfolio_values = pd.Series(np.median(portfolio_path.paths, axis=1))
    viz.plot_drawdown_analysis(
        drawdown_series=risk_report.drawdown_analysis.drawdown_series,
        portfolio_values=portfolio_values
    )
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION MODULE TEST SUCCESSFUL!")
    print("="*80)
    print(f"\n📁 All charts saved to: outputs/charts/")
    print(f"   Check the directory to see 6 beautiful visualizations!")
