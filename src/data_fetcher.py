"""
Data Fetcher Module - Monte Carlo Portfolio Simulation
Fetches historical stock data using yfinance with modern Python 3.12 features
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MarketData:
    """Container for market data with metadata"""
    prices: pd.DataFrame
    returns: pd.DataFrame
    tickers: list[str]
    start_date: str
    end_date: str
    trading_days: int
    
    def __post_init__(self) -> None:
        """Validate data after initialization"""
        if self.prices.empty:
            raise ValueError("Price data cannot be empty")
        if self.returns.empty:
            raise ValueError("Returns data cannot be empty")


@dataclass
class StatisticsResult:
    """Container for statistical calculations"""
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    corr_matrix: pd.DataFrame
    volatility: pd.Series
    annualized_returns: pd.Series
    sharpe_ratios: pd.Series
    
    def to_dict(self) -> dict:
        """Convert statistics to dictionary"""
        return {
            'mean_returns': self.mean_returns.to_dict(),
            'annualized_returns': self.annualized_returns.to_dict(),
            'volatility': self.volatility.to_dict(),
            'sharpe_ratios': self.sharpe_ratios.to_dict()
        }


class DataFetcher:
    """
    Advanced data fetcher with caching, validation, and error handling
    
    Features:
    - Automatic retry logic
    - Data caching to disk
    - Comprehensive validation
    - Type-safe operations
    - Structured logging
    """
    
    def __init__(
        self,
        tickers: Sequence[str],
        start_date: str | datetime,
        end_date: str | datetime,
        cache_dir: Path = Path('data/raw'),
        risk_free_rate: float = 0.02
    ) -> None:
        """
        Initialize DataFetcher with modern type hints
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date (YYYY-MM-DD or datetime object)
            end_date: End date (YYYY-MM-DD or datetime object)
            cache_dir: Directory for caching data
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self.tickers = list(tickers)
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date)
        self.cache_dir = cache_dir
        self.risk_free_rate = risk_free_rate
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.market_data: Optional[MarketData] = None
        self.statistics: Optional[StatisticsResult] = None
        
        logger.info(f"Initialized DataFetcher for {len(self.tickers)} tickers")
    
    @staticmethod
    def _parse_date(date_input: str | datetime) -> str:
        """Parse date to string format"""
        if isinstance(date_input, datetime):
            return date_input.strftime('%Y-%m-%d')
        return date_input
    
    def _get_cache_path(self) -> Path:
        """Generate cache file path based on parameters"""
        ticker_str = '_'.join(sorted(self.tickers))
        filename = f"{ticker_str}_{self.start_date}_{self.end_date}.parquet"
        return self.cache_dir / filename
    
    def fetch_historical_data(
        self, 
        use_cache: bool = True,
        retry_attempts: int = 3
    ) -> MarketData:
        """
        Fetch historical stock data with caching and retry logic
        
        Args:
            use_cache: Whether to use cached data if available
            retry_attempts: Number of retry attempts on failure
            
        Returns:
            MarketData object containing prices and returns
        """
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        if use_cache and cache_path.exists():
            logger.info(f"Loading data from cache: {cache_path}")
            try:
                prices = pd.read_parquet(cache_path)
                logger.success(f"✓ Loaded {len(prices)} days from cache")
                return self._process_data(prices)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Fetching fresh data.")
        
        # Fetch fresh data
        logger.info(f"Fetching data for: {', '.join(self.tickers)}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        
        for attempt in range(1, retry_attempts + 1):
            try:
                prices = self._download_data()
                
                if prices is not None and not prices.empty:
                    # Cache the data
                    prices.to_parquet(cache_path)
                    logger.success(f"✓ Data cached to: {cache_path}")
                    
                    return self._process_data(prices)
                    
            except Exception as e:
                logger.error(f"Attempt {attempt}/{retry_attempts} failed: {e}")
                if attempt == retry_attempts:
                    raise RuntimeError(f"Failed to fetch data after {retry_attempts} attempts")
        
        raise RuntimeError("Data fetch failed")
    
    def _download_data(self) -> pd.DataFrame:
        """Download data using yfinance"""
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by='ticker'
        )
        
        # Handle different data structures based on number of tickers
        if len(self.tickers) == 1:
            prices = data[['Adj Close']].copy()
            prices.columns = self.tickers
        else:
            # Multi-level columns: extract Adj Close for each ticker
            if isinstance(data.columns, pd.MultiIndex):
                prices = pd.DataFrame()
                for ticker in self.tickers:
                    if ticker in data:
                        prices[ticker] = data[ticker]['Adj Close']
            else:
                prices = data['Adj Close']
        
        # Data cleaning
        prices = self._clean_data(prices)
        
        logger.info(f"✓ Downloaded {len(prices)} days of data")
        logger.info(f"✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        return prices
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data"""
        # Remove columns with too many missing values (>20%)
        threshold = len(df) * 0.8
        df = df.dropna(axis=1, thresh=int(threshold))
        
        # Forward fill missing values
        df = df.ffill()
        
        # Backward fill any remaining NaN
        df = df.bfill()
        
        # Remove any remaining rows with NaN
        df = df.dropna()
        
        # Validate: no negative prices
        if (df < 0).any().any():
            logger.warning("Negative prices detected and removed")
            df = df[df > 0].dropna()
        
        # Validate: no zeros
        if (df == 0).any().any():
            logger.warning("Zero prices detected and removed")
            df = df[df != 0].dropna()
        
        return df
    
    def _process_data(self, prices: pd.DataFrame) -> MarketData:
        """Process prices into MarketData object"""
        # Calculate returns
        returns = self._calculate_returns(prices)
        
        # Create MarketData object
        self.market_data = MarketData(
            prices=prices,
            returns=returns,
            tickers=list(prices.columns),
            start_date=self.start_date,
            end_date=self.end_date,
            trading_days=len(prices)
        )
        
        # Display summary
        self._display_data_summary()
        
        return self.market_data
    
    def _calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns"""
        # Log returns: ln(P_t / P_t-1)
        returns = np.log(prices / prices.shift(1))
        returns = returns.dropna()
        
        logger.info(f"✓ Calculated returns for {len(returns)} trading days")
        
        # Save to CSV
        returns.to_csv('data/processed/daily_returns.csv')
        
        return returns
    
    def calculate_statistics(self) -> StatisticsResult:
        """
        Calculate comprehensive portfolio statistics
        
        Returns:
            StatisticsResult object with all statistics
        """
        if self.market_data is None:
            raise ValueError("No market data available. Run fetch_historical_data() first.")
        
        returns = self.market_data.returns
        
        # Basic statistics
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        corr_matrix = returns.corr()
        
        # Annualized metrics (252 trading days)
        annualized_returns = mean_returns * 252
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratios = (annualized_returns - self.risk_free_rate) / volatility
        
        self.statistics = StatisticsResult(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            corr_matrix=corr_matrix,
            volatility=volatility,
            annualized_returns=annualized_returns,
            sharpe_ratios=sharpe_ratios
        )
        
        self._display_statistics()
        
        return self.statistics
    
    def _display_data_summary(self) -> None:
        """Display formatted data summary"""
        if self.market_data is None:
            return
        
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(f"Tickers:       {', '.join(self.market_data.tickers)}")
        print(f"Trading Days:  {self.market_data.trading_days}")
        print(f"Start Date:    {self.market_data.start_date}")
        print(f"End Date:      {self.market_data.end_date}")
        print("\nFirst 5 days:")
        print(self.market_data.prices.head())
        print("\nLast 5 days:")
        print(self.market_data.prices.tail())
        print("="*80)
    
    def _display_statistics(self) -> None:
        """Display formatted statistics"""
        if self.statistics is None:
            return
        
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
        
        print("\n📊 Annualized Returns:")
        for ticker, ret in self.statistics.annualized_returns.items():
            print(f"  {ticker:10s}: {ret*100:>8.2f}%")
        
        print("\n📉 Annualized Volatility:")
        for ticker, vol in self.statistics.volatility.items():
            print(f"  {ticker:10s}: {vol*100:>8.2f}%")
        
        print("\n⚡ Sharpe Ratios:")
        for ticker, sharpe in self.statistics.sharpe_ratios.items():
            print(f"  {ticker:10s}: {sharpe:>8.3f}")
        
        print("\n🔗 Correlation Matrix:")
        print(self.statistics.corr_matrix.round(3))
        print("="*80)
    
    def export_data(self) -> None:
        """Export all data to files"""
        if self.market_data is None or self.statistics is None:
            logger.warning("No data to export")
            return
        
        # Export prices
        self.market_data.prices.to_csv('data/processed/prices.csv')
        
        # Export statistics
        stats_df = pd.DataFrame({
            'Ticker': self.market_data.tickers,
            'Annualized_Return_%': self.statistics.annualized_returns.values * 100,
            'Volatility_%': self.statistics.volatility.values * 100,
            'Sharpe_Ratio': self.statistics.sharpe_ratios.values
        })
        stats_df.to_csv('data/processed/statistics.csv', index=False)
        
        # Export correlation matrix
        self.statistics.corr_matrix.to_csv('data/processed/correlation_matrix.csv')
        
        logger.success("✓ All data exported to data/processed/")


# Example usage and testing
if __name__ == "__main__":
    from loguru import logger
    
    logger.add("data/logs/data_fetcher.log", rotation="10 MB")
    
    print("\n" + "="*80)
    print("TESTING DATA FETCHER MODULE - LATEST TECH STACK 2025")
    print("="*80)
    
    # Test configuration
    test_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    test_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    test_end = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Initialize fetcher
        fetcher = DataFetcher(
            tickers=test_tickers,
            start_date=test_start,
            end_date=test_end,
            risk_free_rate=0.02
        )
        
        # Fetch data
        market_data = fetcher.fetch_historical_data(use_cache=True)
        
        # Calculate statistics
        stats = fetcher.calculate_statistics()
        
        # Export everything
        fetcher.export_data()
        
        print("\n" + "="*80)
        print("✅ DATA FETCHER MODULE TEST SUCCESSFUL!")
        print("="*80)
        
    except Exception as e:
        logger.exception("Test failed")
        print(f"\n❌ TEST FAILED: {e}")
