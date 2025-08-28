"""
Financial Analysis: Forward Excess Log Return and Maximum Drawdown Calculator

This module calculates forward-looking excess returns and maximum drawdowns
for financial time series data, generating binary labels for investment decisions (is the stock attractive or not?).
"""
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from utilities.parquet_utils import ParquetManager
from utilities.general_utils import ConfigLoader

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_DRAWDOWN_CAP = 0.20  # 20% maximum allowed drawdown

# Configuration dictionaries
HORIZON_BUCKETS = {
    "le_5y": 5 * TRADING_DAYS_PER_YEAR,         # ≤ 5 years
    "gt_5_to_10y": 10 * TRADING_DAYS_PER_YEAR,  # 5-10 years  
    "gt_10y": 15 * TRADING_DAYS_PER_YEAR,       # > 10 years
}

RETURN_THRESHOLDS = {
    "le_5y": 0.00,      # Beat risk-free rate
    "gt_5_to_10y": 0.01, # 1% margin above risk-free
    "gt_10y": 0.00,     # Beat risk-free rate
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('./logs/financial_data_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """
    A class for calculating forward excess returns and maximum drawdowns
    for financial time series analysis.
    """
    
    def __init__(self, config_path: str = './config/config.yml'):
        """
        Initialize the FinancialAnalyzer.
        
       Args:
            config_path: Path to the configuration YAML file
        """

        self.parquet_manager = ParquetManager()
        self.config_loader = ConfigLoader()
        self.config = self.config_loader._load_config(config_path)
        
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load price and risk-free rate data from parquet files.
        
        Returns:
            Tuple of (prices DataFrame, risk-free rate Series)
            
        Raises:
            FileNotFoundError: If required data files are not found
            ValueError: If data format is invalid
        """
        try:
            prices_path = self.config["DATA"] + "/prices.parquet"
            rf_path = self.config["DATA"] + "/risk_free.parquet"
                
            prices = self.parquet_manager.read_parquet(prices_path)
            rf = self.parquet_manager.read_parquet(rf_path)
            
            # Validate data structure
            if prices.empty:
                raise ValueError("Prices data is empty")
            if "rf" not in rf.columns:
                raise ValueError("Risk-free rate data must contain 'rf' column")             
            
            logger.info(f"Loaded data: {len(prices)} dates, {len(prices.columns)} assets")
            
            return prices, rf
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def calculate_forward_excess_log_return(
        self, 
        prices: pd.DataFrame, 
        rf_daily: pd.Series, 
        horizon_days: int
    ) -> pd.DataFrame:
        """
        Calculate forward H-day excess log return.
        
        Formula: ln(P_{t+H}/P_t) - sum_{t+1..t+H} rf
        
        Args:
            prices: DataFrame with price data (index=date, columns=tickers)
            rf_daily: Series with daily risk-free rates
            horizon_days: Forward-looking horizon in trading days
            
        Returns:
            DataFrame with forward excess log returns
        """
        log_prices = np.log(prices)
        forward_log_return = log_prices.shift(-horizon_days) - log_prices
        
        # Calculate cumulative risk-free return over horizon
        rf_cumulative = rf_daily.rolling(horizon_days).sum().reindex(prices.index).ffill()
        
        # Broadcast risk-free rates to match prices DataFrame structure
        rf_panel = pd.DataFrame(
            np.repeat(rf_cumulative.values.reshape(-1, 1), prices.shape[1], axis=1),
            index=prices.index, 
            columns=prices.columns
        )
        
        return forward_log_return - rf_panel
    
    def calculate_forward_maximum_drawdown(
        self, 
        prices: pd.DataFrame, 
        horizon_days: int
    ) -> pd.DataFrame:
        """
        Calculate forward maximum drawdown over [t, t+H] for each start date t.
        
        Args:
            prices: DataFrame with price data
            horizon_days: Forward-looking horizon in trading days
            
        Returns:
            DataFrame with maximum drawdowns (≤ 0)
        """
        mdd_results = pd.DataFrame(
            index=prices.index, 
            columns=prices.columns, 
            dtype=float
        )
        
        price_values = prices.values
        num_dates, num_assets = price_values.shape
        
        for asset_idx in range(num_assets):
            asset_prices = price_values[:, asset_idx]
            drawdowns = np.full(num_dates, np.nan, dtype=float)
            
            # Calculate rolling maximum drawdown for each starting date
            for start_idx in range(num_dates - horizon_days):
                window_prices = asset_prices[start_idx:start_idx + horizon_days + 1]
                drawdowns[start_idx] = self._calculate_window_max_drawdown(window_prices)
                
            mdd_results.iloc[:, asset_idx] = drawdowns
            
        return mdd_results
    
    def _calculate_window_max_drawdown(self, price_window: np.ndarray) -> float:
        """
        Calculate maximum drawdown for a single price window.
        
        Args:
            price_window: Array of prices for the window period
            
        Returns:
            Maximum drawdown (most negative value)
        """
        peak_price = price_window[0]
        max_drawdown = 0.0
        
        for price in price_window:
            if price > peak_price:
                peak_price = price
            
            current_drawdown = price / peak_price - 1.0
            if current_drawdown < max_drawdown:
                max_drawdown = current_drawdown
                
        return max_drawdown
    
    def generate_labels_for_horizon(
        self, 
        prices: pd.DataFrame, 
        rf: pd.Series, 
        horizon_days: int, 
        return_threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate binary labels based on return and risk criteria.
        
        Args:
            prices: Price data DataFrame
            rf: Risk-free rate Series
            horizon_days: Investment horizon in trading days
            return_threshold: Minimum annualized excess return threshold
            
        Returns:
            Tuple of (labels, annualized_excess_returns, max_drawdowns)
        """
        # Calculate forward excess returns and annualize them
        excess_returns = self.calculate_forward_excess_log_return(prices, rf, horizon_days)
        annualized_excess_returns = excess_returns * (TRADING_DAYS_PER_YEAR / horizon_days)
        
        # Calculate forward maximum drawdowns
        max_drawdowns = self.calculate_forward_maximum_drawdown(prices, horizon_days)
        
        # Apply criteria for labeling
        return_criterion = annualized_excess_returns >= return_threshold
        risk_criterion = max_drawdowns >= (DEFAULT_DRAWDOWN_CAP)
        
        # Generate binary labels (1 if both criteria met, 0 otherwise)
        labels = (return_criterion & risk_criterion).astype(int)
        
        # Remove tail where forward horizon data isn't available
        if horizon_days < len(prices.index):
            valid_index = prices.index[:-horizon_days]
        else:
            valid_index = prices.index[:0]  # Empty index if horizon too long
            
        labels = labels.loc[valid_index]
        annualized_excess_returns = annualized_excess_returns.loc[valid_index]
        max_drawdowns = max_drawdowns.loc[valid_index]
        
        return labels, annualized_excess_returns, max_drawdowns
    
    def save_results(
        self, 
        bucket_name: str, 
        labels: pd.DataFrame, 
        returns: pd.DataFrame, 
        drawdowns: pd.DataFrame
    ) -> None:
        """
        Save analysis results to parquet files.
        
        Args:
            bucket_name: Name identifier for the time horizon bucket
            labels: Binary labels DataFrame
            returns: Annualized excess returns DataFrame
            drawdowns: Maximum drawdowns DataFrame
        """
        try:
            # Save individual components
            self.parquet_manager.save_parquet(returns, (self.config["ART"] + f"/target_ann_excess_{bucket_name}.parquet"))
            self.parquet_manager.save_parquet(drawdowns, (self.config["ART"] + f"/mdd_{bucket_name}.parquet"))
            self.parquet_manager.save_parquet(labels, (self.config["ART"] + f"/labels_{bucket_name}.parquet"))
            
            logger.info(f"Saved results for {bucket_name}")
            
        except Exception as e:
            logger.error(f"Error saving results for {bucket_name}: {e}")
            raise
    
    def run_analysis(
        self, 
        horizon_buckets: Optional[Dict[str, int]] = None,
        return_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete financial analysis pipeline.
        
        Args:
            horizon_buckets: Dictionary mapping bucket names to horizon days
            return_thresholds: Dictionary mapping bucket names to return thresholds
            
        Returns:
            Dictionary mapping bucket names to label DataFrames
        """
        if horizon_buckets is None:
            horizon_buckets = HORIZON_BUCKETS
        if return_thresholds is None:
            return_thresholds = RETURN_THRESHOLDS
            
        # Load data
        prices, rf = self.load_data()
        
        # Process each time horizon bucket
        all_labels = {}
        
        for bucket_name, horizon_days in horizon_buckets.items():
            if horizon_days >= len(prices.index):
                logger.warning(
                    f"Insufficient data for '{bucket_name}' "
                    f"(requires {horizon_days} days, have {len(prices.index)}). Skipping."
                )
                continue
                
            logger.info(f"Processing {bucket_name} (horizon: {horizon_days} days)")
            
            # Generate labels and components
            labels, returns, drawdowns = self.generate_labels_for_horizon(
                prices, rf, horizon_days, return_thresholds[bucket_name]
            )
            
            # Save results
            self.save_results(bucket_name, labels, returns, drawdowns)
            all_labels[bucket_name] = labels
        
        # Save combined results if any labels were generated
        if all_labels:
            combined_labels = pd.concat(all_labels, axis=1)
            combined_labels.to_parquet(self.config["ART"] + "/labels_all_buckets.parquet")
            logger.info("Saved combined labels to labels_all_buckets.parquet")
        else:
            logger.info("No labels generated due to insufficient historical data")
            
        return all_labels


def main():
    """Main execution function."""
    try:
        analyzer = FinancialAnalyzer()
        results = analyzer.run_analysis()
        
        if results:
            logger.info(f"Analysis completed successfully. Generated {len(results)} label sets.")
        else:
            logger.warning("No results generated. Check data availability and horizon settings.")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
