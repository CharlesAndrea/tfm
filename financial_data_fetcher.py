import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
import yfinance as yf
from utilities.parquet_utils import ParquetManager

# Constants
TRADING_DAYS_PER_YEAR = 252.0
IRX_TICKER = '^IRX'
BUSINESS_DAY_FREQ = 'B'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('financial_data_fetch.log')
    ]
)
logger = logging.getLogger(__name__)

class FinancialDataFetcher:

    def __init__(self, config_path: str = './config/config.yml'):
        """
        Initialize the FinancialDataFetcher.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.parquet_manager = ParquetManager()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise
    
    def fetch_prices_data(self) -> pd.DataFrame:
        """
        Fetch and process stock prices data.
        
        Returns:
            DataFrame with processed prices data
            
        Raises:
            Exception: If data fetching fails
        """
        try:
            logger.info("=== Step 1: Fetching prices data ===")
            
            tickers = self.config['universe']['tickers']
            start_date = pd.Timestamp(self.config['universe']['start_date'])
            end_date = pd.Timestamp.today().normalize() 

            # Fetch raw data
            raw_data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                auto_adjust=True, 
                progress=False
            )
            
            if raw_data.empty:
                raise ValueError("No price data retrieved")
            
            # Extract close prices
            if isinstance(raw_data.columns, pd.MultiIndex):
                prices = raw_data['Close']
            else:
                prices = raw_data
            
            # Process data: business day frequency and forward fill
            prices = prices.asfreq(BUSINESS_DAY_FREQ).ffill()
            prices.index.name = 'date'
            
            logger.info(f"✓ Prices data collected for {len(prices.columns)} tickers")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching prices data: {e}")
            raise
    
    def fetch_risk_free_rate(self, prices_index: any) -> pd.DataFrame:
        """
        Fetch and process risk-free rate data.
        
        Args:
            prices_index: DatetimeIndex from prices data for alignment
            
        Returns:
            DataFrame with daily risk-free rates
            
        Raises:
            Exception: If data fetching fails
        """
        try:
            logger.info("=== Step 2: Fetching risk-free rate ===")
            
            start_date = pd.Timestamp(self.config['universe']['start_date'])
            end_date = pd.Timestamp.today().normalize() 

            # Fetch IRX data (1-month Treasury bill rate)
            irx_data = yf.download(
                IRX_TICKER, 
                start=start_date, 
                end=end_date, 
                progress=False
            )
            
            if irx_data.empty:
                raise ValueError("No risk-free rate data retrieved")
            
            irx_close = irx_data['Close']
 
            # Convert annualized percentage to daily rate
            rf_daily = (irx_close / 100.0 / TRADING_DAYS_PER_YEAR).reindex(prices_index).ffill()
            
            # Create DataFrame with proper column name
            rf_daily.rename(columns={'^IRX': 'rf'}, inplace=True)
            rf_daily.index.name = 'date'
                    
            logger.info("✓ Risk-free rate data collected")
            
            return rf_daily
            
        except Exception as e:
            logger.error(f"Error fetching risk-free rate data: {e}")
            raise
    
    def save_data(self, prices: pd.DataFrame, rf_daily: pd.DataFrame) -> None:
        """
        Save processed data to parquet files.
        
        Args:
            prices: Processed prices DataFrame
            rf_daily: Processed risk-free rate DataFrame
        """
        try:
            logger.info("=== Saving parquet files ===")
            
            data_dir = self.config['DATA']
            
            prices_path = data_dir + '/' + 'prices.parquet'
            rf_path = data_dir + '/' + 'risk_free.parquet'
            
            self.parquet_manager.save_parquet(prices, prices_path)
            self.parquet_manager.save_parquet(rf_daily, rf_path)
            
            logger.info("✅ SUCCESS! Files saved:")
            logger.info(f"  • {prices_path}")
            logger.info(f"  • {rf_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def validate_saved_data(self) -> None:
        """
        Perform sanity checks on saved data.
        
        Raises:
            AssertionError: If data validation fails
        """
        try:
            logger.info("=== Parquet files sanity check ===")
            
            data_dir = self.config['DATA']
            prices_path = data_dir + '/' + 'prices.parquet'
            rf_path = data_dir + '/' + 'risk_free.parquet'
            
            # Load saved data
            prices = self.parquet_manager.read_parquet(prices_path)
            rf_daily = self.parquet_manager.read_parquet(rf_path)
            
            # Validation checks
            assert prices.index.is_monotonic_increasing, "Prices index not monotonic"
            assert rf_daily.index.is_monotonic_increasing, "Risk-free rate index not monotonic"
            assert prices.index.tz is None, "Prices index has timezone info"
            assert rf_daily.index.tz is None, "Risk-free rate index has timezone info"
            assert prices.notna().sum().sum() > 0, "Empty prices data"
            assert set(prices.index).issuperset(rf_daily.index[:5]), "Risk-free rate index not aligned"
            
            # Display sample data
            logger.info(f"Prices index frequency: {prices.index.freq}")
            logger.info("Last 3 prices entries:")
            logger.info(f"\n{prices.tail(3)}")
            logger.info("Last 3 risk-free rate entries:")
            logger.info(f"\n{rf_daily.tail(3)}")
            
            logger.info("✓ Data validation passed")
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def run(self) -> None:
        """Execute the complete data fetching and processing pipeline."""
        try:
            logger.info("Starting financial data fetching pipeline")
            
            # Fetch and process data
            prices = self.fetch_prices_data()
            rf_daily = self.fetch_risk_free_rate(prices.index)
            
            # Save processed data
            self.save_data(prices, rf_daily)
            
            # Validate saved data
            self.validate_saved_data()
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main() -> None:
    """Main entry point for the script."""
    try:
        fetcher = FinancialDataFetcher()
        fetcher.run()
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
