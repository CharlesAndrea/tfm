"""
Financial Features Engineering Pipeline

This module processes financial price data to generate various technical indicators
and features for quantitative analysis, including momentum, volatility, beta, and turnover metrics.
"""
import logging
import sys
from typing import List
import warnings

import numpy as np
import pandas as pd
from utilities.general_utils import ConfigLoader
from utilities.parquet_utils import ParquetManager

# Constants
TRADING_DAYS_PER_YEAR = 252.0
DEFAULT_WINSORIZE_LOW = 1
DEFAULT_WINSORIZE_HIGH = 99

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('.logs/financial_features.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class FinancialFeatureBuilder:
    """Configuration settings for feature engineering pipeline."""
    
    def __init__(self, config_path: str = './config/config.yml'):
        self.momentum_windows = [63, 126, 252, 756, 1260, 2520, 3780] 
        self.volatility_window = [63, 252, 756, 1260, 2520, 3780] 
        self.beta_window = [252, 756, 1260, 2520, 3780] 
        self.config_loader = ConfigLoader()
        self.config = self.config_loader._load_config(config_path)
        self.parquet_manager = ParquetManager


    def annualize_volatility(self, volatility: pd.Series) -> pd.Series:
        """
        Annualize volatility by scaling with square root of trading days.
    
        Args:
            volatility: Daily volatility series
        
        Returns:
            Annualized volatility series
        """
        return volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple returns from price data.
    
        Args:
            prices: DataFrame with price data
        
        Returns:
            DataFrame with return data
        """
        logger.info("Calculating returns from price data")
        return prices.pct_change()

    def calculate_momentum(self, prices: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculate price momentum over specified window.
    
        Args:
            prices: DataFrame with price data
            window: Number of periods for momentum calculation
        
        Returns:
            DataFrame with momentum values
        """
        logger.info(f"Calculating momentum with {window}-day window")
        return prices.pct_change(window)

    def calculate_rolling_volatility(self, returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculate annualized rolling volatility.
    
        Args:
            returns: DataFrame with return data
            window: Rolling window size
        
        Returns:
            DataFrame with annualized volatility
        """
        logger.info(f"Calculating rolling volatility with {window}-day window")
        return self.annualize_volatility(returns.rolling(window).std())

    def calculate_downside_deviation(self, returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculate annualized downside deviation (volatility of negative returns only).
    
        Args:
            returns: DataFrame with return data
            window: Rolling window size
        
        Returns:
            DataFrame with downside deviation
        """
        logger.info(f"Calculating downside deviation with {window}-day window")
        negative_returns = returns.clip(upper=0)
        downside_var = negative_returns.pow(2).rolling(window).mean()
        return self.annualize_volatility(np.sqrt(downside_var))


    def calculate_beta_to_market(self, returns: pd.DataFrame, benchmark: pd.Series, window: int) -> pd.DataFrame:
        """
        Calculate rolling beta to market benchmark.
    
        Args:
            returns: DataFrame with asset returns
            benchmark: Series with benchmark returns
            window: Rolling window size
        
        Returns:
            DataFrame with beta values
        """
        logger.info(f"Calculating beta to market with {window}-day window")
    
        try:
            covariance = returns.rolling(window).cov(benchmark)
            variance = benchmark.rolling(window).var()
            beta = covariance.div(variance, axis=0)
        
            # Handle infinite values
            beta = beta.replace([np.inf, -np.inf], np.nan)
        
            return beta
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return pd.DataFrame(index=returns.index, columns=returns.columns)

    def winsorize_cross_sectional(
        self,
        features: pd.DataFrame, 
        percentile_low: int = DEFAULT_WINSORIZE_LOW, 
        percentile_high: int = DEFAULT_WINSORIZE_HIGH
    ) -> pd.DataFrame:
        """
        Apply cross-sectional winsorization to features.
    
        Args:
            features: DataFrame with MultiIndex columns (ticker, feature)
            percentile_low: Lower percentile for clipping
            percentile_high: Upper percentile for clipping
        
        Returns:
            Winsorized DataFrame
        """
        logger.info(f"Applying cross-sectional winsorization ({percentile_low}%-{percentile_high}%)")
    
        output = features.copy()
        feature_names = output.columns.get_level_values("feature").unique()
    
        for feature_name in feature_names:
            try:
                feature_data = output.xs(feature_name, axis=1, level="feature")
            
                # Calculate percentiles across tickers for each date
                lower_bound = feature_data.quantile(percentile_low / 100.0, axis=1)
                upper_bound = feature_data.quantile(percentile_high / 100.0, axis=1)
            
                # Apply clipping
                clipped_data = feature_data.clip(lower=lower_bound, upper=upper_bound, axis=0)
            
                # Update output DataFrame
                for ticker in clipped_data.columns:
                    output[(ticker, feature_name)] = clipped_data[ticker]
                
            except Exception as e:
                logger.warning(f"Error winsorizing feature {feature_name}: {e}")
    
        # Sort columns for consistent ordering
        output = output.reindex(columns=sorted(output.columns, key=lambda x: (x[1], x[0])))
        return output

    def create_multiindex_feature(self, dataframe: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Convert DataFrame to MultiIndex format for feature storage.
    
        Args:
            dataframe: Input DataFrame
            feature_name: Name of the feature
        
        Returns:
            DataFrame with MultiIndex columns (ticker, feature)
        """
        result = dataframe.copy()
        result.columns = pd.MultiIndex.from_tuples(
            [(col, feature_name) for col in dataframe.columns],
            names=["ticker", "feature"]
        )
        return result


    def get_market_benchmark(self, returns: pd.DataFrame) -> pd.Series:
        """
        Get market benchmark for beta calculation.
    
        Args:
            returns: DataFrame with return data
        
        Returns:
            Series with benchmark returns
        """
        if "SPY" in returns.columns:
            logger.info("Using SPY as market benchmark")
            return returns["SPY"]
        else:
            logger.info("SPY not available, using equal-weight index as benchmark")
            return returns.mean(axis=1)


    def build_momentum_features(self, prices: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Build momentum features for different time windows.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            List of DataFrames with momentum features
        """
        logger.info("Building momentum features")
        features = []
        
        for window in self.momentum_windows:
            try:
                momentum_data = self.calculate_momentum(prices, window)
                features.append(self.create_multiindex_feature(momentum_data, f"mom{window}"))
            except Exception as e:
                logger.error(f"Error calculating momentum for window {window}: {e}")
        
        return features

    def build_volatility_features(self, returns: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Build volatility-related features.
        
        Args:
            returns: DataFrame with return data
            
        Returns:
            List of DataFrames with volatility features
        """
        logger.info("Building volatility features")
        features = []
        
        try:
            # Rolling volatility
            vol_data = self.calculate_rolling_volatility(returns, self.volatility_window)
            features.append(self.create_multiindex_feature(vol_data, f"vol{self.volatility_window}"))
            
            # Downside deviation
            downside_data = self.calculate_downside_deviation(returns, self.volatility_window)
            features.append(self.create_multiindex_feature(downside_data, f"ddown{self.volatility_window}"))
            
        except Exception as e:
            logger.error(f"Error calculating volatility features: {e}")
        
        return features

    def build_beta_features(self, returns: pd.DataFrame, benchmark: pd.Series) -> List[pd.DataFrame]:
        """
        Build beta features relative to market benchmark.
        
        Args:
            returns: DataFrame with return data
            benchmark: Series with benchmark returns
            
        Returns:
            List of DataFrames with beta features
        """
        logger.info("Building beta features")
        features = []
        
        try:
            beta_data = self.calculate_beta_to_market(returns, benchmark, self.beta_window)
            features.append(self.create_multiindex_feature(beta_data, f"beta{self.beta_window}"))
        except Exception as e:
            logger.error(f"Error calculating beta features: {e}")
        
        return features


# Main pipeline
    def run_feature_engineering_pipeline(self) -> None:
        """
        Execute the complete feature engineering pipeline.
        """
        logger.info("Starting feature engineering pipeline")
        
        try:
            # Load data
            prices = self.parquet_manager.read_parquet(f"{self.config["DATA"]}/prices.parquet")
            returns = self.calculate_returns(prices)
            benchmark = self.get_market_benchmark(returns)
            
            # Build all features
            all_features = []
            
            # Momentum features
            all_features.extend(self.build_momentum_features(prices))
            
            # Volatility features
            all_features.extend(self.build_volatility_features(returns))
            
            # Beta features
            all_features.extend(self.build_beta_features(returns, benchmark))
            
            if not all_features:
                logger.error("No features were successfully generated")
                return
            
            # Combine all features
            logger.info("Combining and processing features")
            combined_features = pd.concat(all_features, axis=1).sort_index(axis=1)
            
            # Clean data
            combined_features = combined_features.dropna(how="all")
            
            # Apply winsorization
            final_features = self.winsorize_cross_sectional(combined_features)
            
            # Save results
            logger.info("Saving features")
            self.parquet_manager.save_parquet(final_features, (self.config["ART"] + "/features.parquet"))
            final_features.to_parquet(self.output_file)
            
            # Log summary statistics
            feature_names = sorted(final_features.columns.get_level_values('feature').unique().tolist())
            date_range = f"{final_features.index.min().date()} → {final_features.index.max().date()}"
            
            logger.info("=" * 50)
            logger.info("FEATURE ENGINEERING COMPLETE")
            logger.info("=" * 50)
            logger.info("Output file: features.parquet")
            logger.info(f"Feature set: {feature_names}")
            logger.info(f"Date range: {date_range}")
            logger.info(f"Total rows: {len(final_features):,}")
            logger.info(f"Total features: {len(feature_names)}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise

if __name__ == "__main__":
    # Suppress pandas warnings for cleaner output
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    
    try:
        feature_builder = FinancialFeatureBuilder()
        feature_builder.run_feature_engineering_pipeline()

        logger.info("Features built successfully")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
