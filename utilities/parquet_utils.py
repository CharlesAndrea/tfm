import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class ParquetManager:
    """Manager class for parquet file operations."""
    
    def __init__(self):
        """Initialize ParquetManager."""
        pass
    
    def save_parquet(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments for to_parquet()
        
        Raises:
            Exception: If saving fails
        """
        try:
            # Ensure directory exists
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet
            df.to_parquet(path, index=True, engine='pyarrow', compression='snappy')

            logger.info(f"Saved parquet file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save parquet file {file_path}: {e}")
            raise
    
    def read_parquet(self, file_path: str) -> pd.DataFrame:
        """
        Read DataFrame from parquet file.
        
        Args:
            file_path: Path to the parquet file
            **kwargs: Additional arguments for read_parquet()
        
        Returns:
            DataFrame loaded from parquet file
        
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If reading fails
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Parquet file not found: {file_path}")
            
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded parquet file: {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read parquet file {file_path}: {e}")
            raise
