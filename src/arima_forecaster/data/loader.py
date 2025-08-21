"""
Data loading utilities for time series data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any
from ..utils.logger import get_logger
from ..utils.exceptions import DataProcessingError


class DataLoader:
    """
    Load and validate time series data from various sources.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def load_csv(
        self, 
        file_path: Union[str, Path],
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        parse_dates: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load time series data from CSV file.
        
        Args:
            file_path: Path to CSV file
            date_column: Name of date column (if None, assumes index)
            value_column: Name of value column
            parse_dates: Whether to parse dates automatically
            **kwargs: Additional parameters for pd.read_csv
            
        Returns:
            DataFrame with datetime index
            
        Raises:
            DataProcessingError: If file cannot be loaded or processed
        """
        try:
            self.logger.info(f"Loading data from {file_path}")
            
            # Default parameters
            default_params = {
                'parse_dates': parse_dates,
                'index_col': 0 if date_column is None else date_column
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(file_path, **default_params)
            
            # Validate data
            if df.empty:
                raise DataProcessingError("DataFrame caricato è vuoto")
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    raise DataProcessingError(f"Impossibile convertire indice a datetime: {e}")
            
            # Sort by date
            df = df.sort_index()
            
            self.logger.info(f"Successfully loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise DataProcessingError(f"Fallito caricamento dati: {e}")
    
    def validate_time_series(self, df: pd.DataFrame, value_column: str) -> Dict[str, Any]:
        """
        Validate time series data and return quality metrics.
        
        Args:
            df: DataFrame with time series data
            value_column: Name of the target column
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            # Check if column exists
            if value_column not in df.columns:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Column '{value_column}' not found")
                return validation_results
            
            series = df[value_column]
            
            # Check for missing values
            missing_count = series.isnull().sum()
            if missing_count > 0:
                validation_results['issues'].append(f"{missing_count} missing values found")
            
            # Check for constant values
            if series.nunique() == 1:
                validation_results['issues'].append("Series has constant values")
            
            # Check for sufficient data
            if len(series) < 10:
                validation_results['issues'].append("Insufficient data points (< 10)")
            
            # Calculate statistics
            validation_results['statistics'] = {
                'count': len(series),
                'missing_values': missing_count,
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'unique_values': series.nunique()
            }
            
            if validation_results['issues']:
                validation_results['is_valid'] = False
                
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {e}")
        
        return validation_results