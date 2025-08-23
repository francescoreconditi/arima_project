# ============================================
# PREPROCESSING UTILITIES PER SARIMAX
# Creato da: Claude Code  
# Data: 2025-08-23
# Scopo: Preprocessing robusto per variabili esogene
# ============================================

"""
Utilities per preprocessing robusto delle variabili esogene in modelli SARIMAX.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ExogenousPreprocessor:
    """Preprocessore robusto per variabili esogene."""
    
    def __init__(self, method: str = 'robust', handle_outliers: bool = True):
        """
        Inizializza il preprocessore.
        
        Args:
            method: Metodo di scaling ('robust', 'standard', 'minmax', 'none')
            handle_outliers: Se rimuovere/gestire outlier
        """
        self.method = method
        self.handle_outliers = handle_outliers
        self.scalers = {}
        self.outlier_bounds = {}
        self.is_fitted = False
        
    def fit(self, exog_data: pd.DataFrame) -> 'ExogenousPreprocessor':
        """
        Adatta il preprocessore ai dati.
        
        Args:
            exog_data: DataFrame con variabili esogene
            
        Returns:
            Self per method chaining
        """
        logger.info(f"Fitting preprocessore su {len(exog_data)} osservazioni, {len(exog_data.columns)} variabili")
        
        # Reset stato precedente
        self.scalers = {}
        self.outlier_bounds = {}
        
        for col in exog_data.columns:
            series = exog_data[col].copy()
            
            # Gestisci valori mancanti
            if series.isna().any():
                logger.warning(f"Variabile {col} ha {series.isna().sum()} valori mancanti - interpolando")
                series = series.interpolate(method='linear')
                series = series.fillna(series.mean())  # Fallback per estremi
            
            # Calcola bounds per outlier se richiesto
            if self.handle_outliers:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                
                # Conta outlier
                outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"Variabile {col}: {outliers} outlier rilevati su {len(series)} osservazioni")
            
            # Configura scaler
            if self.method == 'robust':
                scaler = RobustScaler()
            elif self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            else:  # 'none'
                scaler = None
            
            if scaler is not None:
                # Fit dello scaler sui dati puliti
                clean_series = self._handle_outliers(series, col) if self.handle_outliers else series
                scaler.fit(clean_series.values.reshape(-1, 1))
                self.scalers[col] = scaler
                
                # Log statistiche
                logger.info(f"Variabile {col}: mean={clean_series.mean():.3f}, std={clean_series.std():.3f}, range=[{clean_series.min():.3f}, {clean_series.max():.3f}]")
        
        self.is_fitted = True
        logger.info("Preprocessore addestrato con successo")
        return self
    
    def transform(self, exog_data: pd.DataFrame) -> pd.DataFrame:
        """
        Trasforma le variabili esogene.
        
        Args:
            exog_data: DataFrame da trasformare
            
        Returns:
            DataFrame trasformato
        """
        if not self.is_fitted:
            raise ValueError("Preprocessore non ancora addestrato. Chiama fit() prima.")
            
        logger.info(f"Trasformazione di {len(exog_data)} osservazioni")
        
        result = exog_data.copy()
        
        for col in exog_data.columns:
            if col not in self.scalers and col not in self.outlier_bounds:
                logger.warning(f"Variabile {col} non vista durante il fit - uso trasformazione identità")
                continue
                
            series = result[col].copy()
            
            # Gestisci valori mancanti
            if series.isna().any():
                series = series.interpolate(method='linear')
                series = series.fillna(series.mean())
            
            # Gestisci outlier
            if self.handle_outliers and col in self.outlier_bounds:
                series = self._handle_outliers(series, col)
            
            # Applica scaling
            if col in self.scalers and self.scalers[col] is not None:
                scaled = self.scalers[col].transform(series.values.reshape(-1, 1))
                result[col] = scaled.flatten()
        
        return result
    
    def fit_transform(self, exog_data: pd.DataFrame) -> pd.DataFrame:
        """Fit e transform in una chiamata."""
        return self.fit(exog_data).transform(exog_data)
    
    def _handle_outliers(self, series: pd.Series, col: str) -> pd.Series:
        """Gestisce outlier in una serie."""
        if col not in self.outlier_bounds:
            return series
            
        lower_bound, upper_bound = self.outlier_bounds[col]
        result = series.copy()
        
        # Clip outlier ai bounds (più conservativo del removal)
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        if outlier_mask.any():
            result = result.clip(lower=lower_bound, upper=upper_bound)
            logger.debug(f"Variabile {col}: {outlier_mask.sum()} outlier clippati ai bounds [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        return result
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Restituisce statistiche del preprocessing."""
        if not self.is_fitted:
            return {}
            
        stats = {}
        for col in self.scalers.keys():
            scaler = self.scalers[col]
            col_stats = {'method': self.method}
            
            if scaler is not None:
                if hasattr(scaler, 'center_'):
                    col_stats['center'] = float(scaler.center_)
                if hasattr(scaler, 'scale_'):
                    col_stats['scale'] = float(scaler.scale_)
                if hasattr(scaler, 'mean_'):
                    col_stats['mean'] = float(scaler.mean_)
                if hasattr(scaler, 'var_'):
                    col_stats['var'] = float(scaler.var_)
            
            if col in self.outlier_bounds:
                col_stats['outlier_bounds'] = self.outlier_bounds[col]
                
            stats[col] = col_stats
            
        return stats

def validate_exog_data(exog_data: pd.DataFrame, series_length: int) -> Tuple[bool, str]:
    """
    Valida i dati esogeni per compatibilità SARIMAX.
    
    Args:
        exog_data: DataFrame variabili esogene
        series_length: Lunghezza della serie temporale
        
    Returns:
        (is_valid, error_message)
    """
    try:
        # Check lunghezza
        if len(exog_data) != series_length:
            return False, f"Lunghezza variabili esogene ({len(exog_data)}) != lunghezza serie ({series_length})"
        
        # Check valori mancanti
        missing_counts = exog_data.isna().sum()
        if missing_counts.any():
            missing_pct = (missing_counts / len(exog_data) * 100).round(1)
            high_missing = missing_pct[missing_pct > 50]
            if len(high_missing) > 0:
                return False, f"Troppe righe mancanti in: {dict(high_missing)}%"
        
        # Check varianza zero
        zero_var_cols = []
        for col in exog_data.columns:
            if exog_data[col].nunique() <= 1:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            return False, f"Variabili con varianza zero: {zero_var_cols}"
        
        # Check valori infiniti
        inf_cols = []
        for col in exog_data.columns:
            if np.isinf(exog_data[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            return False, f"Variabili con valori infiniti: {inf_cols}"
        
        # Check correlazioni perfette
        if len(exog_data.columns) > 1:
            corr_matrix = exog_data.corr().abs()
            # Rimuovi la diagonale
            np.fill_diagonal(corr_matrix.values, 0)
            high_corr = (corr_matrix > 0.99).any().any()
            
            if high_corr:
                # Trova le coppie con correlazione alta
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.99:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                return False, f"Correlazioni troppo alte (>99%) tra: {high_corr_pairs}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Errore validazione: {str(e)}"

def suggest_preprocessing_method(exog_data: pd.DataFrame) -> str:
    """
    Suggerisce il metodo di preprocessing migliore per i dati.
    
    Args:
        exog_data: DataFrame variabili esogene
        
    Returns:
        Metodo raccomandato ('robust', 'standard', 'minmax', 'none')
    """
    try:
        # Analizza le caratteristiche dei dati
        has_outliers = False
        scale_differences = []
        
        for col in exog_data.columns:
            series = exog_data[col].dropna()
            
            # Check outlier con metodo IQR
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_count = ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
            
            if outlier_count / len(series) > 0.05:  # >5% outlier
                has_outliers = True
            
            # Check scala dei dati
            scale_differences.append(abs(np.log10(series.std() + 1e-8)))
        
        # Differenza di scala tra variabili
        max_scale_diff = max(scale_differences) - min(scale_differences) if len(scale_differences) > 1 else 0
        
        # Logica di selezione
        if has_outliers and max_scale_diff > 2:
            return 'robust'  # Dati con outlier e scale diverse
        elif max_scale_diff > 3:
            return 'standard'  # Scale molto diverse
        elif max_scale_diff > 1:
            return 'minmax'  # Scale moderatamente diverse
        else:
            return 'none'  # Dati già in scala simile
            
    except Exception:
        # Fallback sicuro
        return 'robust'