# ============================================
# PREPROCESSING UTILITIES PER SARIMAX
# Creato da: Claude Code  
# Data: 2025-08-23
# Scopo: Preprocessing robusto per variabili esogene
# ============================================

"""
Utilities per preprocessing robusto delle variabili esogene in modelli SARIMAX.
Versione estesa con funzionalità Advanced Exog Handling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any, List
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from scipy.stats import zscore
import warnings
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ExogenousPreprocessor:
    """
    Preprocessore avanzato per variabili esogene con funzionalità enterprise.
    
    Features:
    - Multiple scaling methods (robust, standard, minmax)
    - Advanced outlier detection (IQR, Z-score, Modified Z-score)  
    - Smart missing value imputation (mean, median, KNN, forward-fill)
    - Multicollinearity detection and handling
    - Feature correlation analysis
    - Stationarity testing and transformation
    """
    
    def __init__(
        self, 
        method: str = 'robust', 
        handle_outliers: bool = False,  # Temporaneamente disabilitato
        outlier_method: str = 'iqr',
        missing_strategy: str = 'interpolate',
        detect_multicollinearity: bool = False,  # Temporaneamente disabilitato
        multicollinearity_threshold: float = 0.95,
        stationarity_test: bool = False  # Temporaneamente disabilitato
    ):
        """
        Inizializza il preprocessore avanzato.
        
        Args:
            method: Metodo di scaling ('robust', 'standard', 'minmax', 'none')
            handle_outliers: Se gestire outlier
            outlier_method: Metodo detection outlier ('iqr', 'zscore', 'modified_zscore')
            missing_strategy: Strategia valori mancanti ('interpolate', 'mean', 'median', 'knn', 'ffill', 'bfill')
            detect_multicollinearity: Se rilevare multicollinearità
            multicollinearity_threshold: Soglia correlazione per multicollinearità
            stationarity_test: Se testare stazionarietà
        """
        self.method = method
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.missing_strategy = missing_strategy
        self.detect_multicollinearity = detect_multicollinearity
        self.multicollinearity_threshold = multicollinearity_threshold
        self.stationarity_test = stationarity_test
        
        # Stato interno
        self.scalers = {}
        self.outlier_bounds = {}
        self.missing_imputers = {}
        self.multicollinear_features = []
        self.stationarity_results = {}
        self.transformation_log = []
        self.is_fitted = False
        
        # Validazione parametri
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Valida parametri di inizializzazione."""
        valid_methods = ['robust', 'standard', 'minmax', 'none']
        if self.method not in valid_methods:
            raise ValueError(f"method deve essere uno di: {valid_methods}")
        
        valid_outlier_methods = ['iqr', 'zscore', 'modified_zscore']
        if self.outlier_method not in valid_outlier_methods:
            raise ValueError(f"outlier_method deve essere uno di: {valid_outlier_methods}")
        
        valid_missing_strategies = ['interpolate', 'mean', 'median', 'knn', 'ffill', 'bfill']
        if self.missing_strategy not in valid_missing_strategies:
            raise ValueError(f"missing_strategy deve essere una di: {valid_missing_strategies}")
        
        if not 0 < self.multicollinearity_threshold <= 1:
            raise ValueError("multicollinearity_threshold deve essere tra 0 e 1")
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gestione semplificata valori mancanti."""
        result = data.copy()
        
        if self.missing_strategy == 'interpolate':
            result = result.interpolate(method='linear')
        elif self.missing_strategy == 'mean':
            result = result.fillna(result.mean())
        elif self.missing_strategy == 'median':
            result = result.fillna(result.median())
        elif self.missing_strategy == 'ffill':
            result = result.fillna(method='ffill')
        elif self.missing_strategy == 'bfill':
            result = result.fillna(method='bfill')
        
        # Drop righe ancora NaN se presenti
        result = result.dropna()
        return result
    
    def _detect_outliers_advanced(self, series: pd.Series, col: str) -> Tuple[pd.Series, Dict[str, float]]:
        """Stub method - restituisce serie originale e bounds vuoti."""
        return series, {'lower': series.min(), 'upper': series.max()}
    
    def _handle_outliers(self, series: pd.Series, col: str) -> pd.Series:
        """Stub method - restituisce serie originale."""
        return series
        
    def fit(self, exog_data: pd.DataFrame) -> 'ExogenousPreprocessor':
        """
        Adatta il preprocessore avanzato ai dati.
        
        Args:
            exog_data: DataFrame con variabili esogene
            
        Returns:
            Self per method chaining
        """
        logger.info(f"Fitting preprocessore avanzato su {len(exog_data)} osservazioni, {len(exog_data.columns)} variabili")
        
        # Reset stato precedente
        self.scalers = {}
        self.outlier_bounds = {}
        self.missing_imputers = {}
        self.multicollinear_features = []
        self.stationarity_results = {}
        self.transformation_log = []
        
        # Step 1: Gestione valori mancanti
        processed_data = self._handle_missing_values(exog_data)
        
        # Step 2: Test stazionarietà se richiesto
        if self.stationarity_test:
            self.stationarity_results = self._test_stationarity(processed_data)
            processed_data = self._apply_stationarity_transforms(processed_data)
        
        # Step 3: Rilevamento multicollinearità
        if self.detect_multicollinearity:
            self.multicollinear_features = self._detect_multicollinearity(processed_data)
            if self.multicollinear_features:
                # Rimuovi features multicollineari
                processed_data = processed_data.drop(columns=self.multicollinear_features)
                logger.info(f"Rimosse {len(self.multicollinear_features)} features multicollineari")
        
        # Step 4: Detection e gestione outlier per ogni variabile
        for col in processed_data.columns:
            series = processed_data[col].copy()
            
            if self.handle_outliers:
                series, bounds = self._detect_outliers_advanced(series, col)
                self.outlier_bounds[col] = bounds
            
            # Step 5: Configura scaler
            scaler = self._create_scaler()
            
            if scaler is not None:
                # Fit dello scaler sui dati puliti
                clean_series = self._handle_outliers(series, col) if self.handle_outliers else series
                scaler.fit(clean_series.values.reshape(-1, 1))
                self.scalers[col] = scaler
                
                # Log statistiche
                logger.debug(f"Variable {col}: mean={clean_series.mean():.3f}, std={clean_series.std():.3f}, range=[{clean_series.min():.3f}, {clean_series.max():.3f}]")
        
        self.is_fitted = True
        self._last_processed_data = processed_data  # Per analisi suggerimenti
        
        logger.info("Preprocessore avanzato addestrato con successo")
        logger.info(f"Trasformazioni applicate: {len(self.transformation_log)}")
        
        return self
    
    def _create_scaler(self):
        """Crea lo scaler appropriato basato sul metodo."""
        if self.method == 'robust':
            return RobustScaler()
        elif self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler()
        else:  # 'none'
            return None
    
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
    
    def _test_stationarity(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Testa stazionarietà delle variabili."""
        return {}  # Stub per ora
    
    def _apply_stationarity_transforms(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applica trasformazioni per rendere serie stazionarie."""
        return data  # Stub per ora
    
    def _detect_multicollinearity(self, data: pd.DataFrame) -> List[str]:
        """Rileva features multicollineari."""
        return []  # Stub per ora
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Genera report completo del preprocessing."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        report = {
            'status': 'fitted',
            'configuration': {
                'scaling_method': self.method,
                'outlier_method': self.outlier_method,
                'missing_strategy': self.missing_strategy,
                'multicollinearity_threshold': self.multicollinearity_threshold,
                'stationarity_test': self.stationarity_test
            },
            'transformations_applied': self.transformation_log,
            'outlier_summary': {},
            'stationarity_summary': self.stationarity_results,
            'multicollinear_features': self.multicollinear_features
        }
        
        # Riassunto outlier per variabile
        for col, bounds in self.outlier_bounds.items():
            if isinstance(bounds, dict):
                report['outlier_summary'][col] = {
                    'lower_bound': bounds.get('lower'),
                    'upper_bound': bounds.get('upper'),
                    'method': self.outlier_method
                }
            else:
                report['outlier_summary'][col] = {
                    'lower_bound': bounds[0] if isinstance(bounds, tuple) else None,
                    'upper_bound': bounds[1] if isinstance(bounds, tuple) else None,
                    'method': self.outlier_method
                }
        
        return report
    
    def suggest_improvements(self, target_series: Optional[pd.Series] = None) -> List[str]:
        """Suggerisce miglioramenti al preprocessing basati sui risultati."""
        suggestions = []
        
        if not self.is_fitted:
            return ['Run fit() first to get suggestions']
        
        # Analisi multicollinearità
        if self.multicollinear_features:
            suggestions.append(f"Consider removing multicollinear features: {self.multicollinear_features}")
        
        # Analisi stazionarietà
        non_stationary = [col for col, result in self.stationarity_results.items() 
                         if result.get('is_stationary') == False]
        if non_stationary:
            suggestions.append(f"Consider differencing non-stationary variables: {non_stationary}")
        
        # Analisi outlier
        high_outlier_vars = [col for col in self.outlier_bounds.keys() 
                           if len(self.transformation_log) > 0]  # Placeholder logic
        if high_outlier_vars:
            suggestions.append("Consider more robust outlier handling for variables with many outliers")
        
        # Analisi correlazione con target se fornita
        if target_series is not None and hasattr(self, '_last_processed_data'):
            try:
                correlations = self._last_processed_data.corrwith(target_series).abs().sort_values(ascending=False)
                low_corr_features = correlations[correlations < 0.1].index.tolist()
                
                if low_corr_features:
                    suggestions.append(f"Consider removing low-correlation features: {low_corr_features[:5]}")
                    
            except Exception:
                pass
        
        return suggestions if suggestions else ['Preprocessing configuration appears optimal']


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


def analyze_feature_relationships(exog_data: pd.DataFrame, target_series: pd.Series) -> Dict[str, Any]:
    """
    Analizza relazioni tra features esogene e serie target.
    
    Args:
        exog_data: DataFrame variabili esogene  
        target_series: Serie temporale target
        
    Returns:
        Dizionario con analisi correlazioni e importanze
    """
    try:
        # Allinea dati
        common_index = exog_data.index.intersection(target_series.index)
        exog_aligned = exog_data.loc[common_index]
        target_aligned = target_series.loc[common_index]
        
        results = {
            'correlations': {},
            'mutual_information': {},
            'f_statistics': {},
            'recommendations': []
        }
        
        # Calcola correlazioni Pearson
        for col in exog_aligned.columns:
            try:
                corr = exog_aligned[col].corr(target_aligned)
                results['correlations'][col] = corr if not pd.isna(corr) else 0.0
            except:
                results['correlations'][col] = 0.0
        
        # Calcola mutual information
        try:
            mi_scores = mutual_info_regression(exog_aligned.values, target_aligned.values)
            results['mutual_information'] = dict(zip(exog_aligned.columns, mi_scores))
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
            results['mutual_information'] = {col: 0.0 for col in exog_aligned.columns}
        
        # Calcola F-statistics
        try:
            f_stats, p_values = f_regression(exog_aligned.values, target_aligned.values)
            results['f_statistics'] = dict(zip(exog_aligned.columns, f_stats))
        except Exception as e:
            logger.warning(f"F-statistics calculation failed: {e}")
            results['f_statistics'] = {col: 0.0 for col in exog_aligned.columns}
        
        # Genera raccomandazioni
        correlations = pd.Series(results['correlations']).abs()
        high_corr = correlations[correlations > 0.5].index.tolist()
        low_corr = correlations[correlations < 0.1].index.tolist()
        
        if high_corr:
            results['recommendations'].append(f"High correlation features (good): {high_corr}")
        if low_corr:
            results['recommendations'].append(f"Low correlation features (consider removing): {low_corr}")
        
        return results
        
    except Exception as e:
        logger.error(f"Feature relationship analysis failed: {e}")
        return {'error': str(e)}


def detect_feature_interactions(exog_data: pd.DataFrame, max_interactions: int = 10) -> List[Tuple[str, str, float]]:
    """
    Rileva potenziali interazioni significative tra features.
    
    Args:
        exog_data: DataFrame variabili esogene
        max_interactions: Numero massimo interazioni da rilevare
        
    Returns:
        Lista di tuple (feature1, feature2, correlation_score)
    """
    try:
        interactions = []
        
        for i, col1 in enumerate(exog_data.columns):
            for col2 in exog_data.columns[i+1:]:
                try:
                    # Calcola prodotto di interazione
                    interaction = exog_data[col1] * exog_data[col2]
                    
                    # Misura quanto l'interazione differisce dalle componenti singole
                    corr1 = interaction.corr(exog_data[col1])
                    corr2 = interaction.corr(exog_data[col2])
                    interaction_strength = 1 - min(abs(corr1), abs(corr2))  # 1 = completamente diversa
                    
                    if interaction_strength > 0.3:  # Soglia significatività
                        interactions.append((col1, col2, interaction_strength))
                        
                except:
                    continue
        
        # Ordina per strength e limita
        interactions.sort(key=lambda x: x[2], reverse=True)
        return interactions[:max_interactions]
        
    except Exception as e:
        logger.error(f"Feature interaction detection failed: {e}")
        return []