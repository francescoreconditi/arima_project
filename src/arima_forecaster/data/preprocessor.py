"""
Utility per il preprocessing di serie temporali e variabili esogene.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from ..utils.logger import get_logger
from ..utils.exceptions import DataProcessingError


class TimeSeriesPreprocessor:
    """
    Utility complete per il preprocessing di serie temporali e variabili esogene.
    
    Supporta il preprocessing sia per serie temporali univariate che per variabili esogene
    utilizzate nei modelli SARIMAX, inclusi controlli di validazione e sincronizzazione.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.preprocessing_steps = []
    
    def handle_missing_values(
        self, 
        series: pd.Series, 
        method: str = 'interpolate'
    ) -> pd.Series:
        """
        Gestisce valori mancanti nelle serie temporali.
        
        Args:
            series: Serie temporale di input
            method: Metodo per gestire valori mancanti ('drop', 'interpolate', 'forward_fill', 'backward_fill')
            
        Returns:
            Serie con valori mancanti gestiti
        """
        self.logger.info(f"Gestione valori mancanti usando metodo: {method}")
        
        if method == 'drop':
            result = series.dropna()
        elif method == 'interpolate':
            result = series.interpolate(method='time')
        elif method == 'forward_fill':
            result = series.fillna(method='ffill')
        elif method == 'backward_fill':
            result = series.fillna(method='bfill')
        else:
            raise DataProcessingError(f"Metodo valori mancanti sconosciuto: {method}")
        
        self.preprocessing_steps.append(f"Valori mancanti gestiti: {method}")
        self.logger.info(f"Valori mancanti gestiti: {series.isnull().sum()} -> {result.isnull().sum()}")
        
        return result
    
    def remove_outliers(
        self, 
        series: pd.Series, 
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Rimuove outlier dalle serie temporali.
        
        Args:
            series: Serie temporale di input
            method: Metodo per rilevamento outlier ('iqr', 'zscore', 'modified_zscore')
            threshold: Soglia per rilevamento outlier
            
        Returns:
            Serie con outlier rimossi
        """
        self.logger.info(f"Rimozione outliers usando metodo: {method}")
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (series >= lower_bound) & (series <= upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            mask = z_scores < threshold
            
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            mask = np.abs(modified_z_scores) < threshold
            
        else:
            raise DataProcessingError(f"Metodo rilevamento outlier sconosciuto: {method}")
        
        outliers_removed = (~mask).sum()
        result = series[mask]
        
        self.preprocessing_steps.append(f"Outlier rimossi: {outliers_removed} usando {method}")
        self.logger.info(f"Outliers rimossi: {outliers_removed}")
        
        return result
    
    def check_stationarity(
        self, 
        series: pd.Series,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Controlla se la serie temporale è stazionaria usando il test Augmented Dickey-Fuller.
        
        Args:
            series: Serie temporale di input
            alpha: Livello di significatività
            
        Returns:
            Dizionario con risultati del test di stazionarietà
        """
        self.logger.info("Verifica stazionarietà usando test ADF")
        
        try:
            # Rimuove valori infiniti o NaN per il test
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_series) < 10:
                raise DataProcessingError("Dati insufficienti per test di stazionarietà")
            
            adf_result = adfuller(clean_series)
            
            results = {
                'is_stationary': adf_result[1] < alpha,
                'p_value': adf_result[1],
                'adf_statistic': adf_result[0],
                'critical_values': adf_result[4],
                'used_lag': adf_result[2],
                'n_observations': adf_result[3]
            }
            
            status = "stazionaria" if results['is_stationary'] else "non-stazionaria"
            self.logger.info(f"Serie è {status} (p-value: {results['p_value']:.4f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Errore nel test di stazionarietà: {e}")
            raise DataProcessingError(f"Test di stazionarietà fallito: {e}")
    
    def make_stationary(
        self, 
        series: pd.Series,
        method: str = 'difference',
        max_diff: int = 2
    ) -> Tuple[pd.Series, int]:
        """
        Trasforma la serie per renderla stazionaria.
        
        Args:
            series: Serie temporale di input
            method: Metodo per rendere stazionaria ('difference', 'log_difference')
            max_diff: Numero massimo di differenze da applicare
            
        Returns:
            Tupla di (serie stazionaria, numero di differenze applicate)
        """
        self.logger.info(f"Rendendo serie stazionaria usando metodo: {method}")
        
        current_series = series.copy()
        n_diff = 0
        
        for i in range(max_diff + 1):
            stationarity_test = self.check_stationarity(current_series)
            
            if stationarity_test['is_stationary']:
                break
            
            if i == max_diff:
                self.logger.warning(f"Serie ancora non stazionaria dopo {max_diff} differenze")
                break
            
            if method == 'difference':
                current_series = current_series.diff().dropna()
            elif method == 'log_difference':
                if (current_series <= 0).any():
                    self.logger.warning("Impossibile applicare trasformazione logaritmica a valori non positivi, uso differenza regolare")
                    current_series = current_series.diff().dropna()
                else:
                    current_series = np.log(current_series).diff().dropna()
            else:
                raise DataProcessingError(f"Metodo di stazionarietà sconosciuto: {method}")
            
            n_diff += 1
            
        self.preprocessing_steps.append(f"Resa stazionaria: {n_diff} differenze applicate")
        self.logger.info(f"Applicate {n_diff} differenze per ottenere stazionarietà")
        
        return current_series, n_diff
    
    def preprocess_pipeline(
        self,
        series: pd.Series,
        handle_missing: bool = True,
        missing_method: str = 'interpolate',
        remove_outliers_flag: bool = False,
        outlier_method: str = 'iqr',
        make_stationary_flag: bool = True,
        stationarity_method: str = 'difference'
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Pipeline di preprocessing completa.
        
        Args:
            series: Serie temporale di input
            handle_missing: Se gestire valori mancanti
            missing_method: Metodo per gestire valori mancanti
            remove_outliers_flag: Se rimuovere outlier
            outlier_method: Metodo per rimozione outlier
            make_stationary_flag: Se rendere la serie stazionaria
            stationarity_method: Metodo per rendere stazionaria
            
        Returns:
            Tupla di (serie elaborata, metadati preprocessing)
        """
        self.logger.info("Avvio pipeline preprocessing")
        self.preprocessing_steps = []
        
        result = series.copy()
        metadata = {
            'original_length': len(series),
            'original_missing': series.isnull().sum()
        }
        
        # Gestisce valori mancanti
        if handle_missing and result.isnull().any():
            result = self.handle_missing_values(result, missing_method)
        
        # Rimuove outlier
        if remove_outliers_flag:
            original_length = len(result)
            result = self.remove_outliers(result, outlier_method)
            metadata['outliers_removed'] = original_length - len(result)
        
        # Rende stazionaria
        if make_stationary_flag:
            result, n_diff = self.make_stationary(result, stationarity_method)
            metadata['differencing_order'] = n_diff
        
        metadata['final_length'] = len(result)
        metadata['preprocessing_steps'] = self.preprocessing_steps.copy()
        
        self.logger.info(f"Preprocessing completato: {metadata['original_length']} -> {metadata['final_length']} osservazioni")
        
        return result, metadata
    
    def validate_exog_data(
        self,
        series: pd.Series,
        exog: pd.DataFrame,
        exog_names: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Valida le variabili esogene per l'uso con modelli SARIMAX.
        
        Args:
            series: Serie temporale target
            exog: DataFrame con variabili esogene
            exog_names: Lista attesa dei nomi delle variabili esogene
            
        Returns:
            Tupla di (validazione_ok, report_validazione)
        """
        self.logger.info("Validazione dati esogeni per modello SARIMAX")
        
        validation_report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Controllo base del DataFrame
            if not isinstance(exog, pd.DataFrame):
                validation_report['valid'] = False
                validation_report['errors'].append("Le variabili esogene devono essere un pandas DataFrame")
                return False, validation_report
            
            if exog.empty:
                validation_report['valid'] = False
                validation_report['errors'].append("Il DataFrame delle variabili esogene è vuoto")
                return False, validation_report
            
            # Controllo allineamento lunghezze
            if len(exog) != len(series):
                validation_report['valid'] = False
                validation_report['errors'].append(
                    f"Lunghezza variabili esogene ({len(exog)}) diversa da serie target ({len(series)})"
                )
                return False, validation_report
            
            # Controllo nomi colonne se specificati
            if exog_names is not None:
                missing_cols = set(exog_names) - set(exog.columns)
                extra_cols = set(exog.columns) - set(exog_names)
                
                if missing_cols:
                    validation_report['valid'] = False
                    validation_report['errors'].append(f"Variabili esogene mancanti: {missing_cols}")
                
                if extra_cols:
                    validation_report['warnings'].append(f"Variabili esogene extra non richieste: {extra_cols}")
            
            # Controllo indici temporali
            if not series.index.equals(exog.index):
                validation_report['warnings'].append("Indici di serie target e variabili esogene non corrispondono")
            
            # Controllo valori mancanti
            missing_stats = {}
            for col in exog.columns:
                missing_count = exog[col].isnull().sum()
                missing_pct = missing_count / len(exog) * 100
                missing_stats[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_pct, 2)
                }
                
                if missing_count > 0:
                    if missing_pct > 50:
                        validation_report['errors'].append(
                            f"Variabile esogena '{col}' ha {missing_pct:.1f}% valori mancanti (soglia critica)"
                        )
                        validation_report['valid'] = False
                    elif missing_pct > 10:
                        validation_report['warnings'].append(
                            f"Variabile esogena '{col}' ha {missing_pct:.1f}% valori mancanti"
                        )
            
            validation_report['statistics']['missing_values'] = missing_stats
            
            # Controllo tipi di dati
            non_numeric_cols = exog.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                validation_report['warnings'].append(
                    f"Variabili esogene non numeriche (potrebbero richiedere encoding): {non_numeric_cols}"
                )
            
            # Statistiche descrittive
            numeric_cols = exog.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats_summary = {}
                for col in numeric_cols:
                    col_stats = exog[col].describe()
                    stats_summary[col] = {
                        'mean': float(col_stats['mean']) if pd.notna(col_stats['mean']) else None,
                        'std': float(col_stats['std']) if pd.notna(col_stats['std']) else None,
                        'min': float(col_stats['min']) if pd.notna(col_stats['min']) else None,
                        'max': float(col_stats['max']) if pd.notna(col_stats['max']) else None,
                        'zero_variance': col_stats['std'] < 1e-10 if pd.notna(col_stats['std']) else True
                    }
                    
                    # Controllo varianza zero
                    if stats_summary[col]['zero_variance']:
                        validation_report['warnings'].append(
                            f"Variabile esogena '{col}' ha varianza quasi zero (potrebbe non essere informativa)"
                        )
                
                validation_report['statistics']['descriptive'] = stats_summary
            
            # Controllo correlazioni elevate tra variabili esogene
            if len(numeric_cols) > 1:
                corr_matrix = exog[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.9:
                            high_corr_pairs.append({
                                'var1': corr_matrix.columns[i],
                                'var2': corr_matrix.columns[j],
                                'correlation': round(corr_val, 3)
                            })
                
                if high_corr_pairs:
                    validation_report['warnings'].append(
                        f"Coppie di variabili esogene altamente correlate (>0.9): {high_corr_pairs}"
                    )
                
                validation_report['statistics']['high_correlations'] = high_corr_pairs
            
            # Riepilogo finale
            validation_report['statistics']['summary'] = {
                'n_variables': len(exog.columns),
                'n_observations': len(exog),
                'numeric_variables': len(numeric_cols),
                'non_numeric_variables': len(non_numeric_cols),
                'total_missing_cells': int(exog.isnull().sum().sum()),
                'missing_percentage_overall': round(exog.isnull().sum().sum() / (len(exog) * len(exog.columns)) * 100, 2)
            }
            
            n_errors = len(validation_report['errors'])
            n_warnings = len(validation_report['warnings'])
            
            self.logger.info(f"Validazione completata: {n_errors} errori, {n_warnings} avvisi")
            
            return validation_report['valid'], validation_report
            
        except Exception as e:
            self.logger.error(f"Errore durante validazione variabili esogene: {e}")
            validation_report['valid'] = False
            validation_report['errors'].append(f"Errore di validazione: {str(e)}")
            return False, validation_report
    
    def preprocess_exog_data(
        self,
        exog: pd.DataFrame,
        handle_missing: bool = True,
        missing_method: str = 'interpolate',
        remove_outliers_flag: bool = False,
        outlier_method: str = 'iqr',
        scale_data: bool = False,
        scaling_method: str = 'standardize'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocessa le variabili esogene per l'uso con modelli SARIMAX.
        
        Args:
            exog: DataFrame con variabili esogene
            handle_missing: Se gestire valori mancanti
            missing_method: Metodo per gestire valori mancanti
            remove_outliers_flag: Se rimuovere outlier
            outlier_method: Metodo per rimozione outlier
            scale_data: Se scalare i dati
            scaling_method: Metodo di scaling ('standardize', 'normalize', 'minmax')
            
        Returns:
            Tupla di (DataFrame elaborato, metadati preprocessing)
        """
        self.logger.info("Avvio preprocessing variabili esogene")
        
        result = exog.copy()
        metadata = {
            'original_shape': exog.shape,
            'original_dtypes': exog.dtypes.to_dict(),
            'processing_steps': []
        }
        
        # Gestisce valori mancanti per ogni colonna
        if handle_missing:
            missing_before = result.isnull().sum().sum()
            
            for col in result.columns:
                if result[col].isnull().any():
                    if result[col].dtype in [np.number]:
                        # Variabili numeriche
                        if missing_method == 'interpolate':
                            result[col] = result[col].interpolate(method='linear')
                        elif missing_method == 'forward_fill':
                            result[col] = result[col].fillna(method='ffill')
                        elif missing_method == 'backward_fill':
                            result[col] = result[col].fillna(method='bfill')
                        elif missing_method == 'mean':
                            result[col] = result[col].fillna(result[col].mean())
                        elif missing_method == 'median':
                            result[col] = result[col].fillna(result[col].median())
                    else:
                        # Variabili categoriche
                        if missing_method in ['forward_fill', 'ffill']:
                            result[col] = result[col].fillna(method='ffill')
                        elif missing_method in ['backward_fill', 'bfill']:
                            result[col] = result[col].fillna(method='bfill')
                        else:
                            # Usa la moda per variabili categoriche
                            mode_value = result[col].mode()
                            if not mode_value.empty:
                                result[col] = result[col].fillna(mode_value.iloc[0])
            
            missing_after = result.isnull().sum().sum()
            metadata['processing_steps'].append(f"Valori mancanti gestiti: {missing_before} -> {missing_after}")
            
            if missing_after > 0:
                self.logger.warning(f"Ancora {missing_after} valori mancanti dopo preprocessing")
        
        # Rimuovi outlier per variabili numeriche
        if remove_outliers_flag:
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            outliers_removed_total = 0
            
            for col in numeric_cols:
                original_length = len(result)
                
                if outlier_method == 'iqr':
                    Q1 = result[col].quantile(0.25)
                    Q3 = result[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    mask = (result[col] >= lower_bound) & (result[col] <= upper_bound)
                elif outlier_method == 'zscore':
                    z_scores = np.abs(stats.zscore(result[col].dropna()))
                    mask = z_scores < 3.0
                else:
                    continue  # Salta se metodo non supportato per variabili esogene
                
                # Invece di rimuovere righe, sostituisce outlier con valori limite
                result.loc[result[col] < lower_bound, col] = lower_bound
                result.loc[result[col] > upper_bound, col] = upper_bound
                
                outliers_capped = original_length - mask.sum()
                outliers_removed_total += outliers_capped
            
            if outliers_removed_total > 0:
                metadata['processing_steps'].append(f"Outliers gestiti (capping): {outliers_removed_total}")
        
        # Scaling dei dati numerici
        if scale_data:
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            scaling_metadata = {}
            
            for col in numeric_cols:
                original_values = result[col].copy()
                
                if scaling_method == 'standardize':
                    mean_val = result[col].mean()
                    std_val = result[col].std()
                    if std_val > 0:
                        result[col] = (result[col] - mean_val) / std_val
                        scaling_metadata[col] = {'method': 'standardize', 'mean': mean_val, 'std': std_val}
                
                elif scaling_method == 'normalize':
                    norm = np.linalg.norm(result[col])
                    if norm > 0:
                        result[col] = result[col] / norm
                        scaling_metadata[col] = {'method': 'normalize', 'norm': norm}
                
                elif scaling_method == 'minmax':
                    min_val = result[col].min()
                    max_val = result[col].max()
                    if max_val > min_val:
                        result[col] = (result[col] - min_val) / (max_val - min_val)
                        scaling_metadata[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
            metadata['scaling_parameters'] = scaling_metadata
            metadata['processing_steps'].append(f"Dati scalati usando: {scaling_method}")
        
        metadata['final_shape'] = result.shape
        metadata['final_dtypes'] = result.dtypes.to_dict()
        metadata['final_missing'] = result.isnull().sum().to_dict()
        
        self.logger.info(f"Preprocessing esogeni completato: shape {metadata['original_shape']} -> {metadata['final_shape']}")
        
        return result, metadata
    
    def preprocess_pipeline_with_exog(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        handle_missing: bool = True,
        missing_method: str = 'interpolate',
        remove_outliers_flag: bool = False,
        outlier_method: str = 'iqr',
        make_stationary_flag: bool = True,
        stationarity_method: str = 'difference',
        scale_exog: bool = False,
        exog_scaling_method: str = 'standardize',
        validate_exog: bool = True
    ) -> Tuple[pd.Series, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Pipeline di preprocessing completa per serie temporale e variabili esogene.
        
        Args:
            series: Serie temporale target
            exog: DataFrame con variabili esogene (opzionale)
            handle_missing: Se gestire valori mancanti
            missing_method: Metodo per gestire valori mancanti
            remove_outliers_flag: Se rimuovere outlier
            outlier_method: Metodo per rimozione outlier
            make_stationary_flag: Se rendere la serie stazionaria
            stationarity_method: Metodo per rendere stazionaria
            scale_exog: Se scalare le variabili esogene
            exog_scaling_method: Metodo di scaling per variabili esogene
            validate_exog: Se validare le variabili esogene
            
        Returns:
            Tupla di (serie elaborata, variabili esogene elaborate, metadati completi)
        """
        self.logger.info("Avvio pipeline preprocessing combinata (serie + esogeni)")
        
        # Preprocessa la serie temporale target
        processed_series, series_metadata = self.preprocess_pipeline(
            series=series,
            handle_missing=handle_missing,
            missing_method=missing_method,
            remove_outliers_flag=remove_outliers_flag,
            outlier_method=outlier_method,
            make_stationary_flag=make_stationary_flag,
            stationarity_method=stationarity_method
        )
        
        processed_exog = None
        exog_metadata = {}
        
        # Preprocessa le variabili esogene se fornite
        if exog is not None:
            # Validazione preliminare
            if validate_exog:
                is_valid, validation_report = self.validate_exog_data(series, exog)
                exog_metadata['validation'] = validation_report
                
                if not is_valid:
                    critical_errors = [err for err in validation_report['errors'] if 'critica' in err.lower()]
                    if critical_errors:
                        raise DataProcessingError(f"Errori critici nelle variabili esogene: {critical_errors}")
            
            # Allinea le variabili esogene con la serie processata
            # Nota: questo è complesso perché la serie potrebbe essere stata accorciata
            if len(processed_series) != len(series):
                # La serie è stata accorciata (es. per differencing), allinea exog
                start_idx = len(series) - len(processed_series)
                aligned_exog = exog.iloc[start_idx:].copy()
                aligned_exog.index = processed_series.index
            else:
                aligned_exog = exog.copy()
                aligned_exog.index = processed_series.index
            
            # Preprocessa le variabili esogene allineate
            processed_exog, exog_proc_metadata = self.preprocess_exog_data(
                exog=aligned_exog,
                handle_missing=handle_missing,
                missing_method=missing_method,
                remove_outliers_flag=False,  # Non rimuovere righe dalle esogene
                scale_data=scale_exog,
                scaling_method=exog_scaling_method
            )
            
            exog_metadata.update(exog_proc_metadata)
        
        # Metadati completi
        combined_metadata = {
            'series_processing': series_metadata,
            'exog_processing': exog_metadata,
            'alignment': {
                'series_length_change': len(series) - len(processed_series),
                'exog_aligned': exog is not None,
                'final_alignment_check': (
                    len(processed_series) == len(processed_exog) if processed_exog is not None 
                    else True
                )
            }
        }
        
        self.logger.info("Pipeline preprocessing combinata completata")
        
        return processed_series, processed_exog, combined_metadata