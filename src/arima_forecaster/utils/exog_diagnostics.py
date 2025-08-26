"""
Diagnostica avanzata per variabili esogene nei modelli SARIMAX.
Modulo per validazione, testing e analisi delle relazioni esogene.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import f_regression, chi2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExogDiagnostics:
    """
    Classe per diagnostica avanzata delle variabili esogene.
    
    Features:
    - Causalità di Granger tra exog e target
    - Test di stazionarietà completi (ADF, KPSS, Phillips-Perron)
    - Analisi correlazione avanzata (Pearson, Spearman, Kendall)
    - Test eteroschedasticità residui
    - Validazione assumptions SARIMAX
    - Generazione report diagnostici automatici
    """
    
    def __init__(self, max_lag: int = 12, significance_level: float = 0.05):
        """
        Inizializza il diagnostic engine.
        
        Args:
            max_lag: Numero massimo lag per test causalità
            significance_level: Livello significatività per test statistici
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.results = {}
        self.logger = get_logger(f"{__name__}.ExogDiagnostics")
        
    def full_diagnostic_suite(
        self,
        target_series: pd.Series,
        exog_data: pd.DataFrame,
        fitted_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Esegue suite completa di test diagnostici.
        
        Args:
            target_series: Serie temporale target
            exog_data: DataFrame variabili esogene
            fitted_model: Modello SARIMAX addestrato (per analisi residui)
            
        Returns:
            Dizionario con tutti i risultati diagnostici
        """
        self.logger.info("Iniziando suite diagnostica completa")
        
        results = {
            'summary': {},
            'stationarity': {},
            'causality': {},
            'correlation_analysis': {},
            'residual_analysis': {},
            'feature_importance': {},
            'recommendations': []
        }
        
        try:
            # Test stazionarietà
            results['stationarity'] = self.test_stationarity_comprehensive(target_series, exog_data)
            
            # Test causalità Granger
            results['causality'] = self.test_granger_causality_batch(target_series, exog_data)
            
            # Analisi correlazioni avanzate
            results['correlation_analysis'] = self.advanced_correlation_analysis(target_series, exog_data)
            
            # Feature importance
            results['feature_importance'] = self.calculate_feature_importance(target_series, exog_data)
            
            # Analisi residui se modello fornito
            if fitted_model is not None:
                results['residual_analysis'] = self.analyze_residuals(fitted_model, exog_data)
            
            # Genera raccomandazioni
            results['recommendations'] = self.generate_recommendations(results)
            
            # Summary generale
            results['summary'] = self.create_diagnostic_summary(results)
            
            self.results = results
            self.logger.info("Suite diagnostica completata")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Errore durante diagnostica: {e}")
            return {'error': str(e), 'partial_results': results}
    
    def test_stationarity_comprehensive(
        self, 
        target_series: pd.Series, 
        exog_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Test stazionarietà completi con multiple strategie."""
        self.logger.info("Testing stationarity per tutte le variabili")
        
        results = {}
        all_series = {'target': target_series}
        all_series.update({col: exog_data[col] for col in exog_data.columns})
        
        for name, series in all_series.items():
            results[name] = self._test_single_series_stationarity(series, name)
        
        return results
    
    def _test_single_series_stationarity(self, series: pd.Series, name: str) -> Dict[str, Any]:
        """Test stazionarietà per una singola serie."""
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return {'error': 'Insufficient data for stationarity testing'}
        
        results = {
            'series_name': name,
            'observations': len(series_clean),
            'tests': {}
        }
        
        try:
            # Augmented Dickey-Fuller Test
            adf_result = adfuller(series_clean, autolag='AIC')
            results['tests']['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < self.significance_level,
                'interpretation': 'stationary' if adf_result[1] < self.significance_level else 'non-stationary'
            }
            
            # KPSS Test (null hypothesis: stationary)
            kpss_result = kpss(series_clean, regression='c', nlags="auto")
            results['tests']['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > self.significance_level,  # Opposto di ADF
                'interpretation': 'stationary' if kpss_result[1] > self.significance_level else 'non-stationary'
            }
            
            # Consensus risultato
            adf_stationary = results['tests']['adf']['is_stationary']
            kpss_stationary = results['tests']['kpss']['is_stationary']
            
            if adf_stationary and kpss_stationary:
                consensus = 'stationary'
            elif not adf_stationary and not kpss_stationary:
                consensus = 'non-stationary'
            else:
                consensus = 'inconclusive'
            
            results['consensus'] = consensus
            results['recommendation'] = self._stationarity_recommendation(consensus)
            
        except Exception as e:
            results['error'] = str(e)
            self.logger.warning(f"Stationarity test failed for {name}: {e}")
        
        return results
    
    def test_granger_causality_batch(
        self, 
        target_series: pd.Series, 
        exog_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Test causalità Granger per tutte le variabili esogene."""
        self.logger.info(f"Testing Granger causality per {len(exog_data.columns)} variabili")
        
        results = {}
        
        for col in exog_data.columns:
            results[col] = self._test_granger_causality_single(target_series, exog_data[col], col)
        
        return results
    
    def _test_granger_causality_single(
        self, 
        target: pd.Series, 
        exog_var: pd.Series, 
        var_name: str
    ) -> Dict[str, Any]:
        """Test causalità Granger per una singola variabile esogena."""
        try:
            # Allinea le serie
            common_index = target.index.intersection(exog_var.index)
            target_aligned = target.loc[common_index].dropna()
            exog_aligned = exog_var.loc[common_index].dropna()
            
            # Rimuovi ulteriori NaN per allineamento perfetto
            valid_idx = target_aligned.index.intersection(exog_aligned.index)
            target_final = target_aligned.loc[valid_idx]
            exog_final = exog_aligned.loc[valid_idx]
            
            if len(target_final) < self.max_lag + 10:
                return {
                    'error': f'Insufficient data: {len(target_final)} observations for {self.max_lag} lags'
                }
            
            # Crea DataFrame per Granger test
            data = pd.DataFrame({
                'target': target_final,
                'exog': exog_final
            })
            
            # Test causalità per diversi lag
            max_lag_test = min(self.max_lag, len(data) // 4)
            granger_results = grangercausalitytests(
                data[['target', 'exog']], 
                maxlag=max_lag_test,
                verbose=False
            )
            
            # Estrai risultati
            best_lag = None
            best_p_value = 1.0
            significant_lags = []
            
            for lag in range(1, max_lag_test + 1):
                if lag in granger_results:
                    # F-test è il test principale
                    f_test = granger_results[lag][0]['ssr_ftest']
                    p_value = f_test[1]
                    
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_lag = lag
                    
                    if p_value < self.significance_level:
                        significant_lags.append(lag)
            
            return {
                'variable': var_name,
                'best_lag': best_lag,
                'best_p_value': best_p_value,
                'is_causal': best_p_value < self.significance_level,
                'significant_lags': significant_lags,
                'interpretation': 'causal' if best_p_value < self.significance_level else 'non-causal',
                'strength': self._causality_strength(best_p_value)
            }
            
        except Exception as e:
            self.logger.warning(f"Granger causality test failed for {var_name}: {e}")
            return {'error': str(e), 'variable': var_name}
    
    def advanced_correlation_analysis(
        self, 
        target_series: pd.Series, 
        exog_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Analisi correlazione avanzata con multiple metriche."""
        self.logger.info("Analyzing correlations con metriche multiple")
        
        results = {}
        
        for col in exog_data.columns:
            results[col] = self._correlation_analysis_single(target_series, exog_data[col], col)
        
        return results
    
    def _correlation_analysis_single(
        self, 
        target: pd.Series, 
        exog_var: pd.Series, 
        var_name: str
    ) -> Dict[str, Any]:
        """Analisi correlazione per singola variabile."""
        try:
            # Allinea serie
            common_index = target.index.intersection(exog_var.index)
            target_aligned = target.loc[common_index].dropna()
            exog_aligned = exog_var.loc[common_index].dropna()
            
            # Ulteriore allineamento
            valid_idx = target_aligned.index.intersection(exog_aligned.index)
            if len(valid_idx) < 10:
                return {'error': 'Insufficient aligned data'}
            
            target_final = target_aligned.loc[valid_idx]
            exog_final = exog_aligned.loc[valid_idx]
            
            results = {'variable': var_name}
            
            # Correlazione Pearson (lineare)
            pearson_r, pearson_p = pearsonr(target_final, exog_final)
            results['pearson'] = {
                'correlation': pearson_r,
                'p_value': pearson_p,
                'significant': pearson_p < self.significance_level,
                'strength': self._correlation_strength(abs(pearson_r))
            }
            
            # Correlazione Spearman (monotonica)
            spearman_r, spearman_p = spearmanr(target_final, exog_final)
            results['spearman'] = {
                'correlation': spearman_r,
                'p_value': spearman_p,
                'significant': spearman_p < self.significance_level,
                'strength': self._correlation_strength(abs(spearman_r))
            }
            
            # Correlazione Kendall (ordinale robusta)
            kendall_r, kendall_p = kendalltau(target_final, exog_final)
            results['kendall'] = {
                'correlation': kendall_r,
                'p_value': kendall_p,
                'significant': kendall_p < self.significance_level,
                'strength': self._correlation_strength(abs(kendall_r))
            }
            
            # Mutual Information (non-lineare)
            # Discretizza per mutual info
            target_discrete = pd.qcut(target_final, q=10, duplicates='drop')
            exog_discrete = pd.qcut(exog_final, q=10, duplicates='drop')
            
            mi_score = mutual_info_score(target_discrete, exog_discrete)
            results['mutual_information'] = {
                'score': mi_score,
                'strength': 'high' if mi_score > 0.5 else 'medium' if mi_score > 0.2 else 'low'
            }
            
            # Consensus sulla significatività
            significant_tests = sum([
                results['pearson']['significant'],
                results['spearman']['significant'], 
                results['kendall']['significant']
            ])
            
            results['consensus'] = {
                'significant': significant_tests >= 2,
                'relationship_type': self._infer_relationship_type(results),
                'overall_strength': self._overall_correlation_strength(results)
            }
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Correlation analysis failed for {var_name}: {e}")
            return {'error': str(e), 'variable': var_name}
    
    def calculate_feature_importance(
        self, 
        target_series: pd.Series, 
        exog_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Calcola importanza features con multiple metriche."""
        self.logger.info("Calculating feature importance")
        
        try:
            # Allinea tutti i dati
            common_index = target_series.index
            for col in exog_data.columns:
                common_index = common_index.intersection(exog_data[col].index)
            
            target_aligned = target_series.loc[common_index].dropna()
            exog_aligned = exog_data.loc[common_index].dropna()
            
            # Ulteriore filtro per NaN
            valid_idx = target_aligned.index.intersection(exog_aligned.index)
            if len(valid_idx) < 10:
                return {'error': 'Insufficient data for feature importance'}
            
            X = exog_aligned.loc[valid_idx].values
            y = target_aligned.loc[valid_idx].values
            
            results = {}
            
            # F-statistic importance
            f_stats, f_pvalues = f_regression(X, y)
            
            for i, col in enumerate(exog_aligned.columns):
                results[col] = {
                    'f_statistic': f_stats[i] if i < len(f_stats) else 0,
                    'f_p_value': f_pvalues[i] if i < len(f_pvalues) else 1,
                    'f_significant': (f_pvalues[i] < self.significance_level) if i < len(f_pvalues) else False
                }
            
            # Ranking features
            feature_scores = [(col, data['f_statistic']) for col, data in results.items()]
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'individual_features': results,
                'ranking': feature_scores,
                'top_features': [f[0] for f in feature_scores[:5]],
                'significant_features': [col for col, data in results.items() if data['f_significant']]
            }
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return {'error': str(e)}
    
    def analyze_residuals(self, fitted_model: Any, exog_data: pd.DataFrame) -> Dict[str, Any]:
        """Analizza residui del modello per validazione assumptions."""
        self.logger.info("Analyzing model residuals")
        
        try:
            if not hasattr(fitted_model, 'resid'):
                return {'error': 'Model does not have residuals'}
            
            residuals = fitted_model.resid
            results = {'residual_tests': {}}
            
            # Test normalità residui
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            results['residual_tests']['normality'] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'normal': shapiro_p > self.significance_level
            }
            
            # Test autocorrelazione residui (Durbin-Watson)
            dw_statistic = durbin_watson(residuals)
            results['residual_tests']['autocorrelation'] = {
                'durbin_watson': dw_statistic,
                'interpretation': self._interpret_durbin_watson(dw_statistic)
            }
            
            # Test eteroschedasticità se exog fornito
            if len(exog_data.columns) > 0:
                try:
                    # Allinea exog con residui
                    residual_index = residuals.index if hasattr(residuals, 'index') else range(len(residuals))
                    common_idx = exog_data.index.intersection(residual_index)
                    
                    if len(common_idx) > 10:
                        exog_aligned = exog_data.loc[common_idx].fillna(method='ffill')
                        residuals_aligned = residuals[residuals.index.intersection(common_idx)]
                        
                        # Breusch-Pagan test
                        bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals_aligned, exog_aligned.values)
                        
                        results['residual_tests']['heteroscedasticity'] = {
                            'breusch_pagan_statistic': bp_stat,
                            'breusch_pagan_p_value': bp_p,
                            'homoscedastic': bp_p > self.significance_level
                        }
                        
                except Exception as e:
                    results['residual_tests']['heteroscedasticity'] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Residual analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_recommendations(self, diagnostic_results: Dict[str, Any]) -> List[str]:
        """Genera raccomandazioni basate sui risultati diagnostici."""
        recommendations = []
        
        try:
            # Raccomandazioni stazionarietà
            if 'stationarity' in diagnostic_results:
                non_stationary = []
                for var, result in diagnostic_results['stationarity'].items():
                    if result.get('consensus') == 'non-stationary':
                        non_stationary.append(var)
                
                if non_stationary:
                    recommendations.append(f"Consider differencing non-stationary variables: {non_stationary}")
            
            # Raccomandazioni causalità
            if 'causality' in diagnostic_results:
                non_causal = []
                weak_causal = []
                
                for var, result in diagnostic_results['causality'].items():
                    if result.get('interpretation') == 'non-causal':
                        non_causal.append(var)
                    elif result.get('strength') == 'weak':
                        weak_causal.append(var)
                
                if non_causal:
                    recommendations.append(f"Consider removing non-causal variables: {non_causal[:3]}")
                if weak_causal:
                    recommendations.append(f"Monitor weak causal relationships: {weak_causal[:3]}")
            
            # Raccomandazioni correlazione
            if 'correlation_analysis' in diagnostic_results:
                weak_corr = []
                for var, result in diagnostic_results['correlation_analysis'].items():
                    if result.get('consensus', {}).get('overall_strength') == 'weak':
                        weak_corr.append(var)
                
                if weak_corr:
                    recommendations.append(f"Low correlation features may add noise: {weak_corr[:3]}")
            
            # Raccomandazioni residui
            if 'residual_analysis' in diagnostic_results:
                residual_tests = diagnostic_results['residual_analysis'].get('residual_tests', {})
                
                if residual_tests.get('normality', {}).get('normal') == False:
                    recommendations.append("Residuals are not normal - consider data transformation")
                
                if residual_tests.get('heteroscedasticity', {}).get('homoscedastic') == False:
                    recommendations.append("Heteroscedasticity detected - consider robust standard errors")
                    
                dw_interp = residual_tests.get('autocorrelation', {}).get('interpretation')
                if dw_interp and 'autocorrelation' in dw_interp:
                    recommendations.append("Autocorrelated residuals - consider additional lags or ARMA terms")
            
            if not recommendations:
                recommendations.append("Diagnostic results appear satisfactory")
                
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Unable to generate recommendations due to error")
        
        return recommendations
    
    def create_diagnostic_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Crea summary dei risultati diagnostici."""
        summary = {
            'overall_assessment': 'unknown',
            'key_findings': [],
            'critical_issues': [],
            'feature_count': 0,
            'tests_passed': 0,
            'tests_total': 0
        }
        
        try:
            # Count features
            if 'correlation_analysis' in results:
                summary['feature_count'] = len(results['correlation_analysis'])
            
            # Key findings
            if 'causality' in results:
                causal_vars = [var for var, res in results['causality'].items() 
                             if res.get('is_causal', False)]
                if causal_vars:
                    summary['key_findings'].append(f"Causal variables identified: {len(causal_vars)}")
            
            if 'feature_importance' in results:
                sig_features = results['feature_importance'].get('significant_features', [])
                if sig_features:
                    summary['key_findings'].append(f"Statistically significant features: {len(sig_features)}")
            
            # Critical issues
            if 'stationarity' in results:
                non_stat_count = sum(1 for res in results['stationarity'].values() 
                                   if res.get('consensus') == 'non-stationary')
                if non_stat_count > 0:
                    summary['critical_issues'].append(f"Non-stationary variables: {non_stat_count}")
            
            # Overall assessment
            critical_count = len(summary['critical_issues'])
            finding_count = len(summary['key_findings'])
            
            if critical_count == 0 and finding_count > 0:
                summary['overall_assessment'] = 'good'
            elif critical_count > 0 and finding_count > critical_count:
                summary['overall_assessment'] = 'acceptable'
            elif critical_count > finding_count:
                summary['overall_assessment'] = 'concerning'
            else:
                summary['overall_assessment'] = 'needs_investigation'
                
        except Exception as e:
            self.logger.error(f"Summary creation failed: {e}")
            summary['error'] = str(e)
        
        return summary
    
    # Helper methods
    def _stationarity_recommendation(self, consensus: str) -> str:
        """Raccomandazione basata su test stazionarietà."""
        if consensus == 'stationary':
            return 'Variable is stationary - suitable for ARIMA modeling'
        elif consensus == 'non-stationary':
            return 'Variable is non-stationary - consider differencing'
        else:
            return 'Stationarity is inconclusive - investigate further'
    
    def _causality_strength(self, p_value: float) -> str:
        """Classifica forza causalità basata su p-value."""
        if p_value < 0.001:
            return 'very_strong'
        elif p_value < 0.01:
            return 'strong'
        elif p_value < 0.05:
            return 'moderate'
        elif p_value < 0.1:
            return 'weak'
        else:
            return 'none'
    
    def _correlation_strength(self, abs_corr: float) -> str:
        """Classifica forza correlazione."""
        if abs_corr >= 0.7:
            return 'very_strong'
        elif abs_corr >= 0.5:
            return 'strong'
        elif abs_corr >= 0.3:
            return 'moderate'
        elif abs_corr >= 0.1:
            return 'weak'
        else:
            return 'very_weak'
    
    def _infer_relationship_type(self, corr_results: Dict[str, Any]) -> str:
        """Inferisce tipo di relazione dai test correlazione."""
        pearson_sig = corr_results.get('pearson', {}).get('significant', False)
        spearman_sig = corr_results.get('spearman', {}).get('significant', False)
        mi_high = corr_results.get('mutual_information', {}).get('strength') in ['high', 'medium']
        
        if pearson_sig and spearman_sig:
            return 'linear'
        elif spearman_sig and not pearson_sig:
            return 'monotonic'
        elif mi_high and not pearson_sig:
            return 'nonlinear'
        elif pearson_sig or spearman_sig:
            return 'weak_linear'
        else:
            return 'no_clear_relationship'
    
    def _overall_correlation_strength(self, corr_results: Dict[str, Any]) -> str:
        """Valuta forza correlazione complessiva."""
        strengths = []
        
        for test in ['pearson', 'spearman', 'kendall']:
            if test in corr_results and 'correlation' in corr_results[test]:
                abs_corr = abs(corr_results[test]['correlation'])
                strengths.append(abs_corr)
        
        if strengths:
            max_strength = max(strengths)
            return self._correlation_strength(max_strength)
        else:
            return 'unknown'
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpreta statistica Durbin-Watson."""
        if dw_stat < 1.5:
            return 'positive_autocorrelation'
        elif dw_stat > 2.5:
            return 'negative_autocorrelation'
        else:
            return 'no_autocorrelation'
    
    def save_diagnostic_report(
        self, 
        output_path: Union[str, Path], 
        format_type: str = 'json'
    ) -> Path:
        """Salva report diagnostico su file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        
        elif format_type == 'csv':
            # Salva summary come CSV
            if 'summary' in self.results:
                summary_df = pd.DataFrame([self.results['summary']])
                summary_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Diagnostic report saved to {output_path}")
        return output_path