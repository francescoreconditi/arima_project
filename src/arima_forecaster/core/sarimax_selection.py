"""
Selezione automatica del modello SARIMAX con ottimizzazione dei parametri stagionali e variabili esogene.
"""

import pandas as pd
import numpy as np
import itertools
from typing import List, Tuple, Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from tqdm import tqdm

from .sarimax_model import SARIMAXForecaster
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError


class SARIMAXModelSelector:
    """
    Selezione automatica del modello SARIMAX usando grid search con parametri stagionali e variabili esogene.
    
    Questo selettore permette di trovare automaticamente i migliori parametri per un modello SARIMAX,
    considerando sia i parametri tradizionali SARIMA sia l'inclusione delle variabili esogene.
    """
    
    def __init__(
        self,
        p_range: Tuple[int, int] = (0, 3),
        d_range: Tuple[int, int] = (0, 2),  
        q_range: Tuple[int, int] = (0, 3),
        P_range: Tuple[int, int] = (0, 2),
        D_range: Tuple[int, int] = (0, 1),
        Q_range: Tuple[int, int] = (0, 2),
        seasonal_periods: Optional[List[int]] = None,
        exog_names: Optional[List[str]] = None,
        information_criterion: str = 'aic',
        max_models: Optional[int] = None,
        n_jobs: int = 1
    ):
        """
        Inizializza il selettore di modelli SARIMAX.
        
        Args:
            p_range: Intervallo valori p (min, max)
            d_range: Intervallo valori d (min, max)
            q_range: Intervallo valori q (min, max)  
            P_range: Intervallo valori P stagionali (min, max)
            D_range: Intervallo valori D stagionali (min, max)
            Q_range: Intervallo valori Q stagionali (min, max)
            seasonal_periods: Lista di periodi stagionali da provare (default: [12])
            exog_names: Lista dei nomi delle variabili esogene
            information_criterion: Criterio per la selezione del modello ('aic', 'bic', 'hqic')
            max_models: Numero massimo di modelli da provare
            n_jobs: Numero di job paralleli per l'addestramento del modello
        """
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.P_range = P_range
        self.D_range = D_range
        self.Q_range = Q_range
        self.seasonal_periods = seasonal_periods or [12]
        self.exog_names = exog_names or []
        self.information_criterion = information_criterion.lower()
        self.max_models = max_models
        self.n_jobs = n_jobs
        
        self.results = []
        self.best_model = None
        self.best_order = None
        self.best_seasonal_order = None
        self.logger = get_logger(__name__)
        
        if self.information_criterion not in ['aic', 'bic', 'hqic']:
            raise ValueError("information_criterion deve essere 'aic', 'bic', o 'hqic'")
    
    def search(
        self, 
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        verbose: bool = True,
        suppress_warnings: bool = True
    ) -> 'SARIMAXModelSelector':
        """
        Esegue grid search per trovare i parametri SARIMAX ottimali.
        
        Args:
            series: Serie temporale da addestrare
            exog: DataFrame con variabili esogene (stesse righe di series)
            verbose: Se mostrare il progresso
            suppress_warnings: Se sopprimere i warning di statsmodels
            
        Returns:
            Self per concatenamento dei metodi
        """
        if suppress_warnings:
            warnings.filterwarnings("ignore")
        
        try:
            model_type = "SARIMAX" if exog is not None else "SARIMA"
            exog_info = f" con {exog.shape[1]} variabili esogene" if exog is not None else ""
            
            self.logger.info(f"Avvio selezione modello {model_type} usando {self.information_criterion.upper()}{exog_info}")
            self.logger.info(f"Intervalli parametri: p{self.p_range}, d{self.d_range}, q{self.q_range}")
            self.logger.info(f"Intervalli stagionali: P{self.P_range}, D{self.D_range}, Q{self.Q_range}")
            self.logger.info(f"Periodi stagionali: {self.seasonal_periods}")
            
            if exog is not None:
                self.logger.info(f"Variabili esogene: {list(exog.columns)}")
                # Aggiorna nomi variabili esogene se non specificati
                if not self.exog_names:
                    self.exog_names = list(exog.columns)
            
            # Genera tutte le combinazioni di parametri
            param_combinations = self._generate_param_combinations()
            
            if self.max_models and len(param_combinations) > self.max_models:
                # Campiona casualmente se troppe combinazioni
                np.random.shuffle(param_combinations)
                param_combinations = param_combinations[:self.max_models]
                self.logger.info(f"Limitato a {self.max_models} combinazioni casuali di parametri")
            
            self.logger.info(f"Test di {len(param_combinations)} combinazioni di parametri")
            
            # Addestra modelli
            if self.n_jobs == 1:
                # Elaborazione sequenziale
                self.results = self._fit_models_sequential(series, exog, param_combinations, verbose)
            else:
                # Elaborazione parallela
                self.results = self._fit_models_parallel(series, exog, param_combinations, verbose)
            
            # Trova miglior modello
            if self.results:
                best_result = min(self.results, key=lambda x: x[self.information_criterion])
                self.best_order = best_result['order']
                self.best_seasonal_order = best_result['seasonal_order']
                
                # Addestra il miglior modello
                self.best_model = SARIMAXForecaster(
                    order=self.best_order,
                    seasonal_order=self.best_seasonal_order,
                    exog_names=self.exog_names
                )
                self.best_model.fit(series, exog=exog)
                
                self.logger.info(f"Miglior modello SARIMAX: {self.best_order}x{self.best_seasonal_order}")
                if exog is not None:
                    self.logger.info(f"Con variabili esogene: {', '.join(self.exog_names)}")
                self.logger.info(f"Best {self.information_criterion.upper()}: {best_result[self.information_criterion]:.2f}")
            else:
                self.logger.error("Nessun modello addestrato con successo")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Selezione modello SARIMAX fallita: {e}")
            raise ModelTrainingError(f"Selezione modello SARIMAX fallita: {e}")
        
        finally:
            if suppress_warnings:
                warnings.resetwarnings()
    
    def _generate_param_combinations(self) -> List[Tuple]:
        """Genera tutte le combinazioni di parametri da testare."""
        combinations = []
        
        for s in self.seasonal_periods:
            for p in range(self.p_range[0], self.p_range[1] + 1):
                for d in range(self.d_range[0], self.d_range[1] + 1):
                    for q in range(self.q_range[0], self.q_range[1] + 1):
                        for P in range(self.P_range[0], self.P_range[1] + 1):
                            for D in range(self.D_range[0], self.D_range[1] + 1):
                                for Q in range(self.Q_range[0], self.Q_range[1] + 1):
                                    combinations.append(((p, d, q), (P, D, Q, s)))
        
        return combinations
    
    def _fit_models_sequential(
        self, 
        series: pd.Series,
        exog: Optional[pd.DataFrame], 
        param_combinations: List[Tuple],
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """Addestra modelli sequenzialmente."""
        results = []
        
        iterator = tqdm(param_combinations, desc="Test modelli SARIMAX") if verbose else param_combinations
        
        for order, seasonal_order in iterator:
            try:
                model = SARIMAXForecaster(
                    order=order, 
                    seasonal_order=seasonal_order,
                    exog_names=self.exog_names
                )
                model.fit(series, exog=exog, validate_input=False)
                
                model_info = model.get_model_info()
                results.append(model_info)
                
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        'Best AIC': min(r['aic'] for r in results),
                        'Current': f"{order}x{seasonal_order}"
                    })
                    
            except Exception as e:
                # Modello fallito nell'addestramento, saltalo
                continue
        
        return results
    
    def _fit_models_parallel(
        self, 
        series: pd.Series,
        exog: Optional[pd.DataFrame], 
        param_combinations: List[Tuple],
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """Addestra modelli in parallelo."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Invia tutti i job
            future_to_params = {
                executor.submit(_fit_sarimax_model, series, exog, order, seasonal_order, self.exog_names): (order, seasonal_order)
                for order, seasonal_order in param_combinations
            }
            
            # Raccogli risultati
            iterator = tqdm(
                as_completed(future_to_params), 
                total=len(param_combinations),
                desc="Test modelli SARIMAX"
            ) if verbose else as_completed(future_to_params)
            
            for future in iterator:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        
                        if verbose and hasattr(iterator, 'set_postfix'):
                            iterator.set_postfix({
                                'Completed': len(results),
                                'Best AIC': min(r['aic'] for r in results) if results else 'N/A'
                            })
                except Exception:
                    # Modello fallito nell'addestramento, saltalo
                    continue
        
        return results
    
    def get_best_model(self) -> Optional[SARIMAXForecaster]:
        """
        Ottieni il miglior modello SARIMAX addestrato.
        
        Returns:
            Istanza SARIMAXForecaster migliore o None se nessun modello addestrato
        """
        return self.best_model
    
    def get_results_summary(self, top_n: int = 10) -> pd.DataFrame:
        """
        Ottieni riepilogo dei risultati di selezione del modello.
        
        Args:
            top_n: Numero di migliori modelli da restituire
            
        Returns:
            DataFrame con risultati dei modelli
        """
        if not self.results:
            return pd.DataFrame()
        
        # Ordina per criterio di informazione
        sorted_results = sorted(self.results, key=lambda x: x[self.information_criterion])
        
        # Crea DataFrame di riepilogo
        summary_data = []
        for result in sorted_results[:top_n]:
            row = {
                'order': str(result['order']),
                'seasonal_order': str(result['seasonal_order']),
                'aic': result['aic'],
                'bic': result['bic'],
                'hqic': result['hqic'],
                'n_observations': result['n_observations']
            }
            
            # Aggiungi informazioni sulle variabili esogene se presenti
            if 'n_exog' in result and result['n_exog'] > 0:
                row['n_exog'] = result['n_exog']
                row['exog_names'] = ', '.join(result.get('exog_names', []))
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_exog_analysis(self) -> Optional[pd.DataFrame]:
        """
        Ottieni analisi dell'importanza delle variabili esogene per tutti i modelli testati.
        
        Returns:
            DataFrame con statistiche delle variabili esogene per i modelli testati
        """
        if not self.results or not self.exog_names:
            return None
        
        analysis_data = []
        
        for result in self.results:
            if 'exog_params' in result:
                for var_name, params in result['exog_params'].items():
                    analysis_data.append({
                        'model': f"{result['order']}x{result['seasonal_order']}",
                        'variable': var_name,
                        'coefficient': params.get('coefficient', np.nan),
                        'pvalue': params.get('pvalue', np.nan),
                        'significant': params.get('pvalue', 1.0) < 0.05,
                        'aic': result['aic'],
                        'bic': result['bic']
                    })
        
        if not analysis_data:
            return None
        
        df = pd.DataFrame(analysis_data)
        
        # Aggiungi statistiche aggregate per variabile
        summary_stats = df.groupby('variable').agg({
            'coefficient': ['mean', 'std', 'count'],
            'pvalue': ['mean', 'min'],
            'significant': 'mean'
        }).round(4)
        
        return df.sort_values(['variable', 'aic'])
    
    def plot_selection_results(self, top_n: int = 20) -> None:
        """
        Traccia i risultati della selezione del modello.
        
        Args:
            top_n: Numero di migliori modelli da tracciare
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results:
                self.logger.warning("Nessun risultato da tracciare")
                return
            
            # Ottieni migliori risultati
            sorted_results = sorted(self.results, key=lambda x: x[self.information_criterion])[:top_n]
            
            # Prepara dati
            model_names = []
            for r in sorted_results:
                name = f"{r['order']}x{r['seasonal_order']}"
                if r.get('n_exog', 0) > 0:
                    name += f" +{r['n_exog']}exog"
                model_names.append(name)
            
            aic_values = [r['aic'] for r in sorted_results]
            bic_values = [r['bic'] for r in sorted_results]
            hqic_values = [r['hqic'] for r in sorted_results]
            
            # Crea grafico
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Grafico principale con criteri di informazione
            x = np.arange(len(model_names))
            width = 0.25
            
            ax1.bar(x - width, aic_values, width, label='AIC', alpha=0.8)
            ax1.bar(x, bic_values, width, label='BIC', alpha=0.8)  
            ax1.bar(x + width, hqic_values, width, label='HQIC', alpha=0.8)
            
            ax1.set_xlabel('Modelli SARIMAX')
            ax1.set_ylabel('Valore Criterio di Informazione')
            ax1.set_title(f'Confronto Top {top_n} Modelli SARIMAX')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Grafico delle variabili esogene se presenti
            if self.exog_names and any(r.get('n_exog', 0) > 0 for r in sorted_results):
                exog_analysis = self.get_exog_analysis()
                if exog_analysis is not None:
                    # Grafico significatività variabili esogene
                    var_significance = exog_analysis.groupby('variable')['significant'].mean().sort_values(ascending=False)
                    
                    ax2.bar(range(len(var_significance)), var_significance.values, alpha=0.7)
                    ax2.set_xlabel('Variabili Esogene')
                    ax2.set_ylabel('% Modelli con Significatività (p<0.05)')
                    ax2.set_title('Significatività delle Variabili Esogene')
                    ax2.set_xticks(range(len(var_significance)))
                    ax2.set_xticklabels(var_significance.index, rotation=45, ha='right')
                    ax2.set_ylim(0, 1)
                    ax2.grid(axis='y', alpha=0.3)
                    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% soglia')
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'Nessuna analisi variabili esogene disponibile', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Analisi Variabili Esogene')
            else:
                ax2.text(0.5, 0.5, 'Nessuna variabile esogena utilizzata', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Analisi Variabili Esogene')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn non disponibili per i grafici")
        except Exception as e:
            self.logger.error(f"Impossibile creare grafico: {e}")


def _fit_sarimax_model(
    series: pd.Series,
    exog: Optional[pd.DataFrame],
    order: Tuple[int, int, int], 
    seasonal_order: Tuple[int, int, int, int],
    exog_names: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Funzione helper per addestrare un singolo modello SARIMAX (per elaborazione parallela).
    
    Args:
        series: Dati della serie temporale
        exog: DataFrame con variabili esogene
        order: Ordine ARIMA
        seasonal_order: Ordine ARIMA stagionale
        exog_names: Lista nomi variabili esogene
        
    Returns:
        Dizionario informazioni modello o None se l'addestramento è fallito
    """
    try:
        model = SARIMAXForecaster(
            order=order, 
            seasonal_order=seasonal_order,
            exog_names=exog_names
        )
        model.fit(series, exog=exog, validate_input=False)
        return model.get_model_info()
    except Exception:
        return None