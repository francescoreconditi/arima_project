"""
Selezione automatica del modello ARIMA e ottimizzazione degli iperparametri.
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import List, Tuple, Dict, Any, Optional
from statsmodels.tsa.arima.model import ARIMA
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError


class ARIMAModelSelector:
    """
    Selezione automatica del modello ARIMA usando grid search e criteri di informazione.
    """
    
    def __init__(
        self,
        p_range: Tuple[int, int] = (0, 3),
        d_range: Tuple[int, int] = (0, 2), 
        q_range: Tuple[int, int] = (0, 3),
        information_criterion: str = 'aic'
    ):
        """
        Inizializza il selettore di modelli.
        
        Args:
            p_range: Intervallo per l'ordine autoregressivo (min, max)
            d_range: Intervallo per l'ordine di differenziazione (min, max)
            q_range: Intervallo per l'ordine di media mobile (min, max)  
            information_criterion: Criterio per la selezione del modello ('aic', 'bic', 'hqic')
        """
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.information_criterion = information_criterion.lower()
        self.results = []
        self.best_order = None
        self.best_model = None
        self.logger = get_logger(__name__)
        
        if self.information_criterion not in ['aic', 'bic', 'hqic']:
            raise ValueError("information_criterion deve essere uno tra: 'aic', 'bic', 'hqic'")
    
    def search(
        self, 
        series: pd.Series,
        verbose: bool = True,
        max_models: Optional[int] = None
    ) -> Tuple[int, int, int]:
        """
        Esegue grid search per trovare l'ordine ARIMA ottimale.
        
        Args:
            series: Serie temporale da addestrare
            verbose: Se stampare il progresso
            max_models: Numero massimo di modelli da valutare
            
        Returns:
            Ordine ARIMA ottimale (p, d, q)
        """
        self.logger.info(f"Avvio selezione modello ARIMA con criterio {self.information_criterion.upper()}")
        self.results = []
        
        # Genera tutte le combinazioni
        p_values = list(range(self.p_range[0], self.p_range[1] + 1))
        d_values = list(range(self.d_range[0], self.d_range[1] + 1))
        q_values = list(range(self.q_range[0], self.q_range[1] + 1))
        
        all_orders = list(product(p_values, d_values, q_values))
        
        if max_models and len(all_orders) > max_models:
            self.logger.info(f"Limitazione ricerca alle prime {max_models} combinazioni di modelli")
            all_orders = all_orders[:max_models]
        
        self.logger.info(f"Valutazione di {len(all_orders)} combinazioni di modelli")
        
        best_criterion = float('inf')
        best_order = None
        
        for i, order in enumerate(all_orders):
            try:
                if verbose and (i + 1) % 10 == 0:
                    self.logger.info(f"Progresso: {i + 1}/{len(all_orders)} modelli valutati")
                
                # Addestra modello
                model = ARIMA(series, order=order)
                fitted_model = model.fit(disp=False)
                
                # Ottieni valore criterio
                if self.information_criterion == 'aic':
                    criterion_value = fitted_model.aic
                elif self.information_criterion == 'bic':
                    criterion_value = fitted_model.bic
                elif self.information_criterion == 'hqic':
                    criterion_value = fitted_model.hqic
                
                # Memorizza risultati
                result = {
                    'order': order,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic, 
                    'hqic': fitted_model.hqic,
                    'llf': fitted_model.llf,
                    'params': len(fitted_model.params),
                    'converged': fitted_model.mle_retvals['converged'] if hasattr(fitted_model, 'mle_retvals') else True
                }
                
                self.results.append(result)
                
                # Controlla se questo è il miglior modello finora
                if criterion_value < best_criterion:
                    best_criterion = criterion_value
                    best_order = order
                    self.best_model = fitted_model
                
            except Exception as e:
                if verbose:
                    self.logger.debug(f"Impossibile addestrare ARIMA{order}: {e}")
                continue
        
        if not self.results:
            raise ModelTrainingError("Nessun modello è stato addestrato con successo")
        
        self.best_order = best_order
        self.logger.info(f"Miglior modello: ARIMA{best_order} con {self.information_criterion.upper()}={best_criterion:.2f}")
        
        return best_order
    
    def get_results_summary(self, top_n: int = 10) -> pd.DataFrame:
        """
        Ottieni riepilogo dei risultati di selezione del modello.
        
        Args:
            top_n: Numero di migliori modelli da restituire
            
        Returns:
            DataFrame con risultati dei modelli ordinati per criterio di informazione
        """
        if not self.results:
            raise ValueError("Nessun risultato disponibile. Esegui prima search().")
        
        df = pd.DataFrame(self.results)
        df = df.sort_values(self.information_criterion).head(top_n)
        
        return df
    
    def plot_selection_results(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Traccia i risultati della selezione del modello.
        
        Args:
            figsize: Dimensione figura per i grafici
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results:
                raise ValueError("Nessun risultato disponibile. Esegui prima search().")
            
            df = pd.DataFrame(self.results)
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('Risultati Selezione Modello ARIMA', fontsize=16)
            
            # Distribuzione AIC
            axes[0, 0].hist(df['aic'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(df['aic'].min(), color='red', linestyle='--', label='Miglior AIC')
            axes[0, 0].set_xlabel('AIC')
            axes[0, 0].set_ylabel('Frequenza')
            axes[0, 0].set_title('Distribuzione AIC')
            axes[0, 0].legend()
            
            # Distribuzione BIC
            axes[0, 1].hist(df['bic'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(df['bic'].min(), color='red', linestyle='--', label='Miglior BIC')
            axes[0, 1].set_xlabel('BIC')
            axes[0, 1].set_ylabel('Frequenza')
            axes[0, 1].set_title('Distribuzione BIC')
            axes[0, 1].legend()
            
            # Conteggio parametri vs AIC
            axes[1, 0].scatter(df['params'], df['aic'], alpha=0.6)
            axes[1, 0].set_xlabel('Numero di Parametri')
            axes[1, 0].set_ylabel('AIC')
            axes[1, 0].set_title('Parametri vs AIC')
            
            # Confronto migliori modelli
            top_models = df.nsmallest(10, self.information_criterion)
            order_labels = [f"({p},{d},{q})" for p, d, q in top_models['order']]
            
            axes[1, 1].bar(range(len(top_models)), top_models[self.information_criterion])
            axes[1, 1].set_xlabel('Ordine Modello')
            axes[1, 1].set_ylabel(self.information_criterion.upper())
            axes[1, 1].set_title(f'Top 10 Modelli per {self.information_criterion.upper()}')
            axes[1, 1].set_xticks(range(len(top_models)))
            axes[1, 1].set_xticklabels(order_labels, rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib non disponibile per i grafici")
            
    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Ottieni informazioni sul miglior modello selezionato.
        
        Returns:
            Dizionario con informazioni del miglior modello
        """
        if self.best_order is None:
            raise ValueError("Nessun miglior modello disponibile. Esegui prima search().")
        
        best_result = next((r for r in self.results if r['order'] == self.best_order), None)
        
        if best_result is None:
            raise ValueError("Risultato del miglior modello non trovato")
        
        info = {
            'best_order': self.best_order,
            'criterion_used': self.information_criterion,
            'aic': best_result['aic'],
            'bic': best_result['bic'],
            'hqic': best_result['hqic'], 
            'llf': best_result['llf'],
            'parameters': best_result['params'],
            'total_models_evaluated': len(self.results)
        }
        
        return info