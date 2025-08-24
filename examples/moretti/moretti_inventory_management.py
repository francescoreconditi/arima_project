"""
Sistema Intelligente di Gestione Scorte per Moretti S.p.A.
Previsione domanda prodotti medicali critici con fattori multipli
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import dalla nostra libreria
from arima_forecaster import (
    ARIMAForecaster,
    SARIMAForecaster,
    VARForecaster,
    TimeSeriesPreprocessor,
    ModelEvaluator,
    ForecastPlotter
)
from arima_forecaster.core import SARIMAModelSelector
from arima_forecaster.reporting import QuartoReportGenerator


# =====================================================
# CONFIGURAZIONE PRODOTTI MEDICALI CRITICI
# =====================================================

class ProdottoCategoria(Enum):
    """Categorie prodotti Moretti S.p.A."""
    CARROZZINE = "Carrozzine e Mobilità"
    ANTIDECUBITO = "Materassi e Cuscini Antidecubito"  
    RIABILITAZIONE = "Attrezzature Riabilitazione"
    ELETTROMEDICALI = "Elettromedicali e Diagnostica"
    HOME_CARE = "Ausili Home Care"


@dataclass
class ProdottoMedicale:
    """Definizione prodotto medicale con caratteristiche"""
    codice: str
    nome: str
    categoria: ProdottoCategoria
    prezzo_medio: float
    lead_time_giorni: int
    scadenza_mesi: Optional[int]
    scorta_minima: int
    scorta_sicurezza: int
    costo_stoccaggio_mensile: float
    criticita: int  # 1-5, 5=massima criticità
    

@dataclass
class Fornitore:
    """Definizione fornitore con pricing multi-tier"""
    codice: str
    nome: str
    lead_time_giorni: int
    affidabilita: float  # 0-1
    pricing_tiers: List[Tuple[int, float]]  # [(quantità_min, prezzo_unitario)]
    

# =====================================================
# GENERAZIONE DATI SINTETICI REALISTICI
# =====================================================

class GeneratoreDatiMoretti:
    """Genera dati sintetici realistici per Moretti S.p.A."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.prodotti = self._init_prodotti()
        self.fornitori = self._init_fornitori()
        
    def _init_prodotti(self) -> List[ProdottoMedicale]:
        """Inizializza catalogo prodotti critici"""
        return [
            # Carrozzine
            ProdottoMedicale(
                "CRZ001", "Carrozzina Pieghevole Standard", 
                ProdottoCategoria.CARROZZINE, 280.0, 
                15, None, 20, 10, 2.5, 5
            ),
            ProdottoMedicale(
                "CRZ002", "Carrozzina Elettrica Kyara",
                ProdottoCategoria.CARROZZINE, 1850.0,
                25, None, 5, 3, 8.0, 5
            ),
            
            # Materassi Antidecubito
            ProdottoMedicale(
                "MAT001", "Materasso Antidecubito Aria",
                ProdottoCategoria.ANTIDECUBITO, 450.0,
                10, 36, 15, 8, 3.5, 5
            ),
            ProdottoMedicale(
                "MAT002", "Cuscino Antidecubito Memory",
                ProdottoCategoria.ANTIDECUBITO, 85.0,
                7, 24, 30, 15, 1.2, 4
            ),
            
            # Riabilitazione
            ProdottoMedicale(
                "RIA001", "Deambulatore Pieghevole",
                ProdottoCategoria.RIABILITAZIONE, 65.0,
                5, None, 40, 20, 0.8, 4
            ),
            ProdottoMedicale(
                "RIA002", "Cyclette Riabilitativa",
                ProdottoCategoria.RIABILITAZIONE, 380.0,
                20, None, 8, 4, 4.5, 3
            ),
            
            # Elettromedicali
            ProdottoMedicale(
                "ELT001", "Saturimetro Professionale",
                ProdottoCategoria.ELETTROMEDICALI, 120.0,
                10, 60, 25, 12, 1.5, 5
            ),
            ProdottoMedicale(
                "ELT002", "Aerosol Ultrasuoni",
                ProdottoCategoria.ELETTROMEDICALI, 95.0,
                12, None, 20, 10, 1.8, 4
            ),
        ]
    
    def _init_fornitori(self) -> Dict[str, List[Fornitore]]:
        """Inizializza fornitori per prodotto con pricing multi-tier"""
        return {
            "CRZ001": [
                Fornitore("F001", "MedSupply Italia", 15, 0.95,
                         [(1, 300), (10, 280), (50, 260), (100, 240)]),
                Fornitore("F002", "EuroMedical", 12, 0.92,
                         [(1, 310), (20, 275), (80, 250)])
            ],
            "MAT001": [
                Fornitore("F003", "AntiDecubito Pro", 10, 0.98,
                         [(1, 480), (5, 450), (20, 420), (50, 400)]),
                Fornitore("F004", "Medical Comfort", 14, 0.90,
                         [(1, 470), (10, 440), (40, 410)])
            ],
            # Altri prodotti...
        }
    
    def genera_serie_vendite(
        self, 
        prodotto: ProdottoMedicale,
        anni: int = 3,
        freq: str = 'D'
    ) -> pd.Series:
        """
        Genera serie storica vendite con pattern realistici
        
        Componenti:
        - Trend crescente (invecchiamento popolazione)
        - Stagionalità (picchi inverno per alcuni prodotti)
        - Eventi speciali (es. COVID)
        - Rumore casuale
        """
        date_range = pd.date_range(
            end=datetime.now(), 
            periods=365 * anni,
            freq=freq
        )
        
        # Base demand
        base_demand = np.random.poisson(
            lam=20 + prodotto.criticita * 3, 
            size=len(date_range)
        )
        
        # Trend invecchiamento popolazione (+2% annuo)
        trend = np.linspace(1.0, 1.0 + (0.02 * anni), len(date_range))
        
        # Stagionalità (più forte per prodotti respiratori in inverno)
        t = np.arange(len(date_range))
        if prodotto.categoria == ProdottoCategoria.ELETTROMEDICALI:
            stagionalita = 1 + 0.3 * np.sin(2 * np.pi * t / 365 - np.pi/2)
        else:
            stagionalita = 1 + 0.1 * np.sin(2 * np.pi * t / 365)
        
        # Eventi speciali (es. COVID marzo 2020)
        eventi = np.ones(len(date_range))
        # Simula picco COVID per saturimetri/aerosol
        if prodotto.codice in ["ELT001", "ELT002"]:
            covid_start = len(date_range) - 1000  # ~3 anni fa
            if covid_start > 0:
                eventi[covid_start:covid_start+180] *= 2.5  # 6 mesi di picco
        
        # Combina componenti
        vendite = base_demand * trend * stagionalita * eventi
        
        # Aggiungi rumore e arrotonda
        vendite = np.round(vendite * (1 + np.random.normal(0, 0.1, len(date_range))))
        vendite = np.maximum(vendite, 0)  # No vendite negative
        
        return pd.Series(vendite, index=date_range, name=f"Vendite_{prodotto.codice}")
    
    def genera_dati_demografici(self, anni: int = 3) -> pd.DataFrame:
        """
        Genera dati demografici ISTAT simulati
        """
        date_range = pd.date_range(
            end=datetime.now(),
            periods=12 * anni,  # Mensile
            freq='M'
        )
        
        # Età media crescente
        eta_media = 45 + np.linspace(0, 1.5, len(date_range))
        
        # % popolazione over 65 crescente
        perc_over_65 = 22 + np.linspace(0, 2, len(date_range))
        
        # Indice dipendenza anziani
        indice_dipendenza = 35 + np.linspace(0, 3, len(date_range))
        
        return pd.DataFrame({
            'data': date_range,
            'eta_media': eta_media + np.random.normal(0, 0.2, len(date_range)),
            'perc_over_65': perc_over_65 + np.random.normal(0, 0.3, len(date_range)),
            'indice_dipendenza': indice_dipendenza + np.random.normal(0, 0.4, len(date_range))
        }).set_index('data')


# =====================================================
# SISTEMA PREVISIONE AVANZATO
# =====================================================

class SistemaGestioneScorte:
    """Sistema completo gestione scorte con ML"""
    
    def __init__(self, prodotti: List[ProdottoMedicale], fornitori: Dict):
        self.prodotti = {p.codice: p for p in prodotti}
        self.fornitori = fornitori
        self.modelli = {}
        self.previsioni = {}
        self.preprocessor = TimeSeriesPreprocessor()
        
    def addestra_modello_prodotto(
        self,
        prodotto_codice: str,
        serie_vendite: pd.Series,
        dati_demografici: Optional[pd.DataFrame] = None
    ):
        """
        Addestra modello SARIMA per singolo prodotto
        con possibili variabili esogene
        """
        prodotto = self.prodotti[prodotto_codice]
        
        print(f"\n[ANALISI] Prodotto: {prodotto.nome}")
        print(f"   Categoria: {prodotto.categoria.value}")
        print(f"   Criticita': {'*' * prodotto.criticita}")
        
        # Preprocessing
        serie_clean, _ = self.preprocessor.preprocess_pipeline(
            serie_vendite,
            handle_missing=True,
            missing_method='interpolate',
            remove_outliers_flag=True,
            outlier_method='iqr',
            make_stationary_flag=False
        )
        
        # Prova SARIMA con fallback ad ARIMA se fallisce
        print("   [SEARCH] Ricerca parametri ottimali...")
        
        model = None
        try:
            # Tentativo SARIMA avanzato
            selector = SARIMAModelSelector(
                p_range=(0, 2),  # Ridotto per velocità
                d_range=(0, 1), 
                q_range=(0, 2),
                P_range=(0, 1),  # Ridotto per stabilità
                D_range=(0, 1),
                Q_range=(0, 1),
                seasonal_periods=[7],
                information_criterion='aic',
                max_models=20  # Ridotto per velocità
            )
            
            selector.search(serie_clean, verbose=False)
            model = selector.get_best_model()
            
            if model is not None:
                print(f"   [OK] SARIMA: {selector.best_order}x{selector.best_seasonal_order}")
            else:
                raise Exception("SARIMA selector returned None")
                
        except Exception as e:
            # Fallback ad ARIMA semplice
            print(f"   [FALLBACK] SARIMA failed ({e}), using ARIMA(1,1,1)")
            model = ARIMAForecaster(order=(1, 1, 1))
            model.fit(serie_clean)
        
        # Verifica che il modello produca previsioni valide
        if model is not None:
            try:
                test_pred = model.forecast(steps=5, confidence_intervals=False)
                if isinstance(test_pred, dict):
                    test_pred = test_pred.get('forecast', test_pred)
                    
                if hasattr(test_pred, 'isna') and test_pred.isna().all():
                    raise Exception("Model produces all NaN predictions")
            except Exception as e2:
                print(f"   [DOUBLE_FALLBACK] Model validation failed ({e2}), using simpler ARIMA")
                model = ARIMAForecaster(order=(0, 1, 1))  # Anche più semplice
                model.fit(serie_clean)
            
        self.modelli[prodotto_codice] = model
        
        # Valuta performance con controlli NaN
        evaluator = ModelEvaluator()
        
        try:
            test_forecast = model.forecast(steps=30, confidence_intervals=False)
            
            # Estrai solo i valori se è un dict
            if isinstance(test_forecast, dict):
                test_forecast = test_forecast['forecast'] if 'forecast' in test_forecast else test_forecast
            
            # Controlla se le previsioni sono valide
            if hasattr(test_forecast, 'notna') and test_forecast.notna().any():
                metrics = evaluator.calculate_forecast_metrics(
                    actual=serie_clean[-30:].values,
                    predicted=test_forecast
                )
            else:
                print("   [WARN] Previsioni contengono NaN - usando metriche di fallback")
                metrics = {
                    'mape': float('inf'),
                    'rmse': float('inf'),
                    'mae': float('inf')
                }
        except Exception as e:
            print(f"   [WARN] Errore valutazione performance: {e}")
            metrics = {
                'mape': float('inf'), 
                'rmse': float('inf'),
                'mae': float('inf')
            }
        
        print(f"   [METRICS] Performance: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")
        
        return model, metrics
    
    def prevedi_domanda(
        self,
        prodotto_codice: str,
        orizzonte_giorni: int = 30,
        livello_confidenza: float = 0.95
    ) -> pd.DataFrame:
        """
        Prevede domanda futura con intervalli di confidenza
        """
        if prodotto_codice not in self.modelli:
            raise ValueError(f"Modello non addestrato per {prodotto_codice}")
            
        model = self.modelli[prodotto_codice]
        prodotto = self.prodotti[prodotto_codice]
        
        # Genera previsioni con gestione robusta
        try:
            forecast = model.forecast(
                steps=orizzonte_giorni,
                confidence_intervals=True,
                alpha=1 - livello_confidenza
            )
            
            # Gestisci diversi formati di output
            if isinstance(forecast, dict):
                previsione_values = forecast.get('forecast', forecast)
                if 'confidence_intervals' in forecast:
                    ci = forecast['confidence_intervals']
                    lower_values = ci.get('lower', previsione_values * 0.8)
                    upper_values = ci.get('upper', previsione_values * 1.2)
                else:
                    # Fallback se non ci sono CI
                    lower_values = previsione_values * 0.8
                    upper_values = previsione_values * 1.2
            else:
                # Se è una serie semplice
                previsione_values = forecast
                lower_values = previsione_values * 0.8
                upper_values = previsione_values * 1.2
                
        except Exception as e:
            # Fallback ulteriore
            print(f"   [WARN] Errore previsioni: {e}")
            forecast_simple = model.forecast(steps=orizzonte_giorni, confidence_intervals=False)
            if isinstance(forecast_simple, dict):
                previsione_values = forecast_simple.get('forecast', forecast_simple)
            else:
                previsione_values = forecast_simple
            lower_values = previsione_values * 0.8
            upper_values = previsione_values * 1.2
        
        # Calcola livelli scorte suggeriti
        forecast_df = pd.DataFrame({
            'previsione': previsione_values,
            'lower_bound': lower_values,
            'upper_bound': upper_values
        })
        
        # Controlla e correggi NaN nelle previsioni
        if forecast_df['previsione'].isna().all():
            print(f"   [CRITICAL] Tutte le previsioni sono NaN - usando valori di fallback")
            # Usa media storica come fallback
            media_storica = 20  # Fallback generico
            forecast_df['previsione'] = media_storica
            forecast_df['lower_bound'] = media_storica * 0.8
            forecast_df['upper_bound'] = media_storica * 1.2
        
        # Punto di riordino considerando lead time
        forecast_df['punto_riordino'] = (
            forecast_df['upper_bound'].rolling(prodotto.lead_time_giorni).sum() +
            prodotto.scorta_sicurezza
        )
        
        # Riempie NaN nei punti di riordino con valori ragionevoli
        if forecast_df['punto_riordino'].isna().any():
            fallback_riordino = prodotto.scorta_minima * 2
            forecast_df['punto_riordino'].fillna(fallback_riordino, inplace=True)
        
        # Quantità economica di riordino (EOQ semplificato)
        domanda_media = forecast_df['previsione'].mean()
        if pd.isna(domanda_media) or domanda_media <= 0:
            domanda_media = 20  # Fallback
            
        forecast_df['quantita_riordino'] = np.sqrt(
            (2 * domanda_media * 50) / max(prodotto.costo_stoccaggio_mensile, 1)
        )
        
        self.previsioni[prodotto_codice] = forecast_df
        
        return forecast_df
    
    def ottimizza_fornitore(
        self,
        prodotto_codice: str,
        quantita_necessaria: int
    ) -> Dict:
        """
        Seleziona fornitore ottimale con pricing multi-tier
        """
        if prodotto_codice not in self.fornitori:
            return None
            
        fornitori_prodotto = self.fornitori[prodotto_codice]
        prodotto = self.prodotti[prodotto_codice]
        
        migliore_offerta = {
            'fornitore': None,
            'prezzo_totale': float('inf'),
            'prezzo_unitario': 0,
            'lead_time': 0
        }
        
        for fornitore in fornitori_prodotto:
            # Trova tier pricing applicabile
            prezzo_unitario = fornitore.pricing_tiers[0][1]  # Default
            for qty_min, prezzo in fornitore.pricing_tiers:
                if quantita_necessaria >= qty_min:
                    prezzo_unitario = prezzo
                    
            prezzo_totale = quantita_necessaria * prezzo_unitario
            
            # Penalizza per scarsa affidabilità
            prezzo_totale_adjusted = prezzo_totale / fornitore.affidabilita
            
            if prezzo_totale_adjusted < migliore_offerta['prezzo_totale']:
                migliore_offerta = {
                    'fornitore': fornitore.nome,
                    'codice_fornitore': fornitore.codice,
                    'prezzo_totale': prezzo_totale,
                    'prezzo_unitario': prezzo_unitario,
                    'lead_time': fornitore.lead_time_giorni,
                    'affidabilita': fornitore.affidabilita,
                    'risparmio_vs_listino': (
                        (fornitore.pricing_tiers[0][1] - prezzo_unitario) * 
                        quantita_necessaria
                    )
                }
        
        return migliore_offerta
    
    def genera_piano_approvvigionamento(
        self,
        orizzonte_giorni: int = 90
    ) -> pd.DataFrame:
        """
        Genera piano completo approvvigionamento multi-prodotto
        """
        piano = []
        
        for codice, prodotto in self.prodotti.items():
            if codice not in self.previsioni:
                continue
                
            forecast = self.previsioni[codice]
            
            # Simula livello scorte attuale
            scorte_attuali = np.random.randint(
                prodotto.scorta_minima,
                prodotto.scorta_minima * 3
            )
            
            # Calcola quando riordinare
            scorte_simulate = scorte_attuali
            for giorno in range(orizzonte_giorni):
                # Sottrai vendite previste
                if giorno < len(forecast):
                    scorte_simulate -= forecast.iloc[giorno]['previsione']
                    
                # Verifica punto riordino
                if scorte_simulate <= forecast.iloc[min(giorno, len(forecast)-1)]['punto_riordino']:
                    # Calcola quantità da ordinare
                    qty = int(forecast.iloc[min(giorno, len(forecast)-1)]['quantita_riordino'])
                    
                    # Ottimizza fornitore
                    fornitore_ottimale = self.ottimizza_fornitore(codice, qty)
                    
                    if fornitore_ottimale:
                        piano.append({
                            'giorno_ordine': giorno,
                            'prodotto_codice': codice,
                            'prodotto_nome': prodotto.nome,
                            'quantita': qty,
                            'fornitore': fornitore_ottimale['fornitore'],
                            'costo_totale': fornitore_ottimale['prezzo_totale'],
                            'lead_time': fornitore_ottimale['lead_time'],
                            'data_consegna_prevista': giorno + fornitore_ottimale['lead_time'],
                            'risparmio': fornitore_ottimale['risparmio_vs_listino']
                        })
                        
                        # Aggiorna scorte simulate
                        scorte_simulate += qty
        
        return pd.DataFrame(piano)


# =====================================================
# ESEMPIO PRATICO COMPLETO
# =====================================================

def esempio_completo_moretti():
    """
    Esempio completo gestione scorte Moretti S.p.A.
    """
    print("=" * 80)
    print("SISTEMA INTELLIGENTE GESTIONE SCORTE - MORETTI S.P.A.")
    print("=" * 80)
    
    # 1. Inizializza sistema
    print("\n[INIT] Inizializzazione sistema...")
    generatore = GeneratoreDatiMoretti()
    sistema = SistemaGestioneScorte(
        generatore.prodotti,
        generatore.fornitori
    )
    
    # 2. Genera dati storici e demografici
    print("\n[DATA] Generazione dati storici...")
    dati_demografici = generatore.genera_dati_demografici()
    
    # 3. Addestra modelli per prodotti critici
    prodotti_critici = [p for p in generatore.prodotti if p.criticita >= 4]
    
    for prodotto in prodotti_critici[:3]:  # Demo con primi 3 prodotti
        print(f"\n{'='*60}")
        serie_vendite = generatore.genera_serie_vendite(prodotto)
        
        # Addestra con dati demografici per prodotti più critici
        if prodotto.criticita == 5:
            sistema.addestra_modello_prodotto(
                prodotto.codice,
                serie_vendite,
                dati_demografici
            )
        else:
            sistema.addestra_modello_prodotto(
                prodotto.codice,
                serie_vendite
            )
        
        # Genera previsioni
        previsioni = sistema.prevedi_domanda(
            prodotto.codice,
            orizzonte_giorni=30
        )
        
        print(f"\n   [FORECAST] Previsioni prossimi 7 giorni:")
        print(previsioni.head(7)[['previsione', 'punto_riordino']].round(0))
        
        # Ottimizza fornitore per quantità media
        qty_media = int(previsioni['quantita_riordino'].mean())
        fornitore = sistema.ottimizza_fornitore(prodotto.codice, qty_media)
        
        if fornitore:
            print(f"\n   [SUPPLIER] Fornitore ottimale per {qty_media} unità:")
            print(f"      {fornitore['fornitore']}: EUR {fornitore['prezzo_totale']:.2f}")
            risparmio = fornitore.get('risparmio_vs_listino', 0)
            print(f"      Risparmio: EUR {risparmio:.2f}")
    
    # 4. Genera piano approvvigionamento
    print(f"\n{'='*60}")
    print("\n[PLAN] PIANO APPROVVIGIONAMENTO TRIMESTRALE")
    print("="*60)
    
    piano = sistema.genera_piano_approvvigionamento(orizzonte_giorni=90)
    
    if not piano.empty:
        # Raggruppa per mese
        piano['mese'] = piano['giorno_ordine'] // 30
        sintesi = piano.groupby('mese').agg({
            'costo_totale': 'sum',
            'risparmio': 'sum',
            'prodotto_codice': 'count'
        }).rename(columns={'prodotto_codice': 'num_ordini'})
        
        print("\n[SUMMARY] Sintesi per mese:")
        for mese, row in sintesi.iterrows():
            print(f"\n   Mese {mese+1}:")
            print(f"   - Ordini previsti: {row['num_ordini']}")
            print(f"   - Investimento: EUR {row['costo_totale']:,.2f}")
            print(f"   - Risparmio stimato: EUR {row['risparmio']:,.2f}")
        
        # Top prodotti da riordinare
        print("\n[TOP] Prodotti piu' frequenti da riordinare:")
        top_prodotti = piano['prodotto_nome'].value_counts().head(3)
        for prod, count in top_prodotti.items():
            print(f"   - {prod}: {count} riordini")
    
    # 5. Salva risultati
    print(f"\n{'='*60}")
    print("[SAVE] Salvataggio risultati...")
    
    # Determina path assoluto
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    outputs_reports = project_root / "outputs" / "reports"
    outputs_plots = project_root / "outputs" / "plots"
    
    # Crea directory se non esistono
    outputs_reports.mkdir(parents=True, exist_ok=True)
    outputs_plots.mkdir(parents=True, exist_ok=True)
    
    # Salva piano (Excel se disponibile, altrimenti CSV)
    try:
        output_file = outputs_reports / "moretti_piano_scorte.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            piano.to_excel(writer, sheet_name='Piano_Approvvigionamento', index=False)
            
            for codice in sistema.previsioni:
                sistema.previsioni[codice].to_excel(
                    writer, 
                    sheet_name=f'Previsioni_{codice}'
                )
    except ImportError:
        # Fallback a CSV se openpyxl non è disponibile
        print("   [INFO] openpyxl non disponibile, salvataggio in CSV...")
        output_file = outputs_reports / "moretti_piano_scorte.csv"
        piano.to_csv(output_file, index=False)
        
        for codice in sistema.previsioni:
            prev_file = outputs_reports / f"moretti_previsioni_{codice}.csv"
            sistema.previsioni[codice].to_csv(prev_file)
    
    print(f"[OK] Piano salvato in: {output_file}")
    
    # 6. Genera visualizzazioni (opzionale)
    print("\n[DASHBOARD] Generazione dashboard...")
    try:
        for codice in list(sistema.modelli.keys())[:2]:  # Demo con primi 2
            plotter = ForecastPlotter()
            
            # Crea dashboard interattivo
            try:
                dashboard = plotter.create_dashboard(sistema.modelli[codice])
            except:
                # Fallback se create_dashboard ha API diversa
                dashboard = plotter.plot_forecast_with_components(sistema.modelli[codice])
            
            output_path = outputs_plots / f"moretti_{codice}_dashboard.html"
            
            if hasattr(dashboard, 'write_html'):
                dashboard.write_html(output_path)
                print(f"   [OK] Dashboard {codice}: {output_path}")
            else:
                print(f"   [SKIP] Dashboard {codice}: formato non supportato")
    except Exception as e:
        print(f"   [WARN] Errore dashboard: {e}")
    
    print(f"\n{'='*80}")
    print("[SUCCESS] SISTEMA COMPLETATO CON SUCCESSO!")
    print("="*80)
    
    return sistema, piano


if __name__ == "__main__":
    # Esegui esempio completo
    sistema, piano = esempio_completo_moretti()
    
    print("\n[TIPS] Suggerimenti operativi:")
    print("1. Integrare con ERP aziendale per dati real-time")
    print("2. Schedulare training modelli settimanale")
    print("3. Configurare alert automatici per riordini")
    print("4. Dashboard Power BI per management")
    print("5. API REST per integrazione con e-commerce")