"""
============================================
FILE DI TEST/DEBUG - NON PER PRODUZIONE
Creato da: Claude Code
Data: 2025-09-02
Scopo: POC Intermittent Demand per ricambi Moretti
============================================

POC: Sistema Gestione Ricambi con Domanda Intermittente
Caso Moretti S.p.A. - Ricambi Carrozzine e Dispositivi Medicali

Questo esempio dimostra come gestire ricambi con domanda sporadica:
- Ruote carrozzine
- Batterie montascale
- Cinghie sollevatori
- Componenti elettronici
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Import moduli intermittent demand
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.arima_forecaster.core.intermittent_model import (
    IntermittentForecaster,
    IntermittentConfig,
    IntermittentMethod,
)
from src.arima_forecaster.evaluation.intermittent_metrics import IntermittentEvaluator


def genera_domanda_intermittente(n_periods=365, tipo="spare_part", seed=42):
    """
    Genera dati realistici domanda intermittente per ricambi

    Args:
        n_periods: Numero giorni da simulare
        tipo: Tipo pattern (spare_part, seasonal, lumpy)
        seed: Random seed per riproducibilità
    """
    np.random.seed(seed)

    if tipo == "spare_part":
        # Ricambio tipico: domanda ogni 10-20 giorni
        prob_demand = 0.08  # 8% probabilità domanda giornaliera
        demand_size = lambda: np.random.poisson(3) + 1  # 1-10 pezzi

    elif tipo == "seasonal":
        # Ricambio stagionale (es. batterie in inverno)
        base_prob = 0.05
        seasonal_factor = 1 + 0.5 * np.sin(np.arange(n_periods) * 2 * np.pi / 365)
        prob_demand = base_prob * seasonal_factor
        demand_size = lambda: np.random.poisson(2) + 1

    elif tipo == "lumpy":
        # Domanda molto irregolare (grandi ordini sporadici)
        prob_demand = 0.03  # Molto rara
        demand_size = lambda: np.random.choice([1, 5, 10, 20], p=[0.4, 0.3, 0.2, 0.1])

    else:
        prob_demand = 0.1
        demand_size = lambda: 1

    # Genera serie
    demand = []
    for i in range(n_periods):
        if isinstance(prob_demand, float):
            p = prob_demand
        else:
            p = prob_demand[i]

        if np.random.random() < p:
            demand.append(demand_size())
        else:
            demand.append(0)

    return np.array(demand)


def analizza_ricambio(
    codice, nome, domanda_storica, costo_unitario, lead_time_giorni, service_level_target=0.95
):
    """
    Analisi completa ricambio con tutti i metodi intermittent

    Args:
        codice: Codice ricambio
        nome: Descrizione ricambio
        domanda_storica: Array con storico domanda
        costo_unitario: Costo unitario ricambio
        lead_time_giorni: Lead time fornitore
        service_level_target: Livello servizio target
    """
    print(f"\n{'=' * 60}")
    print(f"ANALISI RICAMBIO: {codice} - {nome}")
    print(f"{'=' * 60}")

    # Split train/test
    split_point = int(len(domanda_storica) * 0.8)
    train_data = domanda_storica[:split_point]
    test_data = domanda_storica[split_point:]

    print(f"\nDati: {len(train_data)} giorni training, {len(test_data)} giorni test")
    print(f"Domanda totale periodo: {sum(train_data)} unità")
    print(
        f"Giorni con domanda: {sum(train_data > 0)} ({100 * sum(train_data > 0) / len(train_data):.1f}%)"
    )
    print(f"Domanda media (quando presente): {np.mean(train_data[train_data > 0]):.1f} unità")

    # Test tutti i metodi
    metodi = [
        IntermittentMethod.CROSTON,
        IntermittentMethod.SBA,
        IntermittentMethod.TSB,
        IntermittentMethod.ADAPTIVE_CROSTON,
    ]

    risultati = {}
    best_method = None
    best_mase = float("inf")

    print(f"\n{'-' * 40}")
    print("CONFRONTO METODI FORECASTING:")
    print(f"{'-' * 40}")

    for metodo in metodi:
        try:
            # Configura e addestra modello
            config = IntermittentConfig(
                method=metodo,
                alpha=0.1,
                optimize_alpha=True if metodo == IntermittentMethod.ADAPTIVE_CROSTON else False,
            )

            model = IntermittentForecaster(config)
            model.fit(train_data)

            # Analizza pattern
            pattern = model.pattern_

            # Genera forecast
            forecast = model.forecast(len(test_data))

            # Valuta performance
            evaluator = IntermittentEvaluator(
                holding_cost=costo_unitario * 0.2,  # 20% costo annuo stoccaggio
                stockout_cost=costo_unitario * 5,  # Penalità 5x per stockout
            )

            metrics = evaluator.evaluate(test_data, forecast)

            # Calcola quantità riordino
            safety_stock = model.calculate_safety_stock(lead_time_giorni, service_level_target)
            reorder_point = model.calculate_reorder_point(lead_time_giorni, service_level_target)

            # Salva risultati
            risultati[metodo.value] = {
                "model": model,
                "metrics": metrics,
                "forecast": forecast,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
            }

            # Stampa risultati
            print(f"\n{metodo.value.upper()}:")
            print(f"  Pattern: {pattern.classification}")
            print(f"  Forecast giornaliero: {model.forecast_:.3f} unità")
            print(f"  MASE: {metrics.mase:.3f}")
            print(f"  Fill Rate: {metrics.fill_rate:.1f}%")
            print(f"  Service Level: {metrics.achieved_service_level:.1f}%")
            print(f"  Safety Stock: {safety_stock:.0f} unità")
            print(f"  Reorder Point: {reorder_point:.0f} unità")

            if metrics.mase < best_mase:
                best_mase = metrics.mase
                best_method = metodo.value

        except Exception as e:
            print(f"\n{metodo.value}: Errore - {str(e)}")

    # Raccomandazione finale
    if best_method and best_method in risultati:
        best = risultati[best_method]
        print(f"\n{'=' * 40}")
        print("RACCOMANDAZIONE FINALE:")
        print(f"{'=' * 40}")
        print(f"Metodo ottimale: {best_method.upper()}")
        print(f"Quantità riordino consigliata: {best['reorder_point']:.0f} unità")
        print(f"Safety stock: {best['safety_stock']:.0f} unità")
        print(f"Investimento scorta: €{best['reorder_point'] * costo_unitario:,.2f}")

        # Calcolo EOQ semplificato
        domanda_annua = sum(train_data) * 365 / len(train_data)
        costo_ordine = 50  # Costo fisso ordine
        costo_stoccaggio = costo_unitario * 0.2

        if domanda_annua > 0:
            eoq = np.sqrt(2 * domanda_annua * costo_ordine / costo_stoccaggio)
            print(f"EOQ (lotto economico): {eoq:.0f} unità")
            print(f"Ordini/anno consigliati: {domanda_annua / eoq:.1f}")

    return risultati


def esempio_portfolio_ricambi():
    """
    Analizza portfolio completo ricambi Moretti
    """
    print("\n" + "=" * 70)
    print(" MORETTI S.p.A. - ANALISI RICAMBI DOMANDA INTERMITTENTE")
    print("=" * 70)
    print("\nData analisi:", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Portfolio ricambi realistici
    ricambi = [
        {
            "codice": "RC-W001",
            "nome": "Ruota anteriore carrozzina standard",
            "pattern": "spare_part",
            "costo": 45.00,
            "lead_time": 15,
            "service_level": 0.95,
        },
        {
            "codice": "RC-B002",
            "nome": "Batteria montascale 24V",
            "pattern": "seasonal",
            "costo": 120.00,
            "lead_time": 30,
            "service_level": 0.98,  # Critico
        },
        {
            "codice": "RC-C003",
            "nome": "Cinghia sollevatore paziente",
            "pattern": "lumpy",
            "costo": 85.00,
            "lead_time": 20,
            "service_level": 0.90,
        },
    ]

    risultati_portfolio = []

    for ricambio in ricambi:
        # Genera dati storici
        domanda = genera_domanda_intermittente(n_periods=365, tipo=ricambio["pattern"])

        # Analizza ricambio
        risultati = analizza_ricambio(
            codice=ricambio["codice"],
            nome=ricambio["nome"],
            domanda_storica=domanda,
            costo_unitario=ricambio["costo"],
            lead_time_giorni=ricambio["lead_time"],
            service_level_target=ricambio["service_level"],
        )

        # Raccogli per summary
        if risultati:
            best_method = min(
                risultati.items(), key=lambda x: x[1]["metrics"].mase if x[1] else float("inf")
            )

            risultati_portfolio.append(
                {
                    "Codice": ricambio["codice"],
                    "Descrizione": ricambio["nome"][:30],
                    "Metodo": best_method[0],
                    "ROP": best_method[1]["reorder_point"],
                    "SS": best_method[1]["safety_stock"],
                    "Investimento": best_method[1]["reorder_point"] * ricambio["costo"],
                }
            )

    # Summary finale
    print("\n" + "=" * 70)
    print(" SUMMARY PORTFOLIO RICAMBI")
    print("=" * 70)

    df_summary = pd.DataFrame(risultati_portfolio)
    print("\n", df_summary.to_string(index=False))

    print(f"\n{'=' * 40}")
    print("TOTALI:")
    print(f"{'=' * 40}")
    print(f"Investimento totale ricambi: €{df_summary['Investimento'].sum():,.2f}")
    print(f"Numero SKU gestiti: {len(df_summary)}")
    print(f"Investimento medio/SKU: €{df_summary['Investimento'].mean():,.2f}")

    # Salva risultati
    output_file = Path(__file__).parent.parent.parent / "outputs" / "moretti_ricambi_analysis.csv"
    output_file.parent.mkdir(exist_ok=True)
    df_summary.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in: {output_file}")

    return df_summary


def test_veloce():
    """Test veloce singolo ricambio per verifica funzionamento"""
    print("\n=== TEST VELOCE INTERMITTENT DEMAND ===\n")

    # Genera dati test
    domanda = genera_domanda_intermittente(100, "spare_part")

    # Test Croston base
    config = IntermittentConfig(method=IntermittentMethod.CROSTON, alpha=0.1)
    model = IntermittentForecaster(config)
    model.fit(domanda[:80])

    forecast = model.forecast(20)
    pattern = model.pattern_

    print(f"Pattern identificato: {pattern.classification}")
    print(f"ADI (giorni tra ordini): {pattern.adi:.1f}")
    print(f"Intermittenza: {pattern.intermittence:.1%}")
    print(f"Forecast: {model.forecast_:.3f} unità/giorno")
    print(f"Test completato con successo!")

    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test veloce
        test_veloce()
    else:
        # Analisi completa portfolio
        esempio_portfolio_ricambi()
