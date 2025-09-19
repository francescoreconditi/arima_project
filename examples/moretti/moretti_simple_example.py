"""
Esempio Semplificato Sistema Scorte - Moretti S.p.A.
Test base con ARIMA standard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Import dalla nostra libreria
from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ModelEvaluator


def esempio_semplice_moretti():
    """
    Esempio semplificato per test iniziale
    """
    print("=" * 60)
    print("MORETTI S.P.A. - TEST SISTEMA PREVISIONE SCORTE")
    print("=" * 60)

    # 1. Genera dati vendite simulati per Carrozzina Standard
    print("\n[STEP 1] Generazione dati vendite carrozzine...")

    # Simula 2 anni di vendite giornaliere
    date_range = pd.date_range(
        end=datetime.now(),
        periods=730,  # 2 anni
        freq="D",
    )

    # Vendite base con trend e stagionalità
    np.random.seed(42)
    trend = np.linspace(20, 25, 730)  # Trend crescente
    stagionalita = 5 * np.sin(2 * np.pi * np.arange(730) / 365)  # Stagionalità annuale
    rumore = np.random.normal(0, 3, 730)

    vendite = trend + stagionalita + rumore
    vendite = np.maximum(vendite, 0)  # No vendite negative
    vendite = np.round(vendite)

    serie_vendite = pd.Series(vendite, index=date_range, name="Vendite_Carrozzine")

    print(
        f"   Periodo: {date_range[0].strftime('%Y-%m-%d')} - {date_range[-1].strftime('%Y-%m-%d')}"
    )
    print(f"   Media vendite: {vendite.mean():.1f} unità/giorno")
    print(f"   Dev. standard: {vendite.std():.1f}")

    # 2. Preprocessing
    print("\n[STEP 2] Preprocessing dei dati...")
    preprocessor = TimeSeriesPreprocessor()

    serie_clean, metadata = preprocessor.preprocess_pipeline(
        serie_vendite,
        handle_missing=True,
        missing_method="interpolate",
        remove_outliers_flag=False,  # Non rimuovere outliers per evitare problemi
        outlier_method="iqr",
        make_stationary_flag=False,  # ARIMA gestisce già la stazionarietà
    )

    print(f"   Outliers rimossi: {metadata.get('outliers_removed', 0)}")
    print(f"   Osservazioni finali: {len(serie_clean)}")

    # 3. Split train/test
    print("\n[STEP 3] Split dataset train/test...")
    split_point = -60  # Ultimi 60 giorni per test
    train = serie_clean[:split_point]
    test = serie_clean[split_point:]

    print(f"   Train set: {len(train)} giorni")
    print(f"   Test set: {len(test)} giorni")

    # 4. Training modello ARIMA
    print("\n[STEP 4] Training modello ARIMA...")
    model = ARIMAForecaster(order=(1, 1, 1))  # Parametri più semplici per stabilità

    try:
        model.fit(train)
        print("   [OK] Modello addestrato con successo")

        # Info modello
        if hasattr(model.model, "aic"):
            print(f"   AIC: {model.model.aic:.2f}")
        if hasattr(model.model, "bic"):
            print(f"   BIC: {model.model.bic:.2f}")
    except Exception as e:
        print(f"   [ERRORE] Training fallito: {e}")
        return None

    # 5. Previsioni
    print("\n[STEP 5] Generazione previsioni...")

    try:
        # Previsioni su test set (senza intervalli di confidenza per semplicità)
        previsioni_result = model.forecast(steps=len(test), confidence_intervals=False)

        # Il risultato dovrebbe essere una Series
        if isinstance(previsioni_result, pd.Series):
            previsioni = previsioni_result
        elif isinstance(previsioni_result, dict) and "forecast" in previsioni_result:
            previsioni = previsioni_result["forecast"]
        else:
            previsioni = previsioni_result

        print("   [OK] Previsioni generate")
        print(f"   Media prevista: {previsioni.mean():.1f} unità/giorno")

    except Exception as e:
        print(f"   [ERRORE] Previsione fallita: {e}")
        return None

    # 6. Valutazione performance
    print("\n[STEP 6] Valutazione performance...")
    evaluator = ModelEvaluator()

    metrics = evaluator.calculate_forecast_metrics(actual=test.values, predicted=previsioni)

    print(f"   MAE: {metrics['mae']:.2f}")
    print(f"   RMSE: {metrics['rmse']:.2f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    if "r2" in metrics:
        print(f"   R²: {metrics['r2']:.3f}")

    # 7. Calcolo scorte suggerite
    print("\n[STEP 7] Calcolo livelli scorte...")

    # Parametri prodotto
    lead_time = 15  # giorni
    costo_stoccaggio = 2.5  # EUR/unità/mese
    costo_ordine = 100  # EUR/ordine

    # Previsione domanda durante lead time
    previsioni_lead_time = model.forecast(steps=lead_time, confidence_intervals=False)

    domanda_lead_time = previsioni_lead_time.sum()
    std_lead_time = np.std(previsioni_lead_time)

    # Scorta sicurezza (95% service level)
    z_score = 1.65
    scorta_sicurezza = z_score * std_lead_time * np.sqrt(lead_time)

    # Punto riordino
    punto_riordino = domanda_lead_time + scorta_sicurezza

    # EOQ (Economic Order Quantity)
    domanda_annuale = previsioni.mean() * 365
    eoq = np.sqrt((2 * domanda_annuale * costo_ordine) / (costo_stoccaggio * 12))

    print(f"   Domanda durante lead time: {domanda_lead_time:.0f} unità")
    print(f"   Scorta sicurezza: {scorta_sicurezza:.0f} unità")
    print(f"   Punto di riordino: {punto_riordino:.0f} unità")
    print(f"   Quantità economica ordine (EOQ): {eoq:.0f} unità")

    # 8. Previsioni future (30 giorni)
    print("\n[STEP 8] Previsioni prossimi 30 giorni...")

    previsioni_future = model.forecast(steps=30, confidence_intervals=False)

    # Crea DataFrame per visualizzazione
    future_dates = pd.date_range(
        start=serie_clean.index[-1] + timedelta(days=1), periods=30, freq="D"
    )

    df_previsioni = pd.DataFrame({"Data": future_dates, "Previsione": previsioni_future})

    # Mostra prime e ultime previsioni
    print("\n   Primi 5 giorni:")
    for i in range(min(5, len(df_previsioni))):
        row = df_previsioni.iloc[i]
        print(f"   {row['Data'].strftime('%Y-%m-%d')}: {row['Previsione']:.0f}")

    print("\n   Ultimi 5 giorni:")
    for i in range(max(0, len(df_previsioni) - 5), len(df_previsioni)):
        row = df_previsioni.iloc[i]
        print(f"   {row['Data'].strftime('%Y-%m-%d')}: {row['Previsione']:.0f}")

    # 9. Suggerimenti operativi
    print("\n" + "=" * 60)
    print("SUGGERIMENTI OPERATIVI")
    print("=" * 60)

    # Verifica se riordinare
    scorte_attuali_simulate = np.random.randint(50, 150)  # Simula scorte attuali
    print(f"\n[INFO] Scorte attuali simulate: {scorte_attuali_simulate} unità")

    if scorte_attuali_simulate <= punto_riordino:
        print(f"[ALERT] RIORDINO NECESSARIO!")
        print(f"   Ordinare: {eoq:.0f} unità")
        print(f"   Urgenza: {'ALTA' if scorte_attuali_simulate < scorta_sicurezza else 'MEDIA'}")

        # Calcola copertura giorni
        giorni_copertura = scorte_attuali_simulate / previsioni_future.mean()
        print(f"   Copertura stimata: {giorni_copertura:.1f} giorni")

        if giorni_copertura < lead_time:
            print(f"   [CRITICO] Rischio stockout prima dell'arrivo ordine!")
    else:
        giorni_al_riordino = (scorte_attuali_simulate - punto_riordino) / previsioni_future.mean()
        print(f"[OK] Scorte sufficienti")
        print(f"   Prossimo riordino tra: ~{giorni_al_riordino:.0f} giorni")

    # 10. Salvataggio risultati
    print("\n[STEP 9] Salvataggio risultati...")

    try:
        # Determina path assoluto alla directory outputs
        current_dir = Path(__file__).parent  # examples/moretti
        project_root = current_dir.parent.parent  # root del progetto
        outputs_dir = project_root / "outputs" / "reports"

        # Crea directory se non esiste
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Salva previsioni
        output_file = outputs_dir / "moretti_previsioni_semplici.csv"
        df_previsioni.to_csv(output_file, index=False)
        print(f"   [OK] Previsioni salvate in: {output_file}")

        # Salva metriche
        metrics_df = pd.DataFrame([metrics])
        metrics_file = outputs_dir / "moretti_metriche.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"   [OK] Metriche salvate in: {metrics_file}")

    except Exception as e:
        print(f"   [WARN] Impossibile salvare file: {e}")

    print("\n" + "=" * 60)
    print("[SUCCESS] ANALISI COMPLETATA CON SUCCESSO!")
    print("=" * 60)

    return {
        "model": model,
        "previsioni": df_previsioni,
        "metriche": metrics,
        "scorte": {
            "punto_riordino": punto_riordino,
            "scorta_sicurezza": scorta_sicurezza,
            "eoq": eoq,
        },
    }


if __name__ == "__main__":
    # Esegui esempio
    risultati = esempio_semplice_moretti()

    if risultati:
        print("\n[INFO] Integrazione suggerita con sistema Moretti:")
        print("1. Connettere a database ERP per dati real-time")
        print("2. Schedulare training modello settimanale")
        print("3. Configurare alert email per riordini")
        print("4. Integrare con sistema fornitori per ordini automatici")
        print("5. Dashboard web per monitoraggio continuo")
