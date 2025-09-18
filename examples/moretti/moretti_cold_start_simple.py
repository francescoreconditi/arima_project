"""
Esempio semplice Cold Start Problem - Moretti S.p.A.
Versione senza emoji per compatibilità Windows
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Aggiungi il path del modulo
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from arima_forecaster.core.cold_start import ColdStartForecaster

    print("[OK] Cold Start module imported successfully")
except ImportError as e:
    print(f"[ERROR] Cannot import Cold Start module: {e}")
    sys.exit(1)


def create_demo_data():
    """Crea dati demo semplificati"""

    # Genera 60 giorni di vendite per 3 prodotti
    dates = pd.date_range(start="2024-06-01", end="2024-07-30", freq="D")

    # Dati vendite simulati
    np.random.seed(42)
    vendite_data = {
        "CRZ001": np.random.poisson(25, len(dates)),  # Carrozzina media domanda
        "MAT001": np.random.poisson(20, len(dates)),  # Materasso
        "ELT001": np.random.poisson(15, len(dates)),  # Elettromedicale
    }

    vendite_df = pd.DataFrame(vendite_data, index=dates)

    # Info prodotti
    prodotti_info = {
        "CRZ001": {
            "nome": "Carrozzina Standard",
            "categoria": "Carrozzine",
            "prezzo": 450.0,
            "peso": 12.0,
        },
        "MAT001": {
            "nome": "Materasso Antidecubito",
            "categoria": "Materassi Antidecubito",
            "prezzo": 250.0,
            "peso": 8.0,
        },
        "ELT001": {
            "nome": "Saturimetro Digitale",
            "categoria": "Elettromedicali",
            "prezzo": 89.0,
            "peso": 0.3,
        },
    }

    return vendite_df, prodotti_info


def demo_cold_start():
    """Demo principale Cold Start"""

    print("COLD START PROBLEM DEMO - MORETTI S.p.A.")
    print("=" * 50)

    # 1. Crea dati demo
    print("\n[STEP 1] Creazione dati demo...")
    vendite_df, prodotti_info = create_demo_data()
    print(f"Creati {len(vendite_df.columns)} prodotti con {len(vendite_df)} giorni di storia")

    # 2. Nuovo prodotto da lanciare
    print("\n[STEP 2] Definizione nuovo prodotto...")
    nuovo_prodotto = {
        "codice": "CRZ-PREMIUM-001",
        "nome": "Carrozzina Premium Ultra-Light",
        "categoria": "Carrozzine",
        "prezzo": 750.0,  # Premium vs CRZ001
        "peso": 9.0,  # Più leggera
        "volume": 100.0,
    }

    print(f"Nuovo prodotto: {nuovo_prodotto['nome']}")
    print(f"Categoria: {nuovo_prodotto['categoria']}")
    print(f"Prezzo: EUR {nuovo_prodotto['prezzo']}")

    # 3. Prepara database per Cold Start
    print("\n[STEP 3] Preparazione database prodotti...")

    cold_start_forecaster = ColdStartForecaster()
    products_database = {}

    for codice, vendite_series in vendite_df.items():
        info_prodotto = prodotti_info[codice]

        # Estrai features
        features = cold_start_forecaster.extract_product_features(vendite_series, info_prodotto)

        products_database[codice] = {
            "vendite": vendite_series,
            "info": info_prodotto,
            "features": features,
        }

        print(f"  {codice}: {len(vendite_series)} giorni, {len(features)} features")

    # 4. Prepara features nuovo prodotto
    print("\n[STEP 4] Estrazione features nuovo prodotto...")

    target_features = {
        "price": nuovo_prodotto["prezzo"],
        "category_encoded": hash(nuovo_prodotto["categoria"]) % 1000,
        "weight": nuovo_prodotto["peso"],
        "volume": nuovo_prodotto["volume"],
    }
    nuovo_prodotto["features"] = target_features
    print(f"Features estratte: {len(target_features)}")

    # 5. Trova prodotti simili
    print("\n[STEP 5] Ricerca prodotti simili...")

    similar_products = cold_start_forecaster.find_similar_products(
        target_product_info=nuovo_prodotto, products_database=products_database, top_n=2
    )

    if similar_products:
        print(f"Trovati {len(similar_products)} prodotti simili:")
        for sim in similar_products:
            product_name = products_database[sim.source_product]["info"]["nome"]
            print(f"  - {sim.source_product}: {product_name} (sim: {sim.similarity_score:.3f})")
    else:
        print("Nessun prodotto simile trovato")

    # 6. Genera forecast
    print("\n[STEP 6] Generazione forecast Cold Start...")

    try:
        forecast_series, metadata = cold_start_forecaster.cold_start_forecast(
            target_product_info=nuovo_prodotto,
            products_database=products_database,
            forecast_days=30,
            method="hybrid",
        )

        print(f"Forecast generato con metodo: {metadata.get('method', 'unknown')}")

        # Statistiche
        avg_demand = forecast_series.mean()
        total_demand = forecast_series.sum()
        max_demand = forecast_series.max()

        print(f"\nRISULTATI FORECAST:")
        print(f"  Domanda media: {avg_demand:.1f} unita/giorno")
        print(f"  Domanda totale 30gg: {total_demand:.0f} unita")
        print(f"  Picco massimo: {max_demand:.1f} unita")
        print(f"  Affidabilita: {metadata.get('confidence', 'unknown')}")

        # Raccomandazioni business
        print(f"\nRACCOMANDAZIONI:")
        safety_stock = avg_demand * 10  # 10 giorni sicurezza
        total_stock = total_demand + safety_stock
        unit_cost = nuovo_prodotto["prezzo"] * 0.6  # 60% costo
        investment = total_stock * unit_cost

        print(f"  Scorta iniziale consigliata: {total_stock:.0f} unita")
        print(f"  Investimento stimato: EUR {investment:,.0f}")

        # Salva risultati
        print(f"\n[STEP 7] Salvataggio risultati...")

        output_dir = Path(__file__).parent.parent.parent / "outputs" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV forecast
        forecast_export = forecast_series.reset_index()
        forecast_export.columns = ["Data", "Domanda_Prevista"]
        forecast_export["Prodotto"] = nuovo_prodotto["nome"]
        forecast_export["Metodo"] = metadata.get("method", "unknown")

        output_file = output_dir / f"cold_start_demo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        forecast_export.to_csv(output_file, index=False)

        print(f"Risultati salvati: {output_file}")

    except Exception as e:
        print(f"[ERROR] Errore generazione forecast: {e}")
        return False

    print(f"\n[SUCCESS] Demo completata con successo!")
    return True


if __name__ == "__main__":
    demo_cold_start()
