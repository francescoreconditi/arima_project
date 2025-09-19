"""
Genera file CSV con vendite storiche realistiche per demo Moretti
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_realistic_sales_data():
    """Genera dati di vendite storiche realistici basati sui prodotti Moretti"""

    # Definisci i prodotti con i loro pattern di vendita
    prodotti_config = {
        "CRZ001": {"base_daily": 2.1, "variability": 0.8, "seasonal_factor": 0.15, "trend": 0.002},
        "CRZ002": {"base_daily": 0.8, "variability": 0.4, "seasonal_factor": 0.10, "trend": -0.001},
        "MAT001": {"base_daily": 3.2, "variability": 1.0, "seasonal_factor": 0.20, "trend": 0.003},
        "MAT002": {"base_daily": 4.5, "variability": 1.2, "seasonal_factor": 0.12, "trend": 0.001},
        "RIA001": {"base_daily": 5.8, "variability": 1.5, "seasonal_factor": 0.08, "trend": 0.002},
        "ELT001": {"base_daily": 2.8, "variability": 0.9, "seasonal_factor": 0.18, "trend": 0.004},
        "CRZ003": {"base_daily": 0.3, "variability": 0.2, "seasonal_factor": 0.05, "trend": 0.001},
        "MAT003": {"base_daily": 1.8, "variability": 0.6, "seasonal_factor": 0.15, "trend": 0.002},
        "RIA002": {"base_daily": 8.2, "variability": 2.1, "seasonal_factor": 0.10, "trend": 0.003},
        "ELT002": {"base_daily": 3.5, "variability": 1.1, "seasonal_factor": 0.14, "trend": 0.002},
        "ELT003": {"base_daily": 4.1, "variability": 1.3, "seasonal_factor": 0.25, "trend": 0.005},
        "ASS001": {"base_daily": 0.2, "variability": 0.1, "seasonal_factor": 0.03, "trend": 0.001},
    }

    # Genera 120 giorni di dati storici (4 mesi)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=120)
    date_range = pd.date_range(start=start_date, end=end_date - timedelta(days=1), freq="D")

    # Inizializza DataFrame
    vendite_data = {"data": date_range}

    # Genera dati per ogni prodotto
    for prodotto, config in prodotti_config.items():
        vendite_giornaliere = []

        for i, data in enumerate(date_range):
            # Calcola componenti
            day_of_year = data.timetuple().tm_yday

            # Trend lineare
            trend_component = config["trend"] * i

            # Componente stagionale (simula stagionalità annuale)
            seasonal_component = config["seasonal_factor"] * np.sin(2 * np.pi * day_of_year / 365)

            # Componente ciclica settimanale (meno vendite nei weekend)
            weekly_component = -0.3 if data.weekday() >= 5 else 0.1

            # Media del giorno
            daily_mean = (
                config["base_daily"] + trend_component + seasonal_component + weekly_component
            )

            # Assicura che la media sia positiva
            daily_mean = max(daily_mean, 0.1)

            # Genera vendite con distribuzione Poisson + rumore
            base_sales = np.random.poisson(daily_mean)
            noise = np.random.normal(0, config["variability"])
            final_sales = max(0, int(base_sales + noise))

            # Aggiungi eventi casuali (picchi/cali)
            if np.random.random() < 0.05:  # 5% probabilità evento speciale
                if np.random.random() < 0.7:  # 70% picco, 30% calo
                    final_sales = int(final_sales * np.random.uniform(1.5, 3.0))
                else:
                    final_sales = int(final_sales * np.random.uniform(0.1, 0.5))

            vendite_giornaliere.append(final_sales)

        vendite_data[prodotto] = vendite_giornaliere

    # Crea DataFrame
    df_vendite = pd.DataFrame(vendite_data)
    df_vendite["data"] = pd.to_datetime(df_vendite["data"])

    return df_vendite


def save_vendite_storiche():
    """Salva i dati di vendite storiche in CSV"""

    # Genera i dati
    df_vendite = generate_realistic_sales_data()

    # Salva in CSV
    output_path = Path(__file__).parent / "data" / "vendite_storiche_dettagliate.csv"
    df_vendite.to_csv(output_path, index=False)

    print(f"[OK] File vendite storiche salvato: {output_path}")
    print(f"[INFO] Dati generati: {len(df_vendite)} giorni, {len(df_vendite.columns) - 1} prodotti")

    # Mostra statistiche
    print("\n[STATS] Statistiche Vendite (media giornaliera per prodotto):")
    for col in df_vendite.columns[1:]:  # Escludi colonna data
        media = df_vendite[col].mean()
        std = df_vendite[col].std()
        print(f"  {col}: {media:.1f} ± {std:.1f}")

    return output_path


if __name__ == "__main__":
    save_vendite_storiche()
