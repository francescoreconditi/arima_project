"""
Script per eseguire la dashboard Streamlit di ARIMA Forecaster.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Esegui la dashboard Streamlit."""

    # Percorso al file principale della dashboard
    dashboard_path = (
        Path(__file__).parent.parent / "src" / "arima_forecaster" / "dashboard" / "main.py"
    )

    # Esegui streamlit
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_path),
                "--server.port=8501",
                "--server.address=localhost",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'esecuzione della dashboard Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
