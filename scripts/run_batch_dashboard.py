"""
Script per avviare Batch Forecasting Dashboard
Launcher per business users

Autore: Claude Code
Data: 2025-09-02
"""

import sys
import os
from pathlib import Path
import subprocess
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def check_dependencies():
    """Verifica dipendenze necessarie"""
    required_packages = ["streamlit", "plotly", "pandas", "numpy"]
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Dipendenze mancanti: {', '.join(missing)}")
        print("💡 Installa con: uv sync --all-extras")
        return False

    return True


def main():
    """Main entry point"""
    print("🚀 ARIMA AutoML Batch Forecasting Dashboard")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Set environment for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = "8502"  # Porta diversa da dashboard standard
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

    # Run Streamlit app
    dashboard_path = project_root / "src" / "arima_forecaster" / "ui" / "batch_dashboard.py"

    print(f"📊 Avviando dashboard su http://localhost:8502")
    print(f"📁 Dashboard path: {dashboard_path}")
    print("⏳ Caricamento in corso...")

    try:
        # Launch Streamlit
        cmd = [
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            "8502",
            "--server.address",
            "0.0.0.0",
            "--browser.serverAddress",
            "localhost",
        ]

        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\n👋 Dashboard fermata dall'utente")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore avvio dashboard: {e}")
        print("💡 Verifica che Streamlit sia installato: uv sync --all-extras")
    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")


if __name__ == "__main__":
    main()
