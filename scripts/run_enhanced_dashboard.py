#!/usr/bin/env python3
"""
Script per lanciare la Dashboard Evoluta con funzionalità avanzate.

Avvia Streamlit con la nuova dashboard che include:
- Mobile Responsive Design
- Excel Export per Procurement Team
- What-If Scenario Simulator
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Lancia la dashboard evoluta."""

    # Determina il percorso del progetto
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "src" / "arima_forecaster" / "dashboard" / "enhanced_main.py"

    # Verifica che il file esista
    if not dashboard_path.exists():
        print(f"[ERROR] Dashboard file not found: {dashboard_path}")
        sys.exit(1)

    print("[ROCKET] Launching Enhanced ARIMA Dashboard...")
    print(f"[FOLDER] Project root: {project_root}")
    print(f"[CHART] Dashboard: {dashboard_path}")
    print()
    print("[SPARKLES] New Features:")
    print("   [MOBILE] Mobile Responsive Design")
    print("   [EXCEL] Excel Export for Procurement")
    print("   [TARGET] What-If Scenario Simulator")
    print()
    print("[WEB] Dashboard will open at: http://localhost:8501")
    print("[STOP] Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        # Cambia directory al root del progetto
        os.chdir(project_root)

        # Lancia Streamlit
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=127.0.0.1",
            "--server.headless=false",
            "--browser.gatherUsageStats=false",
            "--theme.primaryColor=#FF6B6B",
            "--theme.backgroundColor=#FFFFFF",
            "--theme.secondaryBackgroundColor=#F0F2F6",
        ]

        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\n[STOP] Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error launching dashboard: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
