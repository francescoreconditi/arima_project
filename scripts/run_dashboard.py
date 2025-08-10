"""
Script to run the ARIMA Forecaster Streamlit dashboard.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit dashboard."""
    
    # Path to the dashboard main file
    dashboard_path = Path(__file__).parent.parent / "src" / "arima_forecaster" / "dashboard" / "main.py"
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()