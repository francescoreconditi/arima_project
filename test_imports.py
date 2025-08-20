#!/usr/bin/env python3
"""Test imports after fixing dependencies."""

print("Testing imports...")

# Test core import
try:
    from arima_forecaster import ARIMAForecaster, SARIMAForecaster
    print("[OK] Core models import successful")
except ImportError as e:
    print(f"[ERROR] Core import failed: {e}")

# Test reporting import
try:
    from arima_forecaster.reporting import QuartoReportGenerator
    print("[OK] Reporting import successful")
except ImportError as e:
    print(f"[WARN] Reporting not available: {e}")

# Test if Quarto CLI is available
import subprocess
try:
    result = subprocess.run(['quarto', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] Quarto CLI available: {result.stdout.strip()}")
    else:
        print("[ERROR] Quarto CLI not found")
except FileNotFoundError:
    print("[WARN] Quarto CLI not installed - install from https://quarto.org")

print("Test completed.")