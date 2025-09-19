"""
Demo Launcher - Sistema Moretti S.p.A.
Esegue la demo più adatta in base al tempo disponibile
"""

import time
import sys
from pathlib import Path

print("=" * 70)
print("MORETTI S.P.A. - LAUNCHER DEMO SISTEMA SCORTE")
print("=" * 70)

print("\n[INFO] Opzioni demo disponibili:")
print("   1. Demo Veloce (30 secondi) - ARIMA con 3 prodotti")
print("   2. Demo Completa (5+ minuti) - SARIMA enterprise")
print("   3. Demo Semplice (45 secondi) - Singolo prodotto dettagliato")

scelta = input("\n[PROMPT] Seleziona demo (1/2/3) [default=1]: ").strip()

if scelta == "2":
    print("\n[LAUNCHING] Demo Enterprise con SARIMA...")
    print("[WARNING] Questo può richiedere 5-10 minuti...")
    exec(open("moretti_inventory_management.py").read())
elif scelta == "3":
    print("\n[LAUNCHING] Demo Semplice dettagliata...")
    exec(open("moretti_simple_example.py").read())
else:
    print("\n[LAUNCHING] Demo Veloce ottimizzata...")
    exec(open("moretti_inventory_fast.py").read())

print("\n[COMPLETED] Demo terminata con successo!")
print("[INFO] Files di output salvati in outputs/reports/")
