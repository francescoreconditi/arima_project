# ============================================
# FILE DI TEST/DEBUG - NON PER PRODUZIONE
# Creato da: Claude Code
# Data: 2025-09-02
# Scopo: Demo classificazione e ottimizzazione slow/fast moving
# ============================================

"""
Demo completo del sistema di classificazione Slow/Fast Moving
con ottimizzazione differenziata per categoria
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modulo balance optimizer
from src.arima_forecaster.inventory.balance_optimizer import (
    CostiGiacenza,
    MovementClassifier,
    SlowFastOptimizer,
    SafetyStockCalculator,
    CategoriaMovimentazione,
    ClassificazioneABC,
    ClassificazioneXYZ
)

def genera_dati_prodotti():
    """Genera dataset con prodotti di diverse categorie"""
    
    prodotti = [
        # Fast Moving - Alta rotazione
        {
            'product_id': 'FM001',
            'nome': 'Mascherina FFP2',
            'categoria': 'Protezione',
            'prezzo_unitario': 2.5,
            'domanda_media': 500,  # pezzi/giorno
            'variabilita': 0.2,     # CV basso
            'lead_time': 7,
            'fatturato_annuo': 456250
        },
        # Fast Moving - Media variabilità
        {
            'product_id': 'FM002',
            'nome': 'Guanti Nitrile L',
            'categoria': 'Protezione',
            'prezzo_unitario': 8.0,
            'domanda_media': 200,
            'variabilita': 0.6,     # CV medio
            'lead_time': 10,
            'fatturato_annuo': 584000
        },
        # Medium Moving
        {
            'product_id': 'MM001',
            'nome': 'Termometro Digitale',
            'categoria': 'Diagnostica',
            'prezzo_unitario': 15.0,
            'domanda_media': 50,
            'variabilita': 0.4,
            'lead_time': 14,
            'fatturato_annuo': 273750
        },
        # Slow Moving - Alto valore
        {
            'product_id': 'SM001',
            'nome': 'Defibrillatore AED',
            'categoria': 'Emergenza',
            'prezzo_unitario': 1200.0,
            'domanda_media': 2,
            'variabilita': 0.8,
            'lead_time': 30,
            'fatturato_annuo': 876000
        },
        # Slow Moving - Basso valore
        {
            'product_id': 'SM002',
            'nome': 'Ricambio Valvola Ossigeno',
            'categoria': 'Ricambi',
            'prezzo_unitario': 45.0,
            'domanda_media': 5,
            'variabilita': 1.2,     # CV alto (erratico)
            'lead_time': 21,
            'fatturato_annuo': 82125
        },
        # Very Slow Moving
        {
            'product_id': 'VS001',
            'nome': 'Kit Chirurgico Specialistico',
            'categoria': 'Chirurgia',
            'prezzo_unitario': 850.0,
            'domanda_media': 0.5,   # 1 ogni 2 giorni
            'variabilita': 1.5,
            'lead_time': 45,
            'fatturato_annuo': 155125
        }
    ]
    
    return pd.DataFrame(prodotti)

def genera_storico_vendite(prodotto, giorni=365):
    """Genera storico vendite simulato per un prodotto"""
    
    np.random.seed(42 + hash(prodotto['product_id']) % 100)
    
    # Base demand
    domanda_base = prodotto['domanda_media']
    std_dev = domanda_base * prodotto['variabilita']
    
    # Genera serie con trend e stagionalità
    t = np.arange(giorni)
    trend = domanda_base * (1 + 0.001 * t)  # Leggero trend crescente
    stagionalita = domanda_base * 0.1 * np.sin(2 * np.pi * t / 365)  # Stagionalità annuale
    rumore = np.random.normal(0, std_dev, giorni)
    
    vendite = trend + stagionalita + rumore
    vendite = np.maximum(vendite, 0)  # No vendite negative
    
    return vendite

def analizza_prodotto(prodotto, vendite, costi):
    """Analisi completa di un singolo prodotto"""
    
    print(f"\n{'='*60}")
    print(f"PRODOTTO: {prodotto['nome']} ({prodotto['product_id']})")
    print(f"{'='*60}")
    
    # 1. Statistiche base
    print("\n[1] STATISTICHE VENDITE")
    print("-" * 40)
    print(f"Domanda media: {np.mean(vendite):.1f} unità/giorno")
    print(f"Deviazione standard: {np.std(vendite):.1f}")
    print(f"Coefficiente Variazione: {np.std(vendite)/np.mean(vendite):.2f}")
    print(f"Vendite annuali: {np.sum(vendite):.0f} unità")
    
    # 2. Classificazione movimento
    classifier = MovementClassifier()
    
    # Calcola turnover
    annual_demand = np.sum(vendite)
    avg_inventory = prodotto['domanda_media'] * 15  # Assumiamo 15gg stock medio
    turnover = annual_demand / avg_inventory if avg_inventory > 0 else 0
    
    categoria_movimento = classifier.classify_by_movement(turnover)
    classe_xyz = classifier.classify_xyz(pd.Series(vendite))
    
    print("\n[2] CLASSIFICAZIONE")
    print("-" * 40)
    print(f"Categoria Movimento: {categoria_movimento.value[1]}")
    print(f"Turnover annuo: {turnover:.1f}x")
    print(f"Classe XYZ: {classe_xyz.value[1]}")
    
    # 3. Classificazione ABC (simulata)
    fatturato_totale = 2527375  # Somma fatturati tutti prodotti
    pct_fatturato = prodotto['fatturato_annuo'] / fatturato_totale
    
    if pct_fatturato > 0.3:
        classe_abc = ClassificazioneABC.A
    elif pct_fatturato > 0.1:
        classe_abc = ClassificazioneABC.B
    else:
        classe_abc = ClassificazioneABC.C
    
    print(f"Classe ABC: {classe_abc.value[1]} ({pct_fatturato:.1%} del fatturato)")
    
    # 4. Strategia raccomandata
    strategia = classifier.get_strategy_by_classification(
        categoria_movimento,
        classe_abc,
        classe_xyz
    )
    
    print("\n[3] STRATEGIA RACCOMANDATA")
    print("-" * 40)
    print(f"Strategia: {strategia['strategia']}")
    print(f"Service Level: {strategia['service_level']:.0%}")
    print(f"Revisione: {strategia['review_period']}")
    print(f"Politica Ordini: {strategia['ordering_policy']}")
    
    # 5. Ottimizzazione specifica
    optimizer = SlowFastOptimizer(costi)
    
    if categoria_movimento in [CategoriaMovimentazione.SLOW_MOVING, CategoriaMovimentazione.VERY_SLOW]:
        ottimizzazione = optimizer.optimize_slow_moving(
            vendite[-30:],
            prodotto['prezzo_unitario'],
            prodotto['lead_time']
        )
        tipo_ottimizzazione = "SLOW MOVING"
    else:
        ottimizzazione = optimizer.optimize_fast_moving(
            vendite[-30:],
            prodotto['prezzo_unitario'],
            prodotto['lead_time']
        )
        tipo_ottimizzazione = "FAST MOVING"
    
    print(f"\n[4] OTTIMIZZAZIONE {tipo_ottimizzazione}")
    print("-" * 40)
    print(f"Safety Stock: {ottimizzazione['safety_stock']} unità")
    print(f"EOQ: {ottimizzazione['eoq']} unità")
    print(f"Reorder Point: {ottimizzazione['reorder_point']} unità")
    print(f"Ordini annui: {ottimizzazione['annual_orders']:.1f}")
    
    if 'obsolescence_risk' in ottimizzazione:
        print(f"Rischio Obsolescenza: {ottimizzazione['obsolescence_risk']:.1%}")
    if 'make_to_order_threshold' in ottimizzazione:
        print(f"Soglia Make-to-Order: {ottimizzazione['make_to_order_threshold']:.0f} unità")
    if 'cycle_time_days' in ottimizzazione:
        print(f"Ciclo riordino: {ottimizzazione['cycle_time_days']} giorni")
    
    # 6. Calcolo risparmio
    print("\n[5] ANALISI ECONOMICA")
    print("-" * 40)
    
    # Costo attuale (politica standard)
    safety_stock_standard = 2 * np.std(vendite) * np.sqrt(prodotto['lead_time'])
    costo_standard = safety_stock_standard * prodotto['prezzo_unitario'] * costi.tasso_capitale
    
    # Costo ottimizzato
    costo_ottimizzato = ottimizzazione['safety_stock'] * prodotto['prezzo_unitario'] * costi.tasso_capitale
    
    risparmio = costo_standard - costo_ottimizzato
    risparmio_pct = (risparmio / costo_standard * 100) if costo_standard > 0 else 0
    
    print(f"Costo giacenza standard: €{costo_standard:.0f}/anno")
    print(f"Costo giacenza ottimizzato: €{costo_ottimizzato:.0f}/anno")
    print(f"Risparmio potenziale: €{risparmio:.0f}/anno ({risparmio_pct:.1f}%)")
    
    return {
        'product_id': prodotto['product_id'],
        'nome': prodotto['nome'],
        'categoria_movimento': categoria_movimento.value[0],
        'classe_abc': classe_abc.value[0],
        'classe_xyz': classe_xyz.value[0],
        'turnover': turnover,
        'safety_stock': ottimizzazione['safety_stock'],
        'eoq': ottimizzazione['eoq'],
        'risparmio_annuo': risparmio
    }

def main():
    """Esecuzione demo completa"""
    
    print("=" * 60)
    print("DEMO SISTEMA CLASSIFICAZIONE SLOW/FAST MOVING")
    print("Con ottimizzazione differenziata per categoria")
    print("=" * 60)
    
    # Setup costi
    costi = CostiGiacenza(
        tasso_capitale=0.08,  # 8% costo capitale
        costo_stoccaggio_mq_mese=20.0,
        tasso_obsolescenza_annuo=0.03,
        costo_stockout_giorno=150.0,
        costo_ordine_urgente=75.0,
        costo_cliente_perso=1000.0
    )
    
    # Carica prodotti
    prodotti_df = genera_dati_prodotti()
    
    # Analizza ogni prodotto
    risultati = []
    for _, prodotto in prodotti_df.iterrows():
        vendite = genera_storico_vendite(prodotto)
        risultato = analizza_prodotto(prodotto, vendite, costi)
        risultati.append(risultato)
    
    # Riepilogo finale
    print("\n" + "=" * 60)
    print("RIEPILOGO PORTFOLIO PRODOTTI")
    print("=" * 60)
    
    risultati_df = pd.DataFrame(risultati)
    
    # Tabella riepilogativa
    print("\nClassificazione Prodotti:")
    print("-" * 40)
    for _, row in risultati_df.iterrows():
        print(f"{row['product_id']}: {row['nome'][:30]:<30} | "
              f"Mov: {row['categoria_movimento']:<10} | "
              f"ABC: {row['classe_abc']} | "
              f"XYZ: {row['classe_xyz']}")
    
    # Statistiche aggregate
    print("\nStatistiche Portfolio:")
    print("-" * 40)
    
    # Conta per categoria movimento
    for cat in ['fast', 'medium', 'slow', 'very_slow']:
        count = len(risultati_df[risultati_df['categoria_movimento'] == cat])
        if count > 0:
            print(f"{cat.upper()}: {count} prodotti")
    
    # Risparmio totale
    risparmio_totale = risultati_df['risparmio_annuo'].sum()
    print(f"\nRisparmio Totale Potenziale: €{risparmio_totale:,.0f}/anno")
    
    # Raccomandazioni finali
    print("\n" + "=" * 60)
    print("RACCOMANDAZIONI IMPLEMENTAZIONE")
    print("=" * 60)
    
    print("\n1. PRIORITÀ IMMEDIATA (Quick Wins):")
    print("   - Implementare politiche slow moving per prodotti VS001, SM002")
    print("   - Ridurre safety stock del 50% per slow movers")
    print("   - Passare a make-to-order per very slow items")
    
    print("\n2. AZIONI MEDIO TERMINE:")
    print("   - Negoziare lotti minimi più bassi con fornitori per slow movers")
    print("   - Implementare continuous review per fast movers critici")
    print("   - Consolidare ordini multi-prodotto stesso fornitore")
    
    print("\n3. OTTIMIZZAZIONI AVANZATE:")
    print("   - Sistema alert automatico per cambio categoria (slow->fast)")
    print("   - Forecast differenziato per classe XYZ")
    print("   - Dynamic safety stock basato su forecast accuracy")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETATA CON SUCCESSO!")
    print("=" * 60)

if __name__ == "__main__":
    main()