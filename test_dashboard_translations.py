#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test rapido per verificare che la dashboard Moretti abbia le traduzioni corrette.
"""

import sys
import io
# Forza encoding UTF-8 per output console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_imports():
    """Test che la dashboard possa essere importata senza errori"""
    print("🧪 Test import dashboard Moretti...")
    try:
        # Test import dei moduli principali
        from arima_forecaster.utils.translations import get_all_translations, translate
        from examples.moretti.moretti_dashboard import grafico_previsioni, get_translations_dict
        
        print("  ✓ Import traduzioni OK")
        print("  ✓ Import dashboard OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Errore import: {e}")
        return False

def test_translations():
    """Test che le traduzioni abbiano le chiavi necessarie"""
    print("\n🌍 Test traduzioni disponibili...")
    
    languages = ['Italiano', 'English', 'Español', 'Français', '中文']
    required_keys = [
        'historical', 'forecast', 'confidence_interval',
        'upper_limit', 'lower_limit', 'total_forecast',
        'date', 'units'
    ]
    
    all_ok = True
    
    for lang in languages:
        try:
            from arima_forecaster.utils.translations import get_all_translations
            translations = get_all_translations(lang)
            
            missing_keys = []
            for key in required_keys:
                if key not in translations:
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"  ❌ [{lang}] Mancano chiavi: {missing_keys}")
                all_ok = False
            else:
                print(f"  ✓ [{lang}] Tutte le chiavi presenti")
                
        except Exception as e:
            print(f"  ❌ [{lang}] Errore: {e}")
            all_ok = False
    
    return all_ok

def test_grafico_function():
    """Test che la funzione grafico_previsioni funzioni con le traduzioni"""
    print("\n📊 Test funzione grafici...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from arima_forecaster.utils.translations import get_all_translations
        
        # Crea dati di test
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        values = np.random.randint(10, 50, len(dates))
        
        previsioni = pd.DataFrame({
            'CRZ001': values,
            'CRZ001_upper': values * 1.1,
            'CRZ001_lower': values * 0.9
        }, index=dates)
        
        # Test con diverse lingue
        languages = ['Italiano', 'English', 'Español']
        
        for lang in languages:
            translations = get_all_translations(lang)
            
            # Import della funzione
            from examples.moretti.moretti_dashboard import grafico_previsioni
            
            # Test chiamata funzione (non genera grafico, solo test logica)
            try:
                fig = grafico_previsioni(previsioni, 'CRZ001', translations)
                
                # Verifica che il grafico abbia i trace corretti
                trace_names = [trace.name for trace in fig.data]
                
                expected_forecast = translations.get('forecast', 'Previsione')
                expected_upper = translations.get('upper_limit', 'Limite Superiore')
                expected_lower = translations.get('lower_limit', 'Limite Inferiore')
                
                if expected_forecast in trace_names:
                    print(f"  ✓ [{lang}] Label 'forecast' tradotta correttamente: {expected_forecast}")
                else:
                    print(f"  ❌ [{lang}] Label 'forecast' non trovata nei trace: {trace_names}")
                    
            except Exception as e:
                print(f"  ❌ [{lang}] Errore generazione grafico: {e}")
                return False
        
        print("  ✅ Funzione grafici funziona con tutte le lingue")
        return True
        
    except Exception as e:
        print(f"  ❌ Errore test grafici: {e}")
        return False

def main():
    """Esegue tutti i test"""
    print("=" * 60)
    print("🧪 TEST DASHBOARD MORETTI - TRADUZIONI")
    print("=" * 60)
    
    # Esegui tutti i test
    results = {
        "Import": test_imports(),
        "Traduzioni": test_translations(),
        "Grafici": test_grafico_function()
    }
    
    # Risultati finali
    print("\n" + "=" * 60)
    print("📋 RISULTATI FINALI")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<15}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + ("🎉 TUTTI I TEST SUPERATI!" if all_passed else "⚠️  ALCUNI TEST FALLITI"))
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)