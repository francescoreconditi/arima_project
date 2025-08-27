"""
Esempio pratico Cold Start Problem - Moretti S.p.A.
Dimostra come utilizzare le funzioni Cold Start per nuovi prodotti
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Aggiungi il path del modulo
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from arima_forecaster.core.cold_start import ColdStartForecaster

def load_moretti_data():
    """Carica i dati demo di Moretti"""
    try:
        # Path ai file dati
        data_dir = Path(__file__).parent / "data"
        
        # Genera dati demo se non esistono
        if not (data_dir / "vendite_storiche.csv").exists():
            print("[DATA] Generazione dati demo...")
            generate_demo_data(data_dir)
        
        # Carica dati vendite
        vendite_df = pd.read_csv(data_dir / "vendite_storiche.csv", index_col=0, parse_dates=True)
        
        # Carica info prodotti
        prodotti_df = pd.read_csv(data_dir / "prodotti_config.csv", index_col=0)
        
        return vendite_df, prodotti_df
        
    except Exception as e:
        print(f"[ERROR] Errore caricamento dati: {e}")
        return None, None

def generate_demo_data(data_dir):
    """Genera dati demo per l'esempio"""
    data_dir.mkdir(exist_ok=True)
    
    # Genera 90 giorni di dati vendite
    dates = pd.date_range(start='2024-05-01', end='2024-07-30', freq='D')
    
    # Prodotti con caratteristiche diverse
    products = {
        'CRZ001': {'base_demand': 25, 'seasonality': 0.2, 'trend': 0.01},
        'MAT001': {'base_demand': 20, 'seasonality': 0.15, 'trend': -0.005},
        'ELT001': {'base_demand': 15, 'seasonality': 0.3, 'trend': 0.02}
    }
    
    vendite_data = {}
    
    for product, params in products.items():
        # Serie con trend, stagionalità e noise
        base = params['base_demand']
        trend = params['trend']
        seasonality = params['seasonality']
        
        values = []
        for i, date in enumerate(dates):
            # Componente trend
            trend_component = base + (trend * i)
            
            # Stagionalità settimanale
            weekly_effect = seasonality * np.sin(2 * np.pi * i / 7)
            
            # Noise
            noise = np.random.normal(0, base * 0.1)
            
            # Valore finale
            value = max(0, trend_component + weekly_effect + noise)
            values.append(value)
        
        vendite_data[product] = values
    
    # Salva vendite
    vendite_df = pd.DataFrame(vendite_data, index=dates)
    vendite_df.to_csv(data_dir / "vendite_storiche.csv")
    
    # Info prodotti
    prodotti_info = pd.DataFrame({
        'nome': [
            'Carrozzina Standard',
            'Materasso Antidecubito',
            'Saturimetro Digitale'
        ],
        'categoria': [
            'Carrozzine',
            'Materassi Antidecubito', 
            'Elettromedicali'
        ],
        'prezzo': [450.0, 250.0, 89.0],
        'peso': [12.0, 8.0, 0.3],
        'volume': [150.0, 50.0, 2.0]
    }, index=['CRZ001', 'MAT001', 'ELT001'])
    
    prodotti_info.to_csv(data_dir / "prodotti_config.csv")
    
    print("[OK] Dati demo generati")

def cold_start_example():
    """Esempio completo di Cold Start Problem"""
    
    print("[ROCKET] ESEMPIO COLD START PROBLEM - MORETTI S.p.A.")
    print("=" * 60)
    
    # 1. Carica dati esistenti
    print("\n📊 Caricamento dati prodotti esistenti...")
    vendite_df, prodotti_df = load_moretti_data()
    
    if vendite_df is None or prodotti_df is None:
        print("❌ Errore caricamento dati. Uscita.")
        return
    
    print(f"✅ Caricati {len(vendite_df.columns)} prodotti con {len(vendite_df)} giorni di storia")
    print(f"   Prodotti: {', '.join(vendite_df.columns)}")
    
    # 2. Definisci nuovo prodotto
    print("\n🆕 Definizione nuovo prodotto...")
    new_product = {
        'codice': 'CRZ-ULTRA-001',
        'nome': 'Carrozzina Ultra-Light Premium',
        'categoria': 'Carrozzine',
        'prezzo': 890.0,  # Premium vs CRZ001 standard
        'peso': 8.5,      # Più leggera
        'volume': 120.0,  # Più compatta
        'expected_demand': 18.0
    }
    
    print(f"   Nuovo prodotto: {new_product['nome']}")
    print(f"   Categoria: {new_product['categoria']}")
    print(f"   Prezzo: €{new_product['prezzo']}")
    
    # 3. Prepara database prodotti per Cold Start
    print("\n🔍 Preparazione database prodotti...")
    
    cold_start_forecaster = ColdStartForecaster(similarity_threshold=0.6)
    products_database = {}
    
    for product_code in vendite_df.columns:
        # Dati vendite
        sales_data = vendite_df[product_code].dropna()
        
        # Info prodotto
        product_info = prodotti_df.loc[product_code].to_dict()
        
        # Estrai features
        features = cold_start_forecaster.extract_product_features(sales_data, product_info)
        
        products_database[product_code] = {
            'vendite': sales_data,
            'info': product_info,
            'features': features
        }
        
        print(f"   ✓ {product_code}: {len(sales_data)} giorni, features: {len(features)}")
    
    # 4. Prepara info nuovo prodotto con features
    print("\n⚙️ Estrazione features nuovo prodotto...")
    
    target_features = {
        'price': new_product['prezzo'],
        'category_encoded': hash(new_product['categoria']) % 1000,
        'weight': new_product['peso'],
        'volume': new_product['volume'],
        'expected_demand_level': new_product['expected_demand']
    }
    
    new_product['features'] = target_features
    print(f"   ✓ Estratte {len(target_features)} features per matching")
    
    # 5. Trova prodotti simili
    print("\n🎯 Ricerca prodotti simili...")
    
    similar_products = cold_start_forecaster.find_similar_products(
        target_product_info=new_product,
        products_database=products_database,
        top_n=3
    )
    
    if similar_products:
        print(f"   ✅ Trovati {len(similar_products)} prodotti simili:")
        for sim in similar_products:
            product_name = products_database[sim.source_product]['info']['nome']
            print(f"      • {sim.source_product}: {product_name} (similarità: {sim.similarity_score:.3f})")
    else:
        print("   ⚠️ Nessun prodotto simile trovato")
    
    # 6. Genera forecasts con diversi metodi
    print("\n🔮 Generazione forecast Cold Start...")
    
    methods = ['pattern', 'analogical', 'hybrid']
    results = {}
    
    for method in methods:
        print(f"\n   🧮 Metodo: {method.upper()}")
        try:
            forecast_series, metadata = cold_start_forecaster.cold_start_forecast(
                target_product_info=new_product,
                products_database=products_database,
                forecast_days=30,
                method=method
            )
            
            results[method] = {
                'forecast': forecast_series,
                'metadata': metadata
            }
            
            # Statistiche forecast
            avg_demand = forecast_series.mean()
            total_demand = forecast_series.sum()
            max_demand = forecast_series.max()
            confidence = metadata.get('confidence', 'unknown')
            
            print(f"      ✓ Domanda media: {avg_demand:.1f} unità/giorno")
            print(f"      ✓ Domanda totale 30gg: {total_demand:.0f} unità")
            print(f"      ✓ Picco massimo: {max_demand:.1f} unità")
            print(f"      ✓ Affidabilità: {confidence}")
            
        except Exception as e:
            print(f"      ❌ Errore metodo {method}: {e}")
    
    # 7. Confronto risultati
    print("\n📊 CONFRONTO RISULTATI")
    print("-" * 40)
    
    comparison_data = []
    for method, result in results.items():
        forecast = result['forecast']
        metadata = result['metadata']
        
        comparison_data.append({
            'Metodo': method.title(),
            'Domanda Media': f"{forecast.mean():.1f}",
            'Totale 30gg': f"{forecast.sum():.0f}",
            'Affidabilità': metadata.get('confidence', 'unknown'),
            'Prodotti Usati': len(metadata.get('source_products', []))
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 8. Raccomandazioni
    print("\n💡 RACCOMANDAZIONI BUSINESS")
    print("-" * 40)
    
    if results:
        # Usa il metodo hybrid se disponibile, altrimenti il migliore
        best_method = 'hybrid' if 'hybrid' in results else list(results.keys())[0]
        best_forecast = results[best_method]['forecast']
        
        avg_demand = best_forecast.mean()
        total_30days = best_forecast.sum()
        
        # Calcola scorta di sicurezza (15 giorni)
        safety_stock = avg_demand * 15
        
        # Calcola investimento iniziale stimato
        unit_cost = new_product['prezzo'] * 0.6  # Stima 60% costo produzione
        initial_investment = (total_30days + safety_stock) * unit_cost
        
        print(f"📈 Metodo raccomandato: {best_method.upper()}")
        print(f"📦 Scorta iniziale consigliata: {total_30days + safety_stock:.0f} unità")
        print(f"   - Vendite previste 30gg: {total_30days:.0f}")
        print(f"   - Scorta sicurezza (15gg): {safety_stock:.0f}")
        print(f"💰 Investimento scorte iniziali: €{initial_investment:,.0f}")
        print(f"📅 Break-even stimato: {(total_30days + safety_stock) / avg_demand:.0f} giorni")
        
        # Analisi del rischio
        demand_volatility = best_forecast.std() / best_forecast.mean()
        if demand_volatility < 0.2:
            risk_level = "🟢 BASSO"
        elif demand_volatility < 0.4:
            risk_level = "🟡 MEDIO"  
        else:
            risk_level = "🔴 ALTO"
            
        print(f"⚠️ Livello rischio: {risk_level} (CV: {demand_volatility:.2f})")
    
    # 9. Salva risultati
    print("\n💾 Salvataggio risultati...")
    
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if results:
        # Salva forecast del metodo migliore
        best_method = 'hybrid' if 'hybrid' in results else list(results.keys())[0]
        best_forecast = results[best_method]['forecast']
        
        # Prepara CSV
        forecast_export = best_forecast.reset_index()
        forecast_export.columns = ['Data', 'Domanda_Prevista']
        forecast_export['Prodotto_Codice'] = new_product['codice']
        forecast_export['Prodotto_Nome'] = new_product['nome']
        forecast_export['Metodo'] = best_method
        forecast_export['Categoria'] = new_product['categoria']
        forecast_export['Prezzo'] = new_product['prezzo']
        
        output_file = output_dir / f"cold_start_forecast_{new_product['codice']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        forecast_export.to_csv(output_file, index=False)
        
        print(f"   ✅ Forecast salvato: {output_file}")
        
        # Salva metadata
        metadata_file = output_dir / f"cold_start_metadata_{new_product['codice']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(results[best_method]['metadata'], f, indent=2, default=str)
        
        print(f"   ✅ Metadata salvato: {metadata_file}")
    
    print("\n🎉 ESEMPIO COMPLETATO!")
    print("=" * 60)

if __name__ == "__main__":
    cold_start_example()