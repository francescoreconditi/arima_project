"""
Demo Sicuro: Dynamic Pricing + VMI Pilot per Moretti S.p.A.
Implementazione graduale con controlli di sicurezza massimi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Import nuove funzionalità v0.5.0
from arima_forecaster.inventory import (
    SafeDynamicPricingManager,
    ControlledVMIPilot,
    DynamicPricingConfig,
    VMIConfig,
    # Classi esistenti per integrazione
    SafetyStockCalculator,
)

# Import core per forecasting
from arima_forecaster import ARIMAForecaster


def crea_dati_demo_moretti():
    """Crea dati realistici per demo Moretti"""

    # 3 prodotti critici Moretti
    prodotti = {
        "CRZ001": {
            "nome": "Carrozzina Standard",
            "prezzo_attuale": 1250.0,
            "inventario_attuale": 120,  # 45 giorni di scorte (eccesso!)
            "domanda_media_giorno": 27.2,
            "categoria": "SLOW_MOVING_ONLY",
            "vendor_data": {
                "market_share": 0.75,
                "reliability_score": 0.88,
                "criticality": "medium",
                "lead_time_days": 5,
                "price_volatility": 0.08,
                "financial_stability": "good"
            }
        },
        "MAT001": {
            "nome": "Materasso Antidecubito",
            "prezzo_attuale": 1650.0,
            "inventario_attuale": 85,   # 32 giorni di scorte (eccesso moderato)
            "domanda_media_giorno": 26.7,
            "categoria": "SLOW_MOVING_ONLY",
            "vendor_data": {
                "market_share": 0.65,
                "reliability_score": 0.82,
                "criticality": "high",  # CRITICAL - non adatto VMI
                "lead_time_days": 10,
                "price_volatility": 0.12,
                "financial_stability": "excellent"
            }
        },
        "ELT001": {
            "nome": "Saturimetro",
            "prezzo_attuale": 180.0,
            "inventario_attuale": 240,  # 48 giorni di scorte (super eccesso!)
            "domanda_media_giorno": 19.2,
            "categoria": "SLOW_MOVING_ONLY",
            "vendor_data": {
                "market_share": 0.45,  # Basso - non adatto VMI
                "reliability_score": 0.79,
                "criticality": "low",
                "lead_time_days": 3,
                "price_volatility": 0.15,
                "financial_stability": "good"
            }
        }
    }

    # Genera dati storici (6 mesi) per ogni prodotto
    date_range = pd.date_range(end=datetime.now(), periods=180, freq='D')

    for prodotto_id, info in prodotti.items():
        # Simula vendite storiche con trend e rumore
        np.random.seed(42)  # Riproducibilità
        base_demand = info["domanda_media_giorno"]

        # Trend decrescente per slow-moving
        trend = np.linspace(base_demand * 1.2, base_demand * 0.8, 180)
        noise = np.random.normal(0, base_demand * 0.3, 180)
        historical_demand = np.maximum(0, trend + noise)

        info["historical_data"] = pd.DataFrame({
            'date': date_range,
            'quantity': historical_demand,
            'inventory_value': historical_demand * info["prezzo_attuale"]
        })

        # Forecast prossimi 30 giorni (per Dynamic Pricing)
        forecast_trend = np.linspace(base_demand * 0.8, base_demand * 0.7, 30)
        forecast_noise = np.random.normal(0, base_demand * 0.2, 30)
        forecast_demand = np.maximum(0, forecast_trend + forecast_noise)

        info["forecast_data"] = pd.Series(forecast_demand)

    return prodotti


def demo_dynamic_pricing():
    """Demo Dynamic Pricing con controlli di sicurezza"""

    print("=" * 60)
    print("[TARGET] DEMO DYNAMIC PRICING - MORETTI S.p.A.")
    print("=" * 60)

    # Configurazione SICURA
    config = DynamicPricingConfig(
        max_discount_percentage=0.08,  # Max 8% sconto
        approved_categories=["SLOW_MOVING_ONLY"],
        min_inventory_excess_threshold=0.25,  # Soglia 25%
        promotion_max_duration_days=14
    )

    pricing_manager = SafeDynamicPricingManager(config)
    prodotti = crea_dati_demo_moretti()

    print("\n[CHART] ANALISI PRICING SUGGESTIONS:")
    print("-" * 40)

    suggestions_approved = []

    for prodotto_id, info in prodotti.items():
        print(f"\n[SEARCH] {info['nome']} ({prodotto_id})")
        print(f"  Prezzo attuale: €{info['prezzo_attuale']:,.0f}")
        print(f"  Inventario: {info['inventario_attuale']} unità")
        print(f"  Domanda media: {info['domanda_media_giorno']:.1f}/giorno")

        # Calcola suggestion
        suggestion = pricing_manager.calculate_suggested_pricing(
            product_id=prodotto_id,
            current_price=info['prezzo_attuale'],
            current_inventory=info['inventario_attuale'],
            forecast_data=info['forecast_data'],
            product_category=info['categoria']
        )

        if suggestion:
            print(f"  [OK] SUGGESTION TROVATA:")
            print(f"    Prezzo suggerito: €{suggestion.suggested_price:.0f}")
            print(f"    Sconto: {abs(suggestion.price_change_percentage):.1%}")
            print(f"    Eccesso inventario: {suggestion.inventory_excess_percentage:.1%}")
            print(f"    Boost domanda atteso: +{suggestion.expected_demand_boost:.1%}")
            print(f"    Rischio: {suggestion.risk_level}")
            print(f"    Richiede approvazione: {suggestion.requires_approval}")

            # Simula approvazione
            if suggestion.risk_level in ["LOW", "MEDIUM"]:
                approval = pricing_manager.approve_pricing_suggestion(
                    suggestion, approver="Manager_Procurement"
                )
                suggestions_approved.append({
                    "prodotto": info['nome'],
                    "saving_stimato": (info['prezzo_attuale'] - suggestion.suggested_price) * info['inventario_attuale'],
                    "suggestion": suggestion
                })
                print(f"    [APPROVED] APPROVATO da {approval['approver']}")
            else:
                print(f"    [REJECT] RIFIUTATO - Rischio troppo alto")
        else:
            print(f"  [SKIP] Nessuna suggestion (criteri non soddisfatti)")

    # Riepilogo business impact
    if suggestions_approved:
        print(f"\n[MONEY] BUSINESS IMPACT STIMATO:")
        print("-" * 30)
        total_inventory_reduction = sum(s['saving_stimato'] for s in suggestions_approved)
        print(f"Riduzione inventario stimata: €{total_inventory_reduction:,.0f}")
        print(f"Prodotti con pricing attivo: {len(suggestions_approved)}")
        print(f"Durata promozioni: {config.promotion_max_duration_days} giorni")

        # Cash flow liberation stimato
        cash_flow_boost = total_inventory_reduction * 0.7  # 70% cash flow boost
        print(f"Cash flow liberation: €{cash_flow_boost:,.0f}")

    # Red flags monitoring
    red_flags = pricing_manager.monitor_red_flags()
    print(f"\n[ALERT] RED FLAGS MONITORING:")
    print(f"Status: {red_flags['recommendation']}")
    print(f"Flags critici: {red_flags['critical_count']}/4")

    return suggestions_approved


def demo_vmi_pilot():
    """Demo VMI Pilot con controlli rigorosi"""

    print("\n" + "=" * 60)
    print("[PARTNER] DEMO VMI PILOT - MORETTI S.p.A.")
    print("=" * 60)

    # Configurazione ULTRACONSERVATIVA
    config = VMIConfig(
        pilot_products=["CRZ001"],  # Solo 1 prodotto
        max_vmi_percentage=0.25,    # Max 25%
        vendor_reliability_score_min=0.85
    )

    vmi_pilot = ControlledVMIPilot(config)
    prodotti = crea_dati_demo_moretti()

    print("\n[LIST] VALUTAZIONE VMI ELIGIBILITY:")
    print("-" * 40)

    candidates_evaluated = []

    for prodotto_id, info in prodotti.items():
        print(f"\n[SEARCH] {info['nome']} ({prodotto_id})")

        # Valuta opportunità VMI
        evaluation = vmi_pilot.evaluate_vmi_opportunity(
            product_id=prodotto_id,
            historical_data=info['historical_data'],
            vendor_data=info['vendor_data']
        )

        print(f"  Eligibile: {'[YES] SÌ' if evaluation.eligible else '[NO] NO'}")
        print(f"  Rischio: {evaluation.risk_level}")
        print(f"  Confidence score: {evaluation.confidence_score:.2f}")
        print(f"  Raccomandazione: {evaluation.recommendation}")

        if evaluation.failed_criteria:
            print(f"  Criteri falliti: {', '.join(evaluation.failed_criteria)}")

        if evaluation.eligible:
            print(f"  [MONEY] BENEFICI STIMATI:")
            for benefit, value in evaluation.estimated_benefits.items():
                if isinstance(value, float) and value > 1000:
                    print(f"    {benefit}: €{value:,.0f}")
                elif isinstance(value, float):
                    print(f"    {benefit}: {value:.1%}")

            if evaluation.estimated_risks:
                print(f"  [WARN] RISCHI:")
                for risk, desc in evaluation.estimated_risks.items():
                    print(f"    {risk}: {desc}")

            candidates_evaluated.append({
                "prodotto": info['nome'],
                "evaluation": evaluation
            })

    # Simula pilota per candidati idonei
    if candidates_evaluated:
        print(f"\n[LAB] SIMULAZIONE PILOTA VMI:")
        print("-" * 30)

        for candidate in candidates_evaluated:
            if candidate['evaluation'].recommendation == "PROCEED_WITH_CAUTION":
                prodotto_id = candidate['evaluation'].product_id
                simulation = vmi_pilot.simulate_vmi_pilot(prodotto_id, duration_days=30)

                print(f"\n[CHART] Pilota: {candidate['prodotto']}")
                print(f"Durata: {simulation['pilot_duration_days']} giorni")
                print(f"Start: {simulation['start_date'].strftime('%Y-%m-%d')}")
                print(f"KPI tracciati: {len(simulation['kpis_tracked'])}")
                print(f"Checkpoint: giorni {simulation['checkpoint_schedule']}")
                print(f"Condizioni stop: {len(simulation['abort_conditions'])}")

                # Benefici stimati
                benefits = candidate['evaluation'].estimated_benefits
                total_savings = benefits.get('cash_flow_liberation_eur', 0) + benefits.get('procurement_cost_saving_eur', 0)
                print(f"[MONEY] Savings totali stimati: €{total_savings:,.0f}")
    else:
        print("[NO] Nessun prodotto idoneo per VMI pilot")

    return candidates_evaluated


def demo_integration_test():
    """Test integrazione con sistemi esistenti"""

    print("\n" + "=" * 60)
    print("[TOOL] TEST INTEGRAZIONE SISTEMI ESISTENTI")
    print("=" * 60)

    try:
        # Test integrazione SafetyStockCalculator
        safety_calc = SafetyStockCalculator()
        print("[OK] SafetyStockCalculator integrato correttamente")

        # Test creazione Dynamic Pricing Manager
        pricing_manager = SafeDynamicPricingManager()
        print("[OK] SafeDynamicPricingManager creato correttamente")

        # Test creazione VMI Pilot
        vmi_pilot = ControlledVMIPilot()
        print("[OK] ControlledVMIPilot creato correttamente")

        # Test forecast con ARIMA
        np.random.seed(42)
        test_data = pd.Series(np.random.poisson(25, 100))  # Dati test

        arima_model = ARIMAForecaster(order=(1,1,1))
        arima_model.fit(test_data)
        forecast = arima_model.predict(30)
        print(f"[OK] ARIMA forecast integrato: {len(forecast)} predictions")

        # Test calcolo eccesso inventario
        excess = pricing_manager.calculate_inventory_excess(
            product_id="TEST001",
            current_inventory=100,
            forecast_data=forecast
        )
        print(f"[OK] Calcolo eccesso inventario: {excess:.1%}")

        print("\n[PARTY] TUTTI I TEST DI INTEGRAZIONE SUPERATI!")

    except Exception as e:
        print(f"[ERROR] ERRORE integrazione: {e}")
        return False

    return True


def main():
    """Demo principale con tutte le funzionalità"""

    print("[MEDICAL] MORETTI S.p.A. - DYNAMIC PRICING + VMI PILOT DEMO")
    print("Implementazione SICURA e GRADUALE")
    print("Versione: v0.5.0 - Advanced Features")

    # Test integrazione prima
    if not demo_integration_test():
        print("[STOP] STOP - Problemi integrazione")
        return

    # Demo Dynamic Pricing
    pricing_results = demo_dynamic_pricing()

    # Demo VMI Pilot
    vmi_results = demo_vmi_pilot()

    # Riepilogo finale
    print("\n" + "=" * 60)
    print("[CHART] RIEPILOGO FINALE")
    print("=" * 60)

    print(f"\nDynamic Pricing:")
    print(f"- Suggestions generate: {len(pricing_results)}")
    if pricing_results:
        total_impact = sum(s['saving_stimato'] for s in pricing_results)
        print(f"- Impatto inventario stimato: €{total_impact:,.0f}")
        print(f"- Cash flow potenziale: €{total_impact * 0.7:,.0f}")

    print(f"\nVMI Pilot:")
    print(f"- Prodotti valutati: {len(vmi_results) if 'vmi_results' in locals() else 0}")
    eligible_count = sum(1 for c in (vmi_results if vmi_results else []) if c['evaluation'].eligible)
    print(f"- Prodotti idonei VMI: {eligible_count}")

    if eligible_count > 0:
        total_vmi_benefits = sum(
            c['evaluation'].estimated_benefits.get('cash_flow_liberation_eur', 0)
            for c in vmi_results if c['evaluation'].eligible
        )
        print(f"- Benefici VMI stimati: €{total_vmi_benefits:,.0f}")

    print(f"\n[BULB] RACCOMANDAZIONI:")
    print(f"1. Iniziare con Dynamic Pricing (rischio basso)")
    print(f"2. Valutare VMI pilot solo se Dynamic Pricing ha successo")
    print(f"3. Monitoraggio red flags continuo")
    print(f"4. Review settimanali per primi 30 giorni")

    print(f"\n[SUCCESS] Demo completato con successo!")
    print(f"Implementazione pronta per fase pilota controllata.")


if __name__ == "__main__":
    main()