# ============================================
# FILE DI TEST/DEBUG - NON PER PRODUZIONE
# Creato da: Claude Code
# Data: 2025-09-03
# Scopo: Demo Minimum Shelf Life (MSL) Management
# ============================================

"""
Demo completa del sistema Minimum Shelf Life (MSL) Management
per allocazione ottimale inventory ai canali di vendita.

Scenario:
- Prodotto: Yogurt Biologico
- 5 lotti con scadenze diverse (7-120 giorni)
- 6 canali con requisiti MSL diversi
- Ottimizzazione allocazione per massimizzare valore
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Aggiunge il path per import
sys.path.append(str(Path(__file__).parent.parent / "src"))

from arima_forecaster.inventory.balance_optimizer import (
    MinimumShelfLifeManager,
    TipoCanale,
    RequisitoMSL,
    LottoPerishable,
    TipoScadenza,
)


def crea_scenario_demo():
    """Crea scenario demo realistico per MSL"""

    print("=== SCENARIO DEMO: MINIMUM SHELF LIFE MANAGEMENT ===\n")

    # Inizializza MSL Manager
    msl_manager = MinimumShelfLifeManager()

    # === SETUP REQUISITI MSL PER PRODOTTO ===
    prodotto_codice = "YOG001"  # Yogurt Biologico

    # Configura requisiti MSL personalizzati (alcuni canali hanno requisiti più stringenti)
    requisiti_personalizzati = [
        RequisitoMSL(
            canale=TipoCanale.GDO_PREMIUM,
            prodotto_codice=prodotto_codice,
            msl_giorni=100,  # Più stringente del default (90)
            priorita=1,
            note="Esselunga/Coop richiede MSL 100 giorni per biologico",
        ),
        RequisitoMSL(
            canale=TipoCanale.B2B_WHOLESALE,
            prodotto_codice=prodotto_codice,
            msl_giorni=90,  # Meno stringente del default (120)
            priorita=2,
            note="Grossisti accettano MSL ridotto per biologico",
        ),
    ]

    for requisito in requisiti_personalizzati:
        msl_manager.aggiungi_requisito_msl(requisito)

    print("[MSL] REQUISITI MSL CONFIGURATI:")
    print("- GDO Premium: 100 giorni (customizzato)")
    print("- GDO Standard: 60 giorni (default)")
    print("- Retail Tradizionale: 45 giorni (default)")
    print("- Online Diretto: 30 giorni (default)")
    print("- Outlet Sconti: 15 giorni (default)")
    print("- B2B Wholesale: 90 giorni (customizzato)\n")

    return msl_manager, prodotto_codice


def crea_lotti_demo(prodotto_codice: str):
    """Crea lotti demo con scadenze diverse"""

    oggi = datetime.now()

    lotti = [
        LottoPerishable(
            lotto_id="LOT001",
            quantita=500,
            data_produzione=oggi - timedelta(days=110),
            data_scadenza=oggi + timedelta(days=10),
            shelf_life_giorni=120,
            giorni_residui=10,
            percentuale_vita_residua=0.083,
            valore_unitario=2.50,
            rischio_obsolescenza=0.90,
        ),
        LottoPerishable(
            lotto_id="LOT002",
            quantita=800,
            data_produzione=oggi - timedelta(days=90),
            data_scadenza=oggi + timedelta(days=30),
            shelf_life_giorni=120,
            giorni_residui=30,
            percentuale_vita_residua=0.25,
            valore_unitario=2.50,
            rischio_obsolescenza=0.60,
        ),
        LottoPerishable(
            lotto_id="LOT003",
            quantita=1200,
            data_produzione=oggi - timedelta(days=60),
            data_scadenza=oggi + timedelta(days=60),
            shelf_life_giorni=120,
            giorni_residui=60,
            percentuale_vita_residua=0.50,
            valore_unitario=2.50,
            rischio_obsolescenza=0.20,
        ),
        LottoPerishable(
            lotto_id="LOT004",
            quantita=600,
            data_produzione=oggi - timedelta(days=30),
            data_scadenza=oggi + timedelta(days=90),
            shelf_life_giorni=120,
            giorni_residui=90,
            percentuale_vita_residua=0.75,
            valore_unitario=2.50,
            rischio_obsolescenza=0.05,
        ),
        LottoPerishable(
            lotto_id="LOT005",
            quantita=400,
            data_produzione=oggi,
            data_scadenza=oggi + timedelta(days=120),
            shelf_life_giorni=120,
            giorni_residui=120,
            percentuale_vita_residua=1.0,
            valore_unitario=2.50,
            rischio_obsolescenza=0.0,
        ),
    ]

    print("[LOTTI] LOTTI DISPONIBILI:")
    for lotto in lotti:
        print(
            f"- {lotto.lotto_id}: {lotto.quantita} unità, "
            f"scade tra {lotto.giorni_residui} giorni "
            f"({lotto.percentuale_vita_residua:.1%} vita residua)"
        )
    print()

    return lotti


def configura_domanda_e_prezzi():
    """Configura domanda e prezzi per canale"""

    domanda_canali = {
        "gdo_premium": 800,  # Esselunga, Coop
        "gdo_standard": 1200,  # Conad, Despar
        "retail": 600,  # Negozi tradizionali
        "online": 400,  # E-commerce
        "outlet": 300,  # Discount, outlet
        "b2b": 500,  # Grossisti, horeca
    }

    prezzo_canali = {
        "gdo_premium": 4.20,  # Prezzo premium per canali top
        "gdo_standard": 3.80,  # Prezzo standard GDO
        "retail": 3.60,  # Prezzo retail tradizionale
        "online": 3.90,  # Prezzo e-commerce (medio-alto)
        "outlet": 2.80,  # Prezzo discount
        "b2b": 3.20,  # Prezzo wholesale
    }

    print("[PRICES] DOMANDA E PREZZI PER CANALE:")
    for canale in domanda_canali:
        print(
            f"- {canale.replace('_', ' ').title()}: "
            f"{domanda_canali[canale]} unità @ €{prezzo_canali[canale]:.2f}"
        )
    print()

    return domanda_canali, prezzo_canali


def esegui_ottimizzazione_msl():
    """Esegue demo completa ottimizzazione MSL"""

    # Setup scenario
    msl_manager, prodotto_codice = crea_scenario_demo()
    lotti = crea_lotti_demo(prodotto_codice)
    domanda_canali, prezzo_canali = configura_domanda_e_prezzi()

    print("[RUN] AVVIO OTTIMIZZAZIONE ALLOCAZIONE MSL...\n")

    # Esegui ottimizzazione
    risultati = msl_manager.ottimizza_allocazione_lotti(
        lotti_disponibili=lotti, domanda_canali=domanda_canali, prezzo_canali=prezzo_canali
    )

    # === ANALISI RISULTATI ===
    print("[RESULTS] RISULTATI ALLOCAZIONE MSL:\n")

    totale_allocato = 0
    valore_totale = 0

    for canale_id, allocazioni in risultati.items():
        canale_nome = canale_id.replace("_", " ").title()
        quantita_canale = sum(a.quantita_allocata for a in allocazioni)
        valore_canale = sum(a.valore_allocato for a in allocazioni)

        print(f"[CHANNEL] {canale_nome}:")
        print(f"   Quantità totale: {quantita_canale} unità")
        print(f"   Valore totale: €{valore_canale:,.2f}")
        print(f"   Numero lotti: {len(allocazioni)}")

        # Dettaglio lotti
        for alloc in allocazioni:
            print(
                f"     -> {alloc.lotto_id}: {alloc.quantita_allocata} unità, "
                f"{alloc.giorni_shelf_life_residui} giorni residui, "
                f"margine MSL: {alloc.margine_msl} giorni, "
                f"urgenza: {alloc.urgenza.upper()}"
            )

        print()

        totale_allocato += quantita_canale
        valore_totale += valore_canale

    print(f"[SUMMARY] TOTALE:")
    print(
        f"   Quantità allocata: {totale_allocato} / {sum(l.quantita for l in lotti)} unità "
        f"({totale_allocato / sum(l.quantita for l in lotti) * 100:.1f}%)"
    )
    print(f"   Valore totale: €{valore_totale:,.2f}")
    print(f"   Prezzo medio: €{valore_totale / totale_allocato:.2f}/unità\n")

    # === GENERA REPORT DETTAGLIATO ===
    print("[REPORT] DETTAGLIATO MSL:")
    report = msl_manager.genera_report_allocazioni(risultati)

    print(f"   Data report: {report['data_report'][:16]}")
    print(f"   Canali serviti: {report['summary']['numero_canali_serviti']}")
    print(f"   Efficienza allocazione: {report['efficienza_allocazione']}%")
    print(f"   Canale maggior valore: {report['canale_maggior_valore']}")

    print("\n   Distribuzione urgenze:")
    for urgenza, count in report["distribuzione_urgenze"].items():
        if count > 0:
            print(f"     - {urgenza.capitalize()}: {count} allocazioni")

    # === SUGGERIMENTI AZIONI ===
    print("\n[ACTIONS] AZIONI SUGGERITE:")
    azioni = msl_manager.suggerisci_azioni_msl(risultati)

    for i, azione in enumerate(azioni, 1):
        print(f"{i}. [{azione['priorita']}] {azione['tipo']}")
        print(f"   {azione['descrizione']}")
        print(f"   → {azione['azione']}\n")

    if not azioni:
        print("[OK] Nessuna azione critica richiesta. Allocazione ottimale!\n")

    # === TEST CANALI COMPATIBILI ===
    print("[TEST] COMPATIBILITÀ CANALI:")
    test_giorni = [10, 30, 60, 90, 120]

    for giorni in test_giorni:
        canali_compatibili = msl_manager.get_canali_compatibili(prodotto_codice, giorni)
        nomi_canali = [c.value[1].split(" - ")[0] for c in canali_compatibili]
        print(
            f"   Prodotto con {giorni} giorni residui → {len(canali_compatibili)} canali: "
            f"{', '.join(nomi_canali) if nomi_canali else 'Nessuno'}"
        )

    return risultati, report


def analisi_what_if():
    """Analisi what-if per diversi scenari MSL"""

    print("\n" + "=" * 60)
    print("[WHAT-IF] ANALISI SCENARI MSL ALTERNATIVI")
    print("=" * 60)

    msl_manager, prodotto_codice = crea_scenario_demo()
    lotti = crea_lotti_demo(prodotto_codice)
    domanda_canali, prezzo_canali = configura_domanda_e_prezzi()

    # Scenario 1: MSL più stringenti (conservative)
    print("\n[SCENARIO1] MSL CONSERVATIVI (+20 giorni)")
    for canale in TipoCanale:
        if prodotto_codice not in msl_manager.requisiti_msl:
            msl_manager.requisiti_msl[prodotto_codice] = {}
        msl_giorni_conservativo = canale.value[2] + 20
        requisito = RequisitoMSL(
            canale=canale,
            prodotto_codice=prodotto_codice,
            msl_giorni=msl_giorni_conservativo,
            priorita=1,
        )
        msl_manager.requisiti_msl[prodotto_codice][canale.value[0]] = requisito

    risultati_conservativo = msl_manager.ottimizza_allocazione_lotti(
        lotti_disponibili=[l for l in lotti],  # Copia lotti
        domanda_canali=domanda_canali.copy(),
        prezzo_canali=prezzo_canali,
    )

    valore_conservativo = sum(
        sum(a.valore_allocato for a in allocazioni)
        for allocazioni in risultati_conservativo.values()
    )

    print(f"   Valore totale: €{valore_conservativo:,.2f}")
    print(f"   Canali serviti: {len(risultati_conservativo)}")

    # Scenario 2: MSL più permissivi (aggressive)
    print("\n[SCENARIO2] MSL AGGRESSIVI (-15 giorni)")
    msl_manager_aggressivo = MinimumShelfLifeManager()

    for canale in TipoCanale:
        msl_giorni_aggressivo = max(7, canale.value[2] - 15)  # Min 7 giorni
        requisito = RequisitoMSL(
            canale=canale,
            prodotto_codice=prodotto_codice,
            msl_giorni=msl_giorni_aggressivo,
            priorita=1,
        )
        msl_manager_aggressivo.aggiungi_requisito_msl(requisito)

    risultati_aggressivo = msl_manager_aggressivo.ottimizza_allocazione_lotti(
        lotti_disponibili=crea_lotti_demo(prodotto_codice),  # Nuovi lotti
        domanda_canali=domanda_canali.copy(),
        prezzo_canali=prezzo_canali,
    )

    valore_aggressivo = sum(
        sum(a.valore_allocato for a in allocazioni) for allocazioni in risultati_aggressivo.values()
    )

    print(f"   Valore totale: €{valore_aggressivo:,.2f}")
    print(f"   Canali serviti: {len(risultati_aggressivo)}")

    print(f"\n[IMPACT] IMPATTO STRATEGIE MSL:")
    print(f"   Conservativo vs Standard: {valore_conservativo - 13000:+,.0f} €")
    print(f"   Aggressivo vs Standard: {valore_aggressivo - 13000:+,.0f} €")
    print(f"   Delta Conservative-Aggressive: €{valore_conservativo - valore_aggressivo:,.0f}")


if __name__ == "__main__":
    try:
        risultati, report = esegui_ottimizzazione_msl()
        analisi_what_if()

        print("\n[SUCCESS] Demo MSL completata con successo!")
        print("[SAVE] Report salvato in memoria per ulteriori analisi.")

    except Exception as e:
        print(f"[ERROR] Errore durante demo MSL: {e}")
        import traceback

        traceback.print_exc()
