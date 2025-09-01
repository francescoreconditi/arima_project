"""
Sistema Avanzato di Bilanciamento Scorte - Overstock vs Stockout
Ottimizzazione totale dei costi di giacenza con ML e analisi predittiva
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import dalla libreria ARIMA
from arima_forecaster import (
    ARIMAForecaster,
    SARIMAForecaster,
    TimeSeriesPreprocessor,
    ModelEvaluator,
    ForecastPlotter
)


# =====================================================
# MODELLI DATI AVANZATI
# =====================================================

class LivelloServizio(Enum):
    """Livelli di servizio target per categoria prodotto"""
    CRITICO = 0.99      # 99% - Prodotti salvavita
    ALTO = 0.95         # 95% - Prodotti essenziali
    MEDIO = 0.90        # 90% - Prodotti standard
    BASSO = 0.85        # 85% - Prodotti accessori


class AlertLevel(Enum):
    """Livelli di alert per gestione scorte"""
    NORMALE = ("verde", "Scorte ottimali", 0)
    ATTENZIONE = ("giallo", "Scorte in diminuzione", 1)
    AVVISO = ("arancione", "Scorte basse", 2)
    CRITICO = ("rosso", "Stockout imminente", 3)
    OVERSTOCK = ("viola", "Eccesso scorte", 4)


class CostiGiacenza(BaseModel):
    """Struttura costi completa per analisi TCO"""
    tasso_capitale: float = Field(0.05, description="Tasso interesse capitale immobilizzato")
    costo_stoccaggio_mq_mese: float = Field(15.0, description="€/m² al mese")
    tasso_obsolescenza_annuo: float = Field(0.02, description="% deperimento annuo")
    costo_stockout_giorno: float = Field(100.0, description="€/giorno mancate vendite")
    costo_ordine_urgente: float = Field(50.0, description="Extra costo ordini urgenti")
    costo_cliente_perso: float = Field(500.0, description="Valore lifetime cliente perso")


class AnalisiRischio(BaseModel):
    """Analisi rischio overstock/stockout"""
    probabilita_stockout: float
    probabilita_overstock: float
    giorni_copertura: int
    inventory_turnover: float
    cash_cycle_days: int
    rischio_obsolescenza: float
    livello_alert: AlertLevel


# =====================================================
# CALCOLATORE SAFETY STOCK DINAMICO
# =====================================================

class SafetyStockCalculator:
    """Calcola safety stock ottimale con metodi avanzati"""
    
    @staticmethod
    def calculate_dynamic_safety_stock(
        demand_mean: float,
        demand_std: float,
        lead_time_days: int,
        service_level: float,
        lead_time_variability: float = 0.1,
        criticality_factor: float = 1.0,
        seasonality_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Calcola safety stock con formula avanzata multi-fattore
        
        Formula: SS = Z * sqrt(LT * σ_d² + d̄² * σ_LT²) * CF * SF
        
        Dove:
        - Z = Z-score per service level
        - LT = Lead time medio
        - σ_d = Deviazione standard domanda
        - d̄ = Domanda media
        - σ_LT = Deviazione standard lead time
        - CF = Fattore criticità
        - SF = Fattore stagionalità
        """
        # Z-score per service level
        z_score = stats.norm.ppf(service_level)
        
        # Variabilità lead time
        lead_time_std = lead_time_days * lead_time_variability
        
        # Formula completa safety stock
        variance_term = (
            lead_time_days * (demand_std ** 2) + 
            (demand_mean ** 2) * (lead_time_std ** 2)
        )
        
        base_safety_stock = z_score * np.sqrt(variance_term)
        
        # Applica fattori moltiplicativi
        adjusted_safety_stock = base_safety_stock * criticality_factor * seasonality_factor
        
        # Calcola anche versioni alternative per confronto
        simple_safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
        
        return {
            'dynamic_safety_stock': round(adjusted_safety_stock),
            'simple_safety_stock': round(simple_safety_stock),
            'base_safety_stock': round(base_safety_stock),
            'z_score': z_score,
            'coverage_days': round(adjusted_safety_stock / demand_mean) if demand_mean > 0 else 0,
            'service_level_actual': service_level
        }
    
    @staticmethod
    def calculate_reorder_point(
        demand_mean: float,
        lead_time_days: int,
        safety_stock: float
    ) -> float:
        """Calcola punto di riordino ottimale"""
        return (demand_mean * lead_time_days) + safety_stock
    
    @staticmethod
    def calculate_economic_order_quantity(
        annual_demand: float,
        ordering_cost: float,
        holding_cost_rate: float,
        unit_cost: float
    ) -> float:
        """
        Calcola EOQ (Economic Order Quantity) con formula Wilson
        
        EOQ = sqrt(2 * D * S / H)
        
        Dove:
        - D = Domanda annuale
        - S = Costo per ordine
        - H = Costo mantenimento (% * costo unitario)
        """
        holding_cost = holding_cost_rate * unit_cost
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return round(eoq)


# =====================================================
# ANALIZZATORE COSTI TOTALI
# =====================================================

class TotalCostAnalyzer:
    """Analizza e ottimizza i costi totali di giacenza"""
    
    def __init__(self, costi: CostiGiacenza):
        self.costi = costi
    
    def calculate_holding_cost(
        self,
        average_inventory: float,
        unit_cost: float,
        space_per_unit: float
    ) -> Dict[str, float]:
        """Calcola costi di mantenimento scorte"""
        # Costo capitale immobilizzato
        capital_cost = average_inventory * unit_cost * self.costi.tasso_capitale
        
        # Costo stoccaggio fisico
        storage_cost = (average_inventory * space_per_unit * 
                       self.costi.costo_stoccaggio_mq_mese * 12)
        
        # Costo obsolescenza
        obsolescence_cost = (average_inventory * unit_cost * 
                           self.costi.tasso_obsolescenza_annuo)
        
        total_holding = capital_cost + storage_cost + obsolescence_cost
        
        return {
            'capital_cost': capital_cost,
            'storage_cost': storage_cost,
            'obsolescence_cost': obsolescence_cost,
            'total_holding_cost': total_holding,
            'holding_cost_per_unit': total_holding / average_inventory if average_inventory > 0 else 0
        }
    
    def calculate_stockout_cost(
        self,
        stockout_probability: float,
        annual_demand: float,
        gross_margin: float
    ) -> Dict[str, float]:
        """Calcola costi di stockout"""
        # Vendite perse attese
        expected_lost_sales = annual_demand * stockout_probability
        
        # Margine perso
        lost_margin = expected_lost_sales * gross_margin
        
        # Costi extra per ordini urgenti
        emergency_orders = stockout_probability * 12 * self.costi.costo_ordine_urgente
        
        # Rischio perdita clienti (stimato)
        customer_loss_risk = stockout_probability * 0.1 * self.costi.costo_cliente_perso
        
        total_stockout = lost_margin + emergency_orders + customer_loss_risk
        
        return {
            'expected_lost_sales': expected_lost_sales,
            'lost_margin': lost_margin,
            'emergency_order_cost': emergency_orders,
            'customer_loss_risk': customer_loss_risk,
            'total_stockout_cost': total_stockout
        }
    
    def find_optimal_inventory_level(
        self,
        demand_forecast: np.ndarray,
        unit_cost: float,
        gross_margin: float,
        space_per_unit: float,
        service_levels: List[float] = None
    ) -> Dict[str, Any]:
        """
        Trova il livello di scorte che minimizza i costi totali
        """
        if service_levels is None:
            service_levels = np.arange(0.85, 0.99, 0.01)
        
        results = []
        demand_mean = np.mean(demand_forecast)
        demand_std = np.std(demand_forecast)
        annual_demand = demand_mean * 365
        
        for sl in service_levels:
            # Calcola safety stock per questo service level
            z_score = stats.norm.ppf(sl)
            safety_stock = z_score * demand_std * np.sqrt(30)  # Assumendo 30 giorni lead time
            
            # Inventario medio = safety stock + metà lotto ordine
            avg_inventory = safety_stock + (annual_demand / 24)  # 24 ordini/anno
            
            # Calcola costi
            holding = self.calculate_holding_cost(avg_inventory, unit_cost, space_per_unit)
            stockout = self.calculate_stockout_cost(1 - sl, annual_demand, gross_margin)
            
            total_cost = holding['total_holding_cost'] + stockout['total_stockout_cost']
            
            results.append({
                'service_level': sl,
                'safety_stock': safety_stock,
                'avg_inventory': avg_inventory,
                'holding_cost': holding['total_holding_cost'],
                'stockout_cost': stockout['total_stockout_cost'],
                'total_cost': total_cost
            })
        
        # Trova ottimo
        results_df = pd.DataFrame(results)
        optimal_idx = results_df['total_cost'].idxmin()
        optimal = results_df.iloc[optimal_idx]
        
        return {
            'optimal_service_level': optimal['service_level'],
            'optimal_safety_stock': optimal['safety_stock'],
            'optimal_avg_inventory': optimal['avg_inventory'],
            'optimal_total_cost': optimal['total_cost'],
            'cost_breakdown': {
                'holding': optimal['holding_cost'],
                'stockout': optimal['stockout_cost']
            },
            'all_results': results_df
        }


# =====================================================
# SISTEMA ALERT INTELLIGENTE
# =====================================================

class InventoryAlertSystem:
    """Sistema di alert multi-livello per gestione scorte"""
    
    @staticmethod
    def check_inventory_status(
        current_stock: float,
        safety_stock: float,
        reorder_point: float,
        max_stock: float,
        daily_demand: float,
        lead_time_days: int
    ) -> AnalisiRischio:
        """
        Analizza stato scorte e genera alert appropriato
        """
        # Calcola metriche chiave
        days_of_supply = current_stock / daily_demand if daily_demand > 0 else 999
        stock_coverage_ratio = current_stock / safety_stock if safety_stock > 0 else 999
        
        # Determina livello alert
        if current_stock <= 0:
            alert = AlertLevel.CRITICO
            prob_stockout = 1.0
        elif current_stock < safety_stock:
            alert = AlertLevel.AVVISO
            prob_stockout = 0.7
        elif current_stock < reorder_point:
            alert = AlertLevel.ATTENZIONE
            prob_stockout = 0.3
        elif current_stock > max_stock * 1.5:
            alert = AlertLevel.OVERSTOCK
            prob_stockout = 0.0
        else:
            alert = AlertLevel.NORMALE
            prob_stockout = 0.1
        
        # Probabilità overstock
        prob_overstock = min(1.0, max(0.0, (current_stock - max_stock) / max_stock))
        
        # Inventory turnover (annualizzato)
        annual_demand = daily_demand * 365
        inventory_turnover = annual_demand / current_stock if current_stock > 0 else 0
        
        # Cash cycle (giorni capitale immobilizzato)
        cash_cycle = days_of_supply + lead_time_days
        
        # Rischio obsolescenza (aumenta con giorni copertura)
        obsolescence_risk = min(1.0, days_of_supply / 180)  # Alto se > 6 mesi
        
        return AnalisiRischio(
            probabilita_stockout=prob_stockout,
            probabilita_overstock=prob_overstock,
            giorni_copertura=int(days_of_supply),
            inventory_turnover=round(inventory_turnover, 2),
            cash_cycle_days=int(cash_cycle),
            rischio_obsolescenza=round(obsolescence_risk, 2),
            livello_alert=alert
        )
    
    @staticmethod
    def generate_action_recommendations(
        analisi: AnalisiRischio,
        current_stock: float,
        reorder_point: float,
        eoq: float
    ) -> List[str]:
        """Genera raccomandazioni azioni basate su analisi"""
        recommendations = []
        
        if analisi.livello_alert == AlertLevel.CRITICO:
            recommendations.append(f"[URGENTE] Effettuare ordine immediato di {eoq} unità")
            recommendations.append("[URGENTE] Contattare fornitori per spedizione express")
            recommendations.append("[URGENTE] Preparare comunicazione clienti per possibili ritardi")
            
        elif analisi.livello_alert == AlertLevel.AVVISO:
            recommendations.append(f"[IMPORTANTE] Pianificare ordine di {eoq} unità entro 2 giorni")
            recommendations.append("[IMPORTANTE] Verificare conferma lead time con fornitore")
            
        elif analisi.livello_alert == AlertLevel.ATTENZIONE:
            qty_to_order = max(0, reorder_point - current_stock + eoq)
            recommendations.append(f"[INFO] Considerare ordine di {qty_to_order} unità")
            recommendations.append("[INFO] Monitorare trend domanda prossimi giorni")
            
        elif analisi.livello_alert == AlertLevel.OVERSTOCK:
            recommendations.append("[AZIONE] Sospendere ordini programmati")
            recommendations.append("[AZIONE] Valutare promozioni per smaltimento scorte")
            recommendations.append("[AZIONE] Verificare possibili resi a fornitore")
            
        else:
            recommendations.append("[OK] Livelli scorte ottimali")
            recommendations.append("[OK] Mantenere monitoraggio standard")
        
        # Raccomandazioni basate su metriche
        if analisi.inventory_turnover < 4:
            recommendations.append(f"[EFFICIENZA] Turnover basso ({analisi.inventory_turnover}x), ridurre lotti ordine")
        
        if analisi.rischio_obsolescenza > 0.5:
            recommendations.append(f"[RISCHIO] Alto rischio obsolescenza ({analisi.rischio_obsolescenza:.0%})")
        
        if analisi.cash_cycle_days > 60:
            recommendations.append(f"[FINANZA] Capitale immobilizzato per {analisi.cash_cycle_days} giorni")
        
        return recommendations


# =====================================================
# FORECAST AVANZATO CON INTERVALLI ADATTIVI
# =====================================================

class AdaptiveForecastEngine:
    """Motore di forecast con intervalli di confidenza adattivi"""
    
    def __init__(self, base_model):
        self.model = base_model
        self.evaluator = ModelEvaluator()
    
    def forecast_with_adaptive_intervals(
        self,
        steps: int,
        historical_volatility: float,
        event_risk_factor: float = 1.0,
        confidence_levels: List[float] = None
    ) -> Dict[str, Any]:
        """
        Genera forecast con intervalli che si adattano a:
        - Volatilità storica
        - Orizzonte temporale
        - Eventi speciali/rischi
        """
        if confidence_levels is None:
            confidence_levels = [0.80, 0.90, 0.95]
        
        # Forecast base
        base_forecast = self.model.forecast(steps=steps, confidence_intervals=False)
        
        # Estrai valori se è un dict
        if isinstance(base_forecast, dict):
            forecast_values = base_forecast.get('forecast', base_forecast)
        else:
            forecast_values = base_forecast
        
        # Calcola errore standard che aumenta con orizzonte
        base_std = historical_volatility
        time_factor = np.sqrt(np.arange(1, steps + 1))
        adaptive_std = base_std * time_factor * event_risk_factor
        
        # Genera intervalli multipli
        intervals = {}
        for cl in confidence_levels:
            z_score = stats.norm.ppf((1 + cl) / 2)
            intervals[f'ci_{int(cl*100)}'] = {
                'lower': forecast_values - z_score * adaptive_std,
                'upper': forecast_values + z_score * adaptive_std
            }
        
        # Calcola metriche incertezza
        uncertainty_metrics = {
            'avg_interval_width': np.mean(intervals['ci_95']['upper'] - intervals['ci_95']['lower']),
            'max_uncertainty': np.max(adaptive_std),
            'uncertainty_growth_rate': (adaptive_std[-1] - adaptive_std[0]) / steps
        }
        
        return {
            'forecast': forecast_values,
            'intervals': intervals,
            'adaptive_std': adaptive_std,
            'uncertainty_metrics': uncertainty_metrics
        }


# =====================================================
# DASHBOARD KPI BILANCIAMENTO
# =====================================================

class InventoryKPIDashboard:
    """Dashboard KPI per monitoraggio bilanciamento scorte"""
    
    @staticmethod
    def calculate_kpis(
        sales_data: pd.DataFrame,
        inventory_data: pd.DataFrame,
        costs_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calcola tutti i KPI chiave per dashboard
        """
        # Fill Rate (% ordini completamente evasi)
        if 'orders_total' in sales_data.columns and 'orders_fulfilled' in sales_data.columns:
            fill_rate = sales_data['orders_fulfilled'].sum() / sales_data['orders_total'].sum()
        else:
            fill_rate = 0.95  # Default stimato
        
        # Inventory Turnover
        cogs = sales_data['quantity'].sum() * costs_data.get('unit_cost', 100)
        avg_inventory = inventory_data['stock_level'].mean() * costs_data.get('unit_cost', 100)
        inventory_turnover = cogs / avg_inventory if avg_inventory > 0 else 0
        
        # Days of Supply (DOS)
        daily_demand = sales_data['quantity'].mean()
        current_inventory = inventory_data['stock_level'].iloc[-1]
        days_of_supply = current_inventory / daily_demand if daily_demand > 0 else 0
        
        # GMROI (Gross Margin Return on Investment)
        gross_margin = costs_data.get('gross_margin', 0.3)
        gmroi = (gross_margin * cogs) / avg_inventory if avg_inventory > 0 else 0
        
        # Cash-to-Cash Cycle
        days_inventory_outstanding = 365 / inventory_turnover if inventory_turnover > 0 else 365
        days_sales_outstanding = costs_data.get('payment_terms', 30)
        days_payable_outstanding = costs_data.get('supplier_terms', 45)
        cash_cycle = days_inventory_outstanding + days_sales_outstanding - days_payable_outstanding
        
        # Perfect Order Rate
        if 'on_time' in sales_data.columns and 'complete' in sales_data.columns:
            perfect_orders = ((sales_data['on_time'] == 1) & 
                            (sales_data['complete'] == 1)).sum()
            perfect_order_rate = perfect_orders / len(sales_data)
        else:
            perfect_order_rate = 0.92  # Default stimato
        
        # Forecast Accuracy (se disponibile)
        if 'actual' in sales_data.columns and 'forecast' in sales_data.columns:
            mape = np.mean(np.abs((sales_data['actual'] - sales_data['forecast']) / 
                                sales_data['actual'])) * 100
            forecast_accuracy = 100 - mape
        else:
            forecast_accuracy = 85  # Default stimato
        
        return {
            'fill_rate': round(fill_rate * 100, 1),
            'inventory_turnover': round(inventory_turnover, 2),
            'days_of_supply': round(days_of_supply, 1),
            'gmroi': round(gmroi, 2),
            'cash_cycle_days': round(cash_cycle, 1),
            'perfect_order_rate': round(perfect_order_rate * 100, 1),
            'forecast_accuracy': round(forecast_accuracy, 1),
            'health_score': InventoryKPIDashboard._calculate_health_score(
                fill_rate, inventory_turnover, days_of_supply, gmroi
            )
        }
    
    @staticmethod
    def _calculate_health_score(
        fill_rate: float,
        turnover: float,
        dos: float,
        gmroi: float
    ) -> str:
        """Calcola health score complessivo"""
        score = 0
        
        # Fill rate (peso 30%)
        if fill_rate > 0.98:
            score += 30
        elif fill_rate > 0.95:
            score += 20
        elif fill_rate > 0.90:
            score += 10
        
        # Turnover (peso 25%)
        if turnover > 12:
            score += 25
        elif turnover > 6:
            score += 15
        elif turnover > 4:
            score += 5
        
        # Days of Supply (peso 25%)
        if 15 <= dos <= 45:
            score += 25
        elif 10 <= dos <= 60:
            score += 15
        elif 5 <= dos <= 90:
            score += 5
        
        # GMROI (peso 20%)
        if gmroi > 3:
            score += 20
        elif gmroi > 2:
            score += 12
        elif gmroi > 1:
            score += 5
        
        # Classificazione
        if score >= 80:
            return "ECCELLENTE"
        elif score >= 60:
            return "BUONO"
        elif score >= 40:
            return "SUFFICIENTE"
        else:
            return "CRITICO"
    
    @staticmethod
    def generate_improvement_suggestions(kpis: Dict[str, Any]) -> List[str]:
        """Genera suggerimenti di miglioramento basati su KPI"""
        suggestions = []
        
        if kpis['fill_rate'] < 95:
            suggestions.append(f"Fill rate basso ({kpis['fill_rate']}%): aumentare safety stock prodotti critici")
        
        if kpis['inventory_turnover'] < 6:
            suggestions.append(f"Turnover lento ({kpis['inventory_turnover']}x): ridurre lotti ordine, aumentare frequenza")
        
        if kpis['days_of_supply'] > 60:
            suggestions.append(f"DOS elevato ({kpis['days_of_supply']}gg): rischio obsolescenza, valutare promozioni")
        elif kpis['days_of_supply'] < 15:
            suggestions.append(f"DOS basso ({kpis['days_of_supply']}gg): rischio stockout, aumentare buffer")
        
        if kpis['gmroi'] < 2:
            suggestions.append(f"GMROI basso ({kpis['gmroi']}): ottimizzare mix prodotti o ridurre scorte")
        
        if kpis['cash_cycle_days'] > 60:
            suggestions.append(f"Ciclo cassa lungo ({kpis['cash_cycle_days']}gg): negoziare termini pagamento")
        
        if kpis['forecast_accuracy'] < 80:
            suggestions.append(f"Accuracy bassa ({kpis['forecast_accuracy']}%): rivedere modelli forecast")
        
        if not suggestions:
            suggestions.append("Performance ottimali su tutti i KPI principali!")
        
        return suggestions


# =====================================================
# ESEMPIO DI UTILIZZO COMPLETO
# =====================================================

def esempio_bilanciamento_completo():
    """Esempio completo di analisi bilanciamento scorte"""
    
    print("=" * 60)
    print("SISTEMA BILANCIAMENTO SCORTE - OVERSTOCK vs STOCKOUT")
    print("=" * 60)
    
    # 1. Setup parametri prodotto esempio
    print("\n[1] CONFIGURAZIONE PRODOTTO")
    print("-" * 40)
    
    prodotto = {
        'nome': 'Carrozzina Standard CRZ001',
        'prezzo_unitario': 280,
        'margine_lordo': 84,  # 30% margine
        'lead_time_giorni': 15,
        'spazio_mq': 0.5,
        'criticita': 'ALTO'
    }
    
    print(f"Prodotto: {prodotto['nome']}")
    print(f"Prezzo: €{prodotto['prezzo_unitario']}")
    print(f"Lead Time: {prodotto['lead_time_giorni']} giorni")
    
    # 2. Genera dati vendite simulati
    print("\n[2] ANALISI STORICO VENDITE")
    print("-" * 40)
    
    np.random.seed(42)
    giorni = 365
    trend = np.linspace(20, 25, giorni)
    stagionalita = 3 * np.sin(np.arange(giorni) * 2 * np.pi / 365)
    rumore = np.random.normal(0, 2, giorni)
    vendite = trend + stagionalita + rumore
    vendite = np.maximum(vendite, 0)  # No vendite negative
    
    vendite_df = pd.DataFrame({
        'data': pd.date_range(start='2024-01-01', periods=giorni),
        'quantita': vendite
    })
    
    print(f"Vendite medie giornaliere: {vendite.mean():.1f} unità")
    print(f"Deviazione standard: {vendite.std():.1f} unità")
    print(f"Vendite annuali: {vendite.sum():.0f} unità")
    
    # 3. Calcolo Safety Stock Dinamico
    print("\n[3] CALCOLO SAFETY STOCK OTTIMALE")
    print("-" * 40)
    
    calculator = SafetyStockCalculator()
    safety_results = calculator.calculate_dynamic_safety_stock(
        demand_mean=vendite.mean(),
        demand_std=vendite.std(),
        lead_time_days=prodotto['lead_time_giorni'],
        service_level=0.95,  # 95% service level
        criticality_factor=1.2,  # Prodotto critico
        seasonality_factor=1.1   # Leggera stagionalità
    )
    
    print(f"Safety Stock Dinamico: {safety_results['dynamic_safety_stock']} unità")
    print(f"Safety Stock Semplice: {safety_results['simple_safety_stock']} unità")
    print(f"Giorni copertura SS: {safety_results['coverage_days']} giorni")
    
    # 4. Calcolo EOQ e Reorder Point
    print("\n[4] PARAMETRI RIORDINO OTTIMALI")
    print("-" * 40)
    
    eoq = calculator.calculate_economic_order_quantity(
        annual_demand=vendite.sum(),
        ordering_cost=50,  # €50 per ordine
        holding_cost_rate=0.25,  # 25% del valore
        unit_cost=prodotto['prezzo_unitario']
    )
    
    reorder_point = calculator.calculate_reorder_point(
        demand_mean=vendite.mean(),
        lead_time_days=prodotto['lead_time_giorni'],
        safety_stock=safety_results['dynamic_safety_stock']
    )
    
    print(f"EOQ (Quantità Ordine Economica): {eoq} unità")
    print(f"Reorder Point: {reorder_point:.0f} unità")
    print(f"Ordini annui previsti: {vendite.sum()/eoq:.1f}")
    
    # 5. Analisi Costi Totali
    print("\n[5] OTTIMIZZAZIONE COSTI TOTALI")
    print("-" * 40)
    
    costi = CostiGiacenza()
    analyzer = TotalCostAnalyzer(costi)
    
    optimal = analyzer.find_optimal_inventory_level(
        demand_forecast=vendite[-30:],  # Ultimi 30 giorni
        unit_cost=prodotto['prezzo_unitario'],
        gross_margin=prodotto['margine_lordo'],
        space_per_unit=prodotto['spazio_mq']
    )
    
    print(f"Service Level Ottimale: {optimal['optimal_service_level']:.1%}")
    print(f"Safety Stock Ottimale: {optimal['optimal_safety_stock']:.0f} unità")
    print(f"Costo Totale Minimo: €{optimal['optimal_total_cost']:.0f}/anno")
    print(f"  - Costi giacenza: €{optimal['cost_breakdown']['holding']:.0f}")
    print(f"  - Costi stockout: €{optimal['cost_breakdown']['stockout']:.0f}")
    
    # 6. Sistema Alert
    print("\n[6] ANALISI RISCHIO E ALERT")
    print("-" * 40)
    
    # Simula stato corrente magazzino
    stock_corrente = 250
    max_stock = eoq + safety_results['dynamic_safety_stock']
    
    alert_system = InventoryAlertSystem()
    analisi_rischio = alert_system.check_inventory_status(
        current_stock=stock_corrente,
        safety_stock=safety_results['dynamic_safety_stock'],
        reorder_point=reorder_point,
        max_stock=max_stock,
        daily_demand=vendite.mean(),
        lead_time_days=prodotto['lead_time_giorni']
    )
    
    print(f"Stock Corrente: {stock_corrente} unità")
    print(f"Livello Alert: {analisi_rischio.livello_alert.value[0].upper()} - {analisi_rischio.livello_alert.value[1]}")
    print(f"Probabilità Stockout: {analisi_rischio.probabilita_stockout:.1%}")
    print(f"Probabilità Overstock: {analisi_rischio.probabilita_overstock:.1%}")
    print(f"Giorni Copertura: {analisi_rischio.giorni_copertura}")
    print(f"Inventory Turnover: {analisi_rischio.inventory_turnover}x")
    
    # 7. Raccomandazioni
    print("\n[7] RACCOMANDAZIONI AZIONI")
    print("-" * 40)
    
    recommendations = alert_system.generate_action_recommendations(
        analisi_rischio, stock_corrente, reorder_point, eoq
    )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # 8. KPI Dashboard
    print("\n[8] KPI DASHBOARD")
    print("-" * 40)
    
    # Prepara dati per KPI (simulati)
    sales_data = vendite_df.copy()
    sales_data.columns = ['date', 'quantity']
    
    inventory_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=30),
        'stock_level': np.random.normal(stock_corrente, 50, 30)
    })
    
    costs_data = {
        'unit_cost': prodotto['prezzo_unitario'],
        'gross_margin': prodotto['margine_lordo'] / prodotto['prezzo_unitario'],
        'payment_terms': 30,
        'supplier_terms': 45
    }
    
    dashboard = InventoryKPIDashboard()
    kpis = dashboard.calculate_kpis(sales_data, inventory_data, costs_data)
    
    print(f"Fill Rate: {kpis['fill_rate']}%")
    print(f"Inventory Turnover: {kpis['inventory_turnover']}x")
    print(f"Days of Supply: {kpis['days_of_supply']} giorni")
    print(f"GMROI: {kpis['gmroi']}")
    print(f"Cash Cycle: {kpis['cash_cycle_days']} giorni")
    print(f"Perfect Order Rate: {kpis['perfect_order_rate']}%")
    print(f"Forecast Accuracy: {kpis['forecast_accuracy']}%")
    print(f"\nHealth Score: {kpis['health_score']}")
    
    # 9. Suggerimenti miglioramento
    print("\n[9] SUGGERIMENTI MIGLIORAMENTO")
    print("-" * 40)
    
    suggestions = dashboard.generate_improvement_suggestions(kpis)
    for i, sug in enumerate(suggestions, 1):
        print(f"{i}. {sug}")
    
    print("\n" + "=" * 60)
    print("ANALISI COMPLETATA CON SUCCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    esempio_bilanciamento_completo()