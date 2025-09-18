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

warnings.filterwarnings("ignore")

# Import dalla libreria ARIMA
from arima_forecaster import (
    ARIMAForecaster,
    SARIMAForecaster,
    TimeSeriesPreprocessor,
    ModelEvaluator,
    ForecastPlotter,
)


# =====================================================
# MODELLI DATI AVANZATI
# =====================================================


class LivelloServizio(Enum):
    """Livelli di servizio target per categoria prodotto"""

    CRITICO = 0.99  # 99% - Prodotti salvavita
    ALTO = 0.95  # 95% - Prodotti essenziali
    MEDIO = 0.90  # 90% - Prodotti standard
    BASSO = 0.85  # 85% - Prodotti accessori


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


class CategoriaMovimentazione(Enum):
    """Classificazione prodotti per velocità di movimento"""

    FAST_MOVING = ("fast", "Alta rotazione", 12)  # Turnover > 12x/anno
    MEDIUM_MOVING = ("medium", "Media rotazione", 6)  # Turnover 6-12x/anno
    SLOW_MOVING = ("slow", "Bassa rotazione", 3)  # Turnover 3-6x/anno
    VERY_SLOW = ("very_slow", "Bassissima rotazione", 1)  # Turnover < 3x/anno


class ClassificazioneABC(Enum):
    """Classificazione ABC per valore economico"""

    A = ("A", "Alto valore - 80% fatturato", 0.8)
    B = ("B", "Medio valore - 15% fatturato", 0.15)
    C = ("C", "Basso valore - 5% fatturato", 0.05)


class ClassificazioneXYZ(Enum):
    """Classificazione XYZ per variabilità domanda"""

    X = ("X", "Domanda stabile - CV < 0.5", 0.5)
    Y = ("Y", "Domanda variabile - CV 0.5-1.0", 1.0)
    Z = ("Z", "Domanda erratica - CV > 1.0", 999)


class ProfiloProdotto(BaseModel):
    """Profilo completo prodotto con classificazioni"""

    codice: str
    nome: str
    categoria_movimento: CategoriaMovimentazione
    classe_abc: ClassificazioneABC
    classe_xyz: ClassificazioneXYZ
    turnover_annuo: float
    coefficiente_variazione: float
    valore_giacenza_media: float
    percentuale_fatturato: float
    strategia_suggerita: str


class AnalisiRischio(BaseModel):
    """Analisi rischio overstock/stockout"""

    probabilita_stockout: float
    probabilita_overstock: float
    giorni_copertura: int
    inventory_turnover: float
    cash_cycle_days: int
    rischio_obsolescenza: float
    livello_alert: AlertLevel
    categoria_movimento: Optional[CategoriaMovimentazione] = None


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
        seasonality_factor: float = 1.0,
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
        variance_term = lead_time_days * (demand_std**2) + (demand_mean**2) * (lead_time_std**2)

        base_safety_stock = z_score * np.sqrt(variance_term)

        # Applica fattori moltiplicativi
        adjusted_safety_stock = base_safety_stock * criticality_factor * seasonality_factor

        # Calcola anche versioni alternative per confronto
        simple_safety_stock = z_score * demand_std * np.sqrt(lead_time_days)

        return {
            "dynamic_safety_stock": round(adjusted_safety_stock),
            "simple_safety_stock": round(simple_safety_stock),
            "base_safety_stock": round(base_safety_stock),
            "z_score": z_score,
            "coverage_days": round(adjusted_safety_stock / demand_mean) if demand_mean > 0 else 0,
            "service_level_actual": service_level,
        }

    @staticmethod
    def calculate_reorder_point(
        demand_mean: float, lead_time_days: int, safety_stock: float
    ) -> float:
        """Calcola punto di riordino ottimale"""
        return (demand_mean * lead_time_days) + safety_stock

    @staticmethod
    def calculate_economic_order_quantity(
        annual_demand: float, ordering_cost: float, holding_cost_rate: float, unit_cost: float
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
# CLASSIFICATORE SLOW/FAST MOVING
# =====================================================


class MovementClassifier:
    """Classificatore prodotti per velocità movimento e variabilità"""

    @staticmethod
    def classify_by_movement(turnover: float) -> CategoriaMovimentazione:
        """Classifica prodotto per velocità di movimento basata su turnover"""
        if turnover >= 12:
            return CategoriaMovimentazione.FAST_MOVING
        elif turnover >= 6:
            return CategoriaMovimentazione.MEDIUM_MOVING
        elif turnover >= 3:
            return CategoriaMovimentazione.SLOW_MOVING
        else:
            return CategoriaMovimentazione.VERY_SLOW

    @staticmethod
    def classify_abc(products_df: pd.DataFrame, value_column: str = "fatturato") -> pd.DataFrame:
        """
        Classifica prodotti secondo analisi ABC (Pareto)

        Args:
            products_df: DataFrame con dati prodotti
            value_column: Colonna da usare per classificazione (fatturato, margine, etc.)
        """
        df = products_df.copy()

        # Calcola percentuale cumulativa
        df = df.sort_values(value_column, ascending=False)
        df["cumulative_value"] = df[value_column].cumsum()
        df["cumulative_pct"] = df["cumulative_value"] / df[value_column].sum()

        # Assegna classi ABC
        df["classe_abc"] = pd.cut(
            df["cumulative_pct"], bins=[0, 0.8, 0.95, 1.0], labels=["A", "B", "C"]
        )

        return df

    @staticmethod
    def classify_xyz(demand_series: pd.Series) -> ClassificazioneXYZ:
        """
        Classifica per variabilità domanda usando coefficiente di variazione

        CV = σ / μ
        """
        mean_demand = demand_series.mean()
        std_demand = demand_series.std()

        if mean_demand == 0:
            return ClassificazioneXYZ.Z

        cv = std_demand / mean_demand

        if cv < 0.5:
            return ClassificazioneXYZ.X
        elif cv <= 1.0:
            return ClassificazioneXYZ.Y
        else:
            return ClassificazioneXYZ.Z

    @staticmethod
    def get_strategy_by_classification(
        movimento: CategoriaMovimentazione, abc: ClassificazioneABC, xyz: ClassificazioneXYZ
    ) -> Dict[str, Any]:
        """
        Definisce strategia ottimale basata su classificazione combinata
        """
        strategies = {
            # Fast Moving + Classe A
            (CategoriaMovimentazione.FAST_MOVING, ClassificazioneABC.A, ClassificazioneXYZ.X): {
                "strategia": "Just-In-Time con safety stock minimo",
                "service_level": 0.99,
                "review_period": "Continuo",
                "ordering_policy": "EOQ ottimizzato",
                "safety_stock_factor": 0.8,
            },
            (CategoriaMovimentazione.FAST_MOVING, ClassificazioneABC.A, ClassificazioneXYZ.Y): {
                "strategia": "Buffer moderato con riordino frequente",
                "service_level": 0.98,
                "review_period": "Settimanale",
                "ordering_policy": "EOQ con aggiustamenti",
                "safety_stock_factor": 1.0,
            },
            (CategoriaMovimentazione.FAST_MOVING, ClassificazioneABC.A, ClassificazioneXYZ.Z): {
                "strategia": "Safety stock elevato, forecast avanzato",
                "service_level": 0.95,
                "review_period": "Bisettimanale",
                "ordering_policy": "Order-up-to level",
                "safety_stock_factor": 1.5,
            },
            # Slow Moving + Classe A (raro ma critico)
            (CategoriaMovimentazione.SLOW_MOVING, ClassificazioneABC.A, ClassificazioneXYZ.X): {
                "strategia": "Stock minimo garantito",
                "service_level": 0.95,
                "review_period": "Mensile",
                "ordering_policy": "Min-Max",
                "safety_stock_factor": 1.2,
            },
            # Slow Moving + Classe C
            (CategoriaMovimentazione.SLOW_MOVING, ClassificazioneABC.C, ClassificazioneXYZ.Z): {
                "strategia": "Make-to-order quando possibile",
                "service_level": 0.85,
                "review_period": "Trimestrale",
                "ordering_policy": "Order on demand",
                "safety_stock_factor": 0.5,
            },
            # Very Slow Moving
            (CategoriaMovimentazione.VERY_SLOW, ClassificazioneABC.C, ClassificazioneXYZ.Z): {
                "strategia": "No stock - solo su ordine",
                "service_level": 0.80,
                "review_period": "Su richiesta",
                "ordering_policy": "Make-to-order",
                "safety_stock_factor": 0,
            },
        }

        # Strategia di default se combinazione non trovata
        default_strategy = {
            "strategia": "Bilanciamento standard",
            "service_level": 0.90,
            "review_period": "Mensile",
            "ordering_policy": "EOQ standard",
            "safety_stock_factor": 1.0,
        }

        return strategies.get((movimento, abc, xyz), default_strategy)

    @staticmethod
    def analyze_product_portfolio(
        products_data: pd.DataFrame, sales_history: pd.DataFrame, value_column: str = "fatturato"
    ) -> pd.DataFrame:
        """
        Analisi completa portfolio prodotti con classificazioni multiple

        Args:
            products_data: Dati anagrafici prodotti
            sales_history: Storico vendite
            value_column: Colonna per analisi ABC
        """
        results = []

        for product_id in products_data["product_id"].unique():
            # Filtra dati prodotto
            product_info = products_data[products_data["product_id"] == product_id].iloc[0]
            product_sales = sales_history[sales_history["product_id"] == product_id]["quantity"]

            # Calcola metriche
            annual_demand = product_sales.sum()
            avg_inventory = product_info.get("avg_inventory", annual_demand / 12)
            turnover = annual_demand / avg_inventory if avg_inventory > 0 else 0

            # Classificazioni
            movimento = MovementClassifier.classify_by_movement(turnover)
            xyz = MovementClassifier.classify_xyz(product_sales)

            # Coefficiente variazione
            cv = product_sales.std() / product_sales.mean() if product_sales.mean() > 0 else 999

            results.append(
                {
                    "product_id": product_id,
                    "product_name": product_info.get("name", ""),
                    "movimento": movimento.value[0],
                    "movimento_desc": movimento.value[1],
                    "classe_xyz": xyz.value[0],
                    "xyz_desc": xyz.value[1],
                    "turnover": turnover,
                    "cv": cv,
                    "annual_demand": annual_demand,
                    "avg_inventory": avg_inventory,
                }
            )

        results_df = pd.DataFrame(results)

        # Aggiungi classificazione ABC
        if value_column in products_data.columns:
            results_df = results_df.merge(
                products_data[["product_id", value_column]], on="product_id"
            )
            results_df = MovementClassifier.classify_abc(results_df, value_column)

        return results_df


# =====================================================
# OTTIMIZZATORE SPECIFICO SLOW/FAST MOVING
# =====================================================


class SlowFastOptimizer:
    """Ottimizzatore specializzato per slow e fast moving"""

    def __init__(self, costi: CostiGiacenza):
        self.costi = costi
        self.classifier = MovementClassifier()

    def optimize_slow_moving(
        self,
        demand_history: np.ndarray,
        unit_cost: float,
        lead_time: int,
        shelf_life_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ottimizzazione specifica per slow moving

        Caratteristiche:
        - Lotti minimi per ridurre capitale immobilizzato
        - Safety stock ridotto
        - Possibile make-to-order
        - Considerazione shelf life
        """
        demand_mean = np.mean(demand_history)
        demand_std = np.std(demand_history)

        # Per slow moving, usa percentili invece di distribuzione normale
        safety_stock = np.percentile(demand_history, 75) * lead_time

        # EOQ modificato per slow moving (lotti più piccoli)
        annual_demand = demand_mean * 365
        eoq_standard = np.sqrt(
            (2 * annual_demand * self.costi.costo_ordine_urgente)
            / (self.costi.tasso_capitale * unit_cost)
        )

        # Riduci EOQ per slow moving
        eoq_adjusted = max(1, eoq_standard * 0.5)

        # Se c'è shelf life, limita ulteriormente
        if shelf_life_days:
            max_stock_days = shelf_life_days * 0.5  # Max 50% shelf life
            max_stock = demand_mean * max_stock_days
            eoq_adjusted = min(eoq_adjusted, max_stock)

        # Calcola costi
        holding_cost = safety_stock * unit_cost * self.costi.tasso_capitale
        obsolescence_risk = min(1.0, (demand_std / demand_mean) * 2) if demand_mean > 0 else 1.0

        return {
            "safety_stock": round(safety_stock),
            "eoq": round(eoq_adjusted),
            "reorder_point": round(demand_mean * lead_time + safety_stock),
            "annual_orders": annual_demand / eoq_adjusted,
            "holding_cost": holding_cost,
            "obsolescence_risk": obsolescence_risk,
            "suggested_policy": "Min-Max con revisione mensile",
            "make_to_order_threshold": demand_mean * 30,  # Se stock > 30gg, considera MTO
        }

    def optimize_fast_moving(
        self,
        demand_history: np.ndarray,
        unit_cost: float,
        lead_time: int,
        supplier_constraints: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Ottimizzazione specifica per fast moving

        Caratteristiche:
        - Focus su disponibilità continua
        - Safety stock robusto
        - Ottimizzazione costi trasporto con lotti
        - Considerazione vincoli fornitore
        """
        demand_mean = np.mean(demand_history)
        demand_std = np.std(demand_history)

        # Safety stock più generoso per fast moving
        z_score = stats.norm.ppf(0.98)  # 98% service level
        safety_stock = z_score * demand_std * np.sqrt(lead_time)

        # EOQ con considerazione trasporto
        annual_demand = demand_mean * 365
        ordering_cost = self.costi.costo_ordine_urgente

        # Se ci sono vincoli fornitore
        if supplier_constraints:
            min_order = supplier_constraints.get("min_order_qty", 1)
            truck_capacity = supplier_constraints.get("truck_capacity", float("inf"))
        else:
            min_order = 1
            truck_capacity = float("inf")

        # EOQ standard
        eoq_standard = np.sqrt(
            (2 * annual_demand * ordering_cost) / (self.costi.tasso_capitale * unit_cost)
        )

        # Arrotonda a multipli del minimo ordine o capacità camion
        eoq_adjusted = max(min_order, eoq_standard)
        if truck_capacity < float("inf"):
            eoq_adjusted = round(eoq_adjusted / truck_capacity) * truck_capacity

        # Calcola metriche
        cycle_time = eoq_adjusted / demand_mean
        annual_orders = annual_demand / eoq_adjusted

        return {
            "safety_stock": round(safety_stock),
            "eoq": round(eoq_adjusted),
            "reorder_point": round(demand_mean * lead_time + safety_stock),
            "cycle_time_days": round(cycle_time),
            "annual_orders": round(annual_orders, 1),
            "service_level": 0.98,
            "suggested_policy": "Continuous review (s,Q)",
            "express_order_threshold": safety_stock * 0.5,  # Ordine express se sotto 50% SS
        }

    def compare_strategies(
        self,
        demand_history: np.ndarray,
        unit_cost: float,
        lead_time: int,
        current_policy: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Confronta strategia attuale con ottimizzata per slow/fast
        """
        demand_mean = np.mean(demand_history)
        turnover = (demand_mean * 365) / current_policy.get("avg_inventory", demand_mean * 30)

        # Classifica il prodotto
        categoria = self.classifier.classify_by_movement(turnover)

        # Ottimizza in base alla categoria
        if categoria in [CategoriaMovimentazione.SLOW_MOVING, CategoriaMovimentazione.VERY_SLOW]:
            optimized = self.optimize_slow_moving(demand_history, unit_cost, lead_time)
            strategy_type = "Slow Moving"
        else:
            optimized = self.optimize_fast_moving(demand_history, unit_cost, lead_time)
            strategy_type = "Fast Moving"

        # Calcola risparmi
        current_holding = (
            current_policy.get("avg_inventory", 0) * unit_cost * self.costi.tasso_capitale
        )
        optimized_holding = optimized["safety_stock"] * unit_cost * self.costi.tasso_capitale

        savings = current_holding - optimized_holding
        savings_pct = (savings / current_holding * 100) if current_holding > 0 else 0

        comparison = pd.DataFrame(
            {
                "Metrica": [
                    "Categoria",
                    "Safety Stock",
                    "EOQ",
                    "Reorder Point",
                    "Costo Giacenza Annuo",
                    "Risparmio Potenziale",
                    "Risparmio %",
                ],
                "Politica Attuale": [
                    "Non classificato",
                    current_policy.get("safety_stock", "N/A"),
                    current_policy.get("eoq", "N/A"),
                    current_policy.get("reorder_point", "N/A"),
                    f"€{current_holding:.0f}",
                    "-",
                    "-",
                ],
                "Politica Ottimizzata": [
                    strategy_type,
                    optimized["safety_stock"],
                    optimized["eoq"],
                    optimized["reorder_point"],
                    f"€{optimized_holding:.0f}",
                    f"€{savings:.0f}",
                    f"{savings_pct:.1f}%",
                ],
            }
        )

        return comparison


# =====================================================
# ANALIZZATORE COSTI TOTALI
# =====================================================


class TotalCostAnalyzer:
    """Analizza e ottimizza i costi totali di giacenza"""

    def __init__(self, costi: CostiGiacenza):
        self.costi = costi

    def calculate_holding_cost(
        self, average_inventory: float, unit_cost: float, space_per_unit: float
    ) -> Dict[str, float]:
        """Calcola costi di mantenimento scorte"""
        # Costo capitale immobilizzato
        capital_cost = average_inventory * unit_cost * self.costi.tasso_capitale

        # Costo stoccaggio fisico
        storage_cost = average_inventory * space_per_unit * self.costi.costo_stoccaggio_mq_mese * 12

        # Costo obsolescenza
        obsolescence_cost = average_inventory * unit_cost * self.costi.tasso_obsolescenza_annuo

        total_holding = capital_cost + storage_cost + obsolescence_cost

        return {
            "capital_cost": capital_cost,
            "storage_cost": storage_cost,
            "obsolescence_cost": obsolescence_cost,
            "total_holding_cost": total_holding,
            "holding_cost_per_unit": total_holding / average_inventory
            if average_inventory > 0
            else 0,
        }

    def calculate_stockout_cost(
        self, stockout_probability: float, annual_demand: float, gross_margin: float
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
            "expected_lost_sales": expected_lost_sales,
            "lost_margin": lost_margin,
            "emergency_order_cost": emergency_orders,
            "customer_loss_risk": customer_loss_risk,
            "total_stockout_cost": total_stockout,
        }

    def find_optimal_inventory_level(
        self,
        demand_forecast: np.ndarray,
        unit_cost: float,
        gross_margin: float,
        space_per_unit: float,
        service_levels: List[float] = None,
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

            total_cost = holding["total_holding_cost"] + stockout["total_stockout_cost"]

            results.append(
                {
                    "service_level": sl,
                    "safety_stock": safety_stock,
                    "avg_inventory": avg_inventory,
                    "holding_cost": holding["total_holding_cost"],
                    "stockout_cost": stockout["total_stockout_cost"],
                    "total_cost": total_cost,
                }
            )

        # Trova ottimo
        results_df = pd.DataFrame(results)
        optimal_idx = results_df["total_cost"].idxmin()
        optimal = results_df.iloc[optimal_idx]

        return {
            "optimal_service_level": optimal["service_level"],
            "optimal_safety_stock": optimal["safety_stock"],
            "optimal_avg_inventory": optimal["avg_inventory"],
            "optimal_total_cost": optimal["total_cost"],
            "cost_breakdown": {
                "holding": optimal["holding_cost"],
                "stockout": optimal["stockout_cost"],
            },
            "all_results": results_df,
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
        lead_time_days: int,
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
            livello_alert=alert,
        )

    @staticmethod
    def generate_action_recommendations(
        analisi: AnalisiRischio, current_stock: float, reorder_point: float, eoq: float
    ) -> List[str]:
        """Genera raccomandazioni azioni basate su analisi"""
        recommendations = []

        if analisi.livello_alert == AlertLevel.CRITICO:
            recommendations.append(f"[URGENTE] Effettuare ordine immediato di {eoq} unità")
            recommendations.append("[URGENTE] Contattare fornitori per spedizione express")
            recommendations.append(
                "[URGENTE] Preparare comunicazione clienti per possibili ritardi"
            )

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
            recommendations.append(
                f"[EFFICIENZA] Turnover basso ({analisi.inventory_turnover}x), ridurre lotti ordine"
            )

        if analisi.rischio_obsolescenza > 0.5:
            recommendations.append(
                f"[RISCHIO] Alto rischio obsolescenza ({analisi.rischio_obsolescenza:.0%})"
            )

        if analisi.cash_cycle_days > 60:
            recommendations.append(
                f"[FINANZA] Capitale immobilizzato per {analisi.cash_cycle_days} giorni"
            )

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
        confidence_levels: List[float] = None,
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
            forecast_values = base_forecast.get("forecast", base_forecast)
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
            intervals[f"ci_{int(cl * 100)}"] = {
                "lower": forecast_values - z_score * adaptive_std,
                "upper": forecast_values + z_score * adaptive_std,
            }

        # Calcola metriche incertezza
        uncertainty_metrics = {
            "avg_interval_width": np.mean(
                intervals["ci_95"]["upper"] - intervals["ci_95"]["lower"]
            ),
            "max_uncertainty": np.max(adaptive_std),
            "uncertainty_growth_rate": (adaptive_std[-1] - adaptive_std[0]) / steps,
        }

        return {
            "forecast": forecast_values,
            "intervals": intervals,
            "adaptive_std": adaptive_std,
            "uncertainty_metrics": uncertainty_metrics,
        }


# =====================================================
# DASHBOARD KPI BILANCIAMENTO
# =====================================================


class InventoryKPIDashboard:
    """Dashboard KPI per monitoraggio bilanciamento scorte"""

    @staticmethod
    def calculate_kpis(
        sales_data: pd.DataFrame, inventory_data: pd.DataFrame, costs_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calcola tutti i KPI chiave per dashboard
        """
        # Fill Rate (% ordini completamente evasi)
        if "orders_total" in sales_data.columns and "orders_fulfilled" in sales_data.columns:
            fill_rate = sales_data["orders_fulfilled"].sum() / sales_data["orders_total"].sum()
        else:
            fill_rate = 0.95  # Default stimato

        # Inventory Turnover
        cogs = sales_data["quantity"].sum() * costs_data.get("unit_cost", 100)
        avg_inventory = inventory_data["stock_level"].mean() * costs_data.get("unit_cost", 100)
        inventory_turnover = cogs / avg_inventory if avg_inventory > 0 else 0

        # Days of Supply (DOS)
        daily_demand = sales_data["quantity"].mean()
        current_inventory = inventory_data["stock_level"].iloc[-1]
        days_of_supply = current_inventory / daily_demand if daily_demand > 0 else 0

        # GMROI (Gross Margin Return on Investment)
        gross_margin = costs_data.get("gross_margin", 0.3)
        gmroi = (gross_margin * cogs) / avg_inventory if avg_inventory > 0 else 0

        # Cash-to-Cash Cycle
        days_inventory_outstanding = 365 / inventory_turnover if inventory_turnover > 0 else 365
        days_sales_outstanding = costs_data.get("payment_terms", 30)
        days_payable_outstanding = costs_data.get("supplier_terms", 45)
        cash_cycle = days_inventory_outstanding + days_sales_outstanding - days_payable_outstanding

        # Perfect Order Rate
        if "on_time" in sales_data.columns and "complete" in sales_data.columns:
            perfect_orders = ((sales_data["on_time"] == 1) & (sales_data["complete"] == 1)).sum()
            perfect_order_rate = perfect_orders / len(sales_data)
        else:
            perfect_order_rate = 0.92  # Default stimato

        # Forecast Accuracy (se disponibile)
        if "actual" in sales_data.columns and "forecast" in sales_data.columns:
            mape = (
                np.mean(
                    np.abs((sales_data["actual"] - sales_data["forecast"]) / sales_data["actual"])
                )
                * 100
            )
            forecast_accuracy = 100 - mape
        else:
            forecast_accuracy = 85  # Default stimato

        return {
            "fill_rate": round(fill_rate * 100, 1),
            "inventory_turnover": round(inventory_turnover, 2),
            "days_of_supply": round(days_of_supply, 1),
            "gmroi": round(gmroi, 2),
            "cash_cycle_days": round(cash_cycle, 1),
            "perfect_order_rate": round(perfect_order_rate * 100, 1),
            "forecast_accuracy": round(forecast_accuracy, 1),
            "health_score": InventoryKPIDashboard._calculate_health_score(
                fill_rate, inventory_turnover, days_of_supply, gmroi
            ),
        }

    @staticmethod
    def _calculate_health_score(fill_rate: float, turnover: float, dos: float, gmroi: float) -> str:
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

        if kpis["fill_rate"] < 95:
            suggestions.append(
                f"Fill rate basso ({kpis['fill_rate']}%): aumentare safety stock prodotti critici"
            )

        if kpis["inventory_turnover"] < 6:
            suggestions.append(
                f"Turnover lento ({kpis['inventory_turnover']}x): ridurre lotti ordine, aumentare frequenza"
            )

        if kpis["days_of_supply"] > 60:
            suggestions.append(
                f"DOS elevato ({kpis['days_of_supply']}gg): rischio obsolescenza, valutare promozioni"
            )
        elif kpis["days_of_supply"] < 15:
            suggestions.append(
                f"DOS basso ({kpis['days_of_supply']}gg): rischio stockout, aumentare buffer"
            )

        if kpis["gmroi"] < 2:
            suggestions.append(
                f"GMROI basso ({kpis['gmroi']}): ottimizzare mix prodotti o ridurre scorte"
            )

        if kpis["cash_cycle_days"] > 60:
            suggestions.append(
                f"Ciclo cassa lungo ({kpis['cash_cycle_days']}gg): negoziare termini pagamento"
            )

        if kpis["forecast_accuracy"] < 80:
            suggestions.append(
                f"Accuracy bassa ({kpis['forecast_accuracy']}%): rivedere modelli forecast"
            )

        if not suggestions:
            suggestions.append("Performance ottimali su tutti i KPI principali!")

        return suggestions


# =====================================================
# 1. PERISHABLE INVENTORY & FEFO MANAGEMENT
# =====================================================


class TipoScadenza(Enum):
    """Tipi di scadenza per prodotti deperibili"""

    FIXED_SHELF_LIFE = ("fixed", "Scadenza fissa dalla produzione")
    DYNAMIC_AGING = ("dynamic", "Deterioramento progressivo")
    REGULATORY = ("regulatory", "Scadenza normativa obbligatoria")
    QUALITY_BASED = ("quality", "Basata su parametri qualità")


class LottoPerishable(BaseModel):
    """Informazioni lotto prodotto deperibile"""

    lotto_id: str
    quantita: int
    data_produzione: datetime
    data_scadenza: datetime
    shelf_life_giorni: int
    giorni_residui: int
    percentuale_vita_residua: float
    valore_unitario: float
    rischio_obsolescenza: float


class PerishableManager:
    """Gestore inventory per prodotti deperibili con logica FEFO"""

    def __init__(self, tipo_scadenza: TipoScadenza = TipoScadenza.FIXED_SHELF_LIFE):
        self.tipo_scadenza = tipo_scadenza
        self.lotti_tracciati: Dict[str, List[LottoPerishable]] = {}
        # Integrazione MSL
        self.msl_manager: Optional[MinimumShelfLifeManager] = None

    def analizza_lotti(self, lotti: List[Dict]) -> List[LottoPerishable]:
        """Analizza lotti esistenti e calcola metriche scadenza"""
        lotti_analizzati = []

        for lotto_data in lotti:
            data_prod = pd.to_datetime(lotto_data["data_produzione"])
            data_scad = pd.to_datetime(lotto_data["data_scadenza"])
            oggi = pd.Timestamp.now()

            shelf_life_totale = (data_scad - data_prod).days
            giorni_residui = max(0, (data_scad - oggi).days)
            perc_vita_residua = giorni_residui / shelf_life_totale if shelf_life_totale > 0 else 0

            # Calcola rischio obsolescenza (aumenta esponenzialmente vicino a scadenza)
            if giorni_residui <= 0:
                rischio = 1.0  # Scaduto
            elif perc_vita_residua < 0.1:
                rischio = 0.9  # <10% vita residua
            elif perc_vita_residua < 0.3:
                rischio = 0.6  # <30% vita residua
            else:
                rischio = min(0.3, (1 - perc_vita_residua))

            lotto = LottoPerishable(
                lotto_id=lotto_data["lotto_id"],
                quantita=lotto_data["quantita"],
                data_produzione=data_prod,
                data_scadenza=data_scad,
                shelf_life_giorni=shelf_life_totale,
                giorni_residui=giorni_residui,
                percentuale_vita_residua=perc_vita_residua,
                valore_unitario=lotto_data["valore_unitario"],
                rischio_obsolescenza=rischio,
            )
            lotti_analizzati.append(lotto)

        return sorted(lotti_analizzati, key=lambda x: x.data_scadenza)  # FEFO sorting

    def calcola_markdown_ottimale(
        self, lotto: LottoPerishable, domanda_giornaliera: float, elasticita_prezzo: float = 1.5
    ) -> Dict[str, float]:
        """
        Calcola markdown ottimale per accelerare vendita prima scadenza

        Args:
            lotto: Lotto da analizzare
            domanda_giornaliera: Domanda media corrente
            elasticita_prezzo: Elasticità domanda/prezzo (>1 = elastica)
        """
        giorni_vendita_normale = (
            lotto.quantita / domanda_giornaliera if domanda_giornaliera > 0 else 999
        )

        if giorni_vendita_normale <= lotto.giorni_residui:
            # Vendita normale possibile
            return {
                "markdown_suggerito": 0,
                "prezzo_finale": lotto.valore_unitario,
                "giorni_smaltimento": giorni_vendita_normale,
                "azione": "Vendita normale",
            }

        # Calcola markdown necessario per accelerare vendita
        accelerazione_necessaria = giorni_vendita_normale / max(1, lotto.giorni_residui - 1)
        markdown_percentuale = min(0.8, (accelerazione_necessaria - 1) / elasticita_prezzo)

        prezzo_scontato = lotto.valore_unitario * (1 - markdown_percentuale)
        domanda_accelerata = domanda_giornaliera * (1 + markdown_percentuale * elasticita_prezzo)
        giorni_smaltimento = lotto.quantita / domanda_accelerata if domanda_accelerata > 0 else 999

        if giorni_smaltimento > lotto.giorni_residui:
            # Anche con markdown massimo non si smaltisce
            return {
                "markdown_suggerito": 0.8,
                "prezzo_finale": lotto.valore_unitario * 0.2,
                "giorni_smaltimento": giorni_smaltimento,
                "azione": "Liquidazione urgente",
            }
        else:
            return {
                "markdown_suggerito": markdown_percentuale,
                "prezzo_finale": prezzo_scontato,
                "giorni_smaltimento": giorni_smaltimento,
                "azione": "Markdown accelerato",
            }

    def strategia_riordino_perishable(
        self,
        lotti_esistenti: List[LottoPerishable],
        forecast_domanda: np.ndarray,
        shelf_life_nuovo_lotto: int,
        costo_obsolescenza_unitario: float,
    ) -> Dict[str, Any]:
        """Strategia riordino considerando rischio obsolescenza"""

        # Calcola copertura lotti esistenti
        stock_totale = sum(lotto.quantita for lotto in lotti_esistenti)
        stock_a_rischio = sum(
            lotto.quantita for lotto in lotti_esistenti if lotto.rischio_obsolescenza > 0.5
        )

        domanda_media = np.mean(forecast_domanda)
        giorni_copertura = stock_totale / domanda_media if domanda_media > 0 else 999

        # Calcola shelf life medio ponderato stock esistente
        if stock_totale > 0:
            shelf_life_medio = (
                sum(lotto.giorni_residui * lotto.quantita for lotto in lotti_esistenti)
                / stock_totale
            )
        else:
            shelf_life_medio = 0

        # Decisione riordino
        if giorni_copertura < 7:  # Meno di 1 settimana
            # Ordina quantità per 2 settimane max (per evitare obsolescenza)
            giorni_target = min(14, shelf_life_nuovo_lotto * 0.7)
            quantita_riordino = domanda_media * giorni_target - stock_totale
            azione = "Riordino normale"

        elif stock_a_rischio > stock_totale * 0.3:  # >30% stock a rischio
            # Sospendi ordini, smaltisci prima stock a rischio
            quantita_riordino = 0
            azione = "Sospendi ordini - Smaltimento prioritario"

        elif giorni_copertura > shelf_life_medio:
            # Troppo stock vs shelf life residua
            quantita_riordino = 0
            azione = "Overstock vs shelf life"

        else:
            # Situazione normale
            quantita_riordino = max(0, domanda_media * 7 - stock_totale)  # Target 1 settimana
            azione = "Riordino conservativo"

        # Calcola costo obsolescenza atteso
        costo_obsolescenza = sum(
            lotto.quantita * lotto.rischio_obsolescenza * costo_obsolescenza_unitario
            for lotto in lotti_esistenti
        )

        return {
            "quantita_riordino": max(0, quantita_riordino),
            "giorni_copertura_attuali": giorni_copertura,
            "stock_a_rischio": stock_a_rischio,
            "percentuale_a_rischio": stock_a_rischio / stock_totale * 100
            if stock_totale > 0
            else 0,
            "costo_obsolescenza_atteso": costo_obsolescenza,
            "shelf_life_medio_residuo": shelf_life_medio,
            "azione_consigliata": azione,
            "urgenza_markdown": len([l for l in lotti_esistenti if l.giorni_residui <= 7]),
        }

    def abilita_msl_integration(self, msl_manager: "MinimumShelfLifeManager"):
        """Abilita integrazione con Minimum Shelf Life Manager"""
        self.msl_manager = msl_manager

    def ottimizza_fefo_con_msl(
        self,
        lotti_esistenti: List[LottoPerishable],
        domanda_canali: Dict[str, int],
        prezzo_canali: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Ottimizzazione FEFO integrata con MSL per allocazione multi-canale
        Combina logica FEFO tradizionale con requisiti MSL per massimizzare valore
        """

        if not self.msl_manager:
            # Fallback a FEFO standard se MSL non abilitato
            return self._fefo_standard(lotti_esistenti, sum(domanda_canali.values()))

        # Usa MSL Manager per allocazione ottimale
        risultati_msl = self.msl_manager.ottimizza_allocazione_lotti(
            lotti_disponibili=lotti_esistenti.copy(),
            domanda_canali=domanda_canali,
            prezzo_canali=prezzo_canali,
        )

        # Genera report unificato FEFO + MSL
        return self._genera_report_fefo_msl(risultati_msl, lotti_esistenti)

    def _fefo_standard(self, lotti: List[LottoPerishable], domanda_totale: int) -> Dict[str, Any]:
        """Logica FEFO standard senza MSL"""
        lotti_ordinati = sorted(lotti, key=lambda l: l.data_scadenza)
        allocazioni = []
        quantita_allocata = 0

        for lotto in lotti_ordinati:
            if quantita_allocata >= domanda_totale:
                break

            quantita_da_lotto = min(lotto.quantita_disponibile, domanda_totale - quantita_allocata)
            if quantita_da_lotto > 0:
                allocazioni.append(
                    {
                        "lotto_id": lotto.lotto_id,
                        "quantita": quantita_da_lotto,
                        "giorni_residui": lotto.giorni_residui,
                        "canale": "standard_fefo",
                    }
                )
                quantita_allocata += quantita_da_lotto

        return {
            "tipo_ottimizzazione": "FEFO_STANDARD",
            "allocazioni": allocazioni,
            "quantita_totale": quantita_allocata,
            "utilizzo_lotti": len(allocazioni),
        }

    def _genera_report_fefo_msl(
        self, risultati_msl: Dict[str, List], lotti_originali: List[LottoPerishable]
    ) -> Dict[str, Any]:
        """Genera report unificato FEFO + MSL"""

        # Calcola metriche aggregate
        totale_quantita = sum(
            sum(a.quantita_allocata for a in allocazioni) for allocazioni in risultati_msl.values()
        )

        totale_valore = sum(
            sum(a.valore_allocato for a in allocazioni) for allocazioni in risultati_msl.values()
        )

        lotti_utilizzati = set()
        for allocazioni in risultati_msl.values():
            for alloc in allocazioni:
                lotti_utilizzati.add(alloc.lotto_id)

        # Analisi efficienza FEFO vs MSL
        efficienza_fefo = self._calcola_efficienza_fefo(risultati_msl, lotti_originali)

        return {
            "tipo_ottimizzazione": "FEFO_MSL_INTEGRATO",
            "summary": {
                "quantita_totale_allocata": totale_quantita,
                "valore_totale_allocato": round(totale_valore, 2),
                "numero_lotti_utilizzati": len(lotti_utilizzati),
                "numero_canali_serviti": len(risultati_msl),
                "prezzo_medio_realizzo": round(totale_valore / totale_quantita, 2)
                if totale_quantita > 0
                else 0,
            },
            "allocazioni_per_canale": risultati_msl,
            "metriche_fefo": efficienza_fefo,
            "vantaggi_msl": {
                "differenza_valore_vs_fefo_std": round(
                    totale_valore - (totale_quantita * 3.0),
                    2,  # Assume prezzo base €3
                ),
                "canali_premium_serviti": len(
                    [c for c in risultati_msl.keys() if "premium" in c or "b2b" in c]
                ),
                "riduzione_rischio_obsolescenza": self._calcola_riduzione_rischio_obsolescenza(
                    risultati_msl
                ),
            },
        }

    def _calcola_efficienza_fefo(self, risultati_msl, lotti_originali) -> Dict[str, Any]:
        """Calcola metriche efficienza FEFO"""

        # Ordina lotti per scadenza (FEFO)
        lotti_fefo = sorted(lotti_originali, key=lambda l: l.data_scadenza)

        # Verifica aderenza FEFO
        allocazioni_flatten = []
        for allocazioni_canale in risultati_msl.values():
            allocazioni_flatten.extend(allocazioni_canale)

        allocazioni_ordinate = sorted(
            allocazioni_flatten, key=lambda a: a.giorni_shelf_life_residui
        )

        violations_fefo = 0
        for i in range(1, len(allocazioni_ordinate)):
            if (
                allocazioni_ordinate[i].giorni_shelf_life_residui
                < allocazioni_ordinate[i - 1].giorni_shelf_life_residui
            ):
                violations_fefo += 1

        return {
            "aderenza_fefo_percentuale": round(
                (1 - violations_fefo / max(1, len(allocazioni_ordinate))) * 100, 1
            ),
            "violazioni_fefo": violations_fefo,
            "shelf_life_medio_allocato": round(
                np.mean([a.giorni_shelf_life_residui for a in allocazioni_ordinate]), 1
            )
            if allocazioni_ordinate
            else 0,
        }

    def _calcola_riduzione_rischio_obsolescenza(self, risultati_msl) -> float:
        """Calcola riduzione rischio obsolescenza grazie a MSL"""

        allocazioni_critiche = []
        for allocazioni_canale in risultati_msl.values():
            allocazioni_critiche.extend(
                [a for a in allocazioni_canale if a.urgenza in ["urgente", "critico"]]
            )

        if not allocazioni_critiche:
            return 0.0

        # Stima riduzione rischio basata su margini MSL
        margini_msl = [a.margine_msl for a in allocazioni_critiche]
        riduzione_media = sum(max(0, 30 - abs(m)) / 30 * 0.3 for m in margini_msl) / len(
            margini_msl
        )

        return round(riduzione_media * 100, 1)


# =====================================================
# 1.2. MINIMUM SHELF LIFE (MSL) MANAGEMENT
# =====================================================


class TipoCanale(Enum):
    """Tipi di canale di vendita con requisiti MSL"""

    GDO_PREMIUM = ("gdo_premium", "GDO Premium - MSL 90 giorni", 90)
    GDO_STANDARD = ("gdo_standard", "GDO Standard - MSL 60 giorni", 60)
    RETAIL_TRADIZIONALE = ("retail", "Retail Tradizionale - MSL 45 giorni", 45)
    ONLINE_DIRETTO = ("online", "E-commerce Diretto - MSL 30 giorni", 30)
    OUTLET_SCONTI = ("outlet", "Outlet/Sconti - MSL 15 giorni", 15)
    B2B_WHOLESALE = ("b2b", "B2B Wholesale - MSL 120 giorni", 120)


class RequisitoMSL(BaseModel):
    """Requisito MSL specifico per canale-prodotto"""

    canale: TipoCanale
    prodotto_codice: str
    msl_giorni: int
    priorita: int = Field(1, description="1=alta, 5=bassa priorità")
    attivo: bool = True
    note: Optional[str] = None


class AllocationResult(BaseModel):
    """Risultato allocazione lotto a canale"""

    lotto_id: str
    canale: TipoCanale
    quantita_allocata: int
    giorni_shelf_life_residui: int
    valore_allocato: float
    margine_msl: int  # giorni di margine oltre MSL minimo
    urgenza: str  # "normale", "attenzione", "urgente", "critico"


class MinimumShelfLifeManager:
    """
    Gestore Minimum Shelf Life (MSL) per allocazione ottimale inventory
    ai diversi canali di vendita in base alla vita residua prodotti
    """

    def __init__(self):
        self.requisiti_msl: Dict[str, Dict[str, RequisitoMSL]] = {}
        self.storico_allocazioni: List[AllocationResult] = []

    def aggiungi_requisito_msl(self, requisito: RequisitoMSL):
        """Aggiunge o aggiorna requisito MSL per canale-prodotto"""
        if requisito.prodotto_codice not in self.requisiti_msl:
            self.requisiti_msl[requisito.prodotto_codice] = {}

        self.requisiti_msl[requisito.prodotto_codice][requisito.canale.value[0]] = requisito

    def get_canali_compatibili(
        self, prodotto_codice: str, giorni_shelf_life_residui: int
    ) -> List[TipoCanale]:
        """
        Restituisce lista canali compatibili per prodotto con shelf life residua
        Ordinati per priorità (canali con MSL più alto = maggior valore)
        """
        if prodotto_codice not in self.requisiti_msl:
            # Se non ci sono requisiti specifici, usa requisiti standard
            canali_compatibili = []
            for canale in TipoCanale:
                if giorni_shelf_life_residui >= canale.value[2]:
                    canali_compatibili.append(canale)
        else:
            # Usa requisiti specifici per prodotto
            canali_compatibili = []
            for canale_id, requisito in self.requisiti_msl[prodotto_codice].items():
                if requisito.attivo and giorni_shelf_life_residui >= requisito.msl_giorni:
                    canali_compatibili.append(requisito.canale)

        # Ordina per MSL decrescente (canali più esigenti = maggior valore)
        return sorted(canali_compatibili, key=lambda c: c.value[2], reverse=True)

    def calcola_urgenza_allocazione(self, giorni_shelf_life_residui: int, msl_minimo: int) -> str:
        """Calcola urgenza allocazione basata su margine MSL"""
        margine = giorni_shelf_life_residui - msl_minimo

        if margine >= 30:
            return "normale"
        elif margine >= 15:
            return "attenzione"
        elif margine >= 7:
            return "urgente"
        else:
            return "critico"

    def ottimizza_allocazione_lotti(
        self,
        lotti_disponibili: List[LottoPerishable],
        domanda_canali: Dict[str, int],  # canale_id -> quantità richiesta
        prezzo_canali: Dict[str, float],  # canale_id -> prezzo unitario
    ) -> Dict[str, List[AllocationResult]]:
        """
        Ottimizza allocazione lotti ai canali massimizzando valore e rispettando MSL

        Algoritmo:
        1. Ordina lotti per FEFO (First Expired, First Out)
        2. Per ogni lotto, trova canale compatibile con maggior valore
        3. Alloca quantità massima possibile rispettando domanda
        4. Traccia margini MSL e urgenze
        """
        risultati_per_canale = {}
        lotti_ordinati = sorted(lotti_disponibili, key=lambda l: l.data_scadenza)
        domanda_residua = domanda_canali.copy()

        for lotto in lotti_ordinati:
            if lotto.quantita <= 0:
                continue

            giorni_residui = (lotto.data_scadenza - datetime.now()).days
            canali_compatibili = self.get_canali_compatibili("YOG001", giorni_residui)  # Temp fix

            for canale in canali_compatibili:
                canale_id = canale.value[0]

                if canale_id not in domanda_residua or domanda_residua[canale_id] <= 0:
                    continue

                # Calcola quantità da allocare
                quantita_allocare = min(lotto.quantita, domanda_residua[canale_id])

                if quantita_allocare > 0:
                    # Crea risultato allocazione
                    prezzo_unitario = prezzo_canali.get(canale_id, 0.0)
                    msl_richiesto = canale.value[2]

                    risultato = AllocationResult(
                        lotto_id=lotto.lotto_id,
                        canale=canale,
                        quantita_allocata=quantita_allocare,
                        giorni_shelf_life_residui=giorni_residui,
                        valore_allocato=quantita_allocare * prezzo_unitario,
                        margine_msl=giorni_residui - msl_richiesto,
                        urgenza=self.calcola_urgenza_allocazione(giorni_residui, msl_richiesto),
                    )

                    # Aggiorna tracking
                    if canale_id not in risultati_per_canale:
                        risultati_per_canale[canale_id] = []
                    risultati_per_canale[canale_id].append(risultato)

                    # Aggiorna quantità residue
                    lotto.quantita -= quantita_allocare
                    domanda_residua[canale_id] -= quantita_allocare

                    break  # Passa al lotto successivo

        return risultati_per_canale

    def genera_report_allocazioni(
        self, risultati: Dict[str, List[AllocationResult]]
    ) -> Dict[str, Any]:
        """Genera report dettagliato allocazioni MSL"""

        totale_valore = 0
        totale_quantita = 0
        allocazioni_per_urgenza = {"normale": 0, "attenzione": 0, "urgente": 0, "critico": 0}

        # Metriche per canale
        metriche_canali = {}

        for canale_id, allocazioni in risultati.items():
            valore_canale = sum(a.valore_allocato for a in allocazioni)
            quantita_canale = sum(a.quantita_allocata for a in allocazioni)
            margine_medio = np.mean([a.margine_msl for a in allocazioni]) if allocazioni else 0

            metriche_canali[canale_id] = {
                "valore_totale": valore_canale,
                "quantita_totale": quantita_canale,
                "numero_lotti": len(allocazioni),
                "margine_msl_medio": round(margine_medio, 1),
                "valore_medio_unitario": valore_canale / quantita_canale
                if quantita_canale > 0
                else 0,
            }

            # Aggiorna totali
            totale_valore += valore_canale
            totale_quantita += quantita_canale

            # Conteggio urgenze
            for allocazione in allocazioni:
                if allocazione.urgenza in allocazioni_per_urgenza:
                    allocazioni_per_urgenza[allocazione.urgenza] += 1

        return {
            "data_report": datetime.now().isoformat(),
            "summary": {
                "valore_totale_allocato": round(totale_valore, 2),
                "quantita_totale_allocata": totale_quantita,
                "numero_canali_serviti": len(risultati),
                "valore_medio_unitario": round(totale_valore / totale_quantita, 2)
                if totale_quantita > 0
                else 0,
            },
            "distribuzione_urgenze": allocazioni_per_urgenza,
            "metriche_per_canale": metriche_canali,
            "canale_maggior_valore": max(
                metriche_canali.items(), key=lambda x: x[1]["valore_totale"]
            )[0]
            if metriche_canali
            else None,
            "efficienza_allocazione": round(
                totale_quantita / sum(len(alloc) for alloc in risultati.values()) * 100, 1
            )
            if risultati
            else 0,
        }

    def suggerisci_azioni_msl(
        self, risultati: Dict[str, List[AllocationResult]]
    ) -> List[Dict[str, str]]:
        """Suggerisce azioni basate sui risultati allocazione MSL"""

        azioni = []

        # Analizza allocazioni critiche
        allocazioni_critiche = []
        for canale_allocazioni in risultati.values():
            allocazioni_critiche.extend([a for a in canale_allocazioni if a.urgenza == "critico"])

        if allocazioni_critiche:
            azioni.append(
                {
                    "priorita": "ALTA",
                    "tipo": "MSL_CRITICO",
                    "descrizione": f"{len(allocazioni_critiche)} lotti con MSL critico (<7 giorni margine)",
                    "azione": "Implementare sconti urgenti o markdown per accelerare rotazione",
                }
            )

        # Analizza efficienza canali
        canali_vuoti = []
        for canale in TipoCanale:
            if canale.value[0] not in risultati:
                canali_vuoti.append(canale.value[1])

        if canali_vuoti:
            azioni.append(
                {
                    "priorita": "MEDIA",
                    "tipo": "CANALI_NON_SERVITI",
                    "descrizione": f"Canali non serviti: {', '.join(canali_vuoti[:3])}",
                    "azione": "Verificare disponibilità prodotti con shelf life adeguata",
                }
            )

        # Analizza concentrazione su singoli canali
        if risultati:
            canale_principale = max(
                risultati.items(), key=lambda x: sum(a.quantita_allocata for a in x[1])
            )
            concentrazione = (
                sum(a.quantita_allocata for a in canale_principale[1])
                / sum(
                    sum(a.quantita_allocata for a in allocazioni)
                    for allocazioni in risultati.values()
                )
                * 100
            )

            if concentrazione > 70:
                azioni.append(
                    {
                        "priorita": "MEDIA",
                        "tipo": "CONCENTRAZIONE_CANALE",
                        "descrizione": f"Alta concentrazione su {canale_principale[0]} ({concentrazione:.1f}%)",
                        "azione": "Diversificare allocazioni per ridurre rischio canale",
                    }
                )

        return azioni


# =====================================================
# 2. MULTI-ECHELON INVENTORY OPTIMIZATION
# =====================================================


class LivelloEchelon(Enum):
    """Livelli della supply chain"""

    CENTRALE = ("central", "Deposito centrale")
    REGIONALE = ("regional", "Hub regionale")
    LOCALE = ("local", "Punto vendita/Filiale")
    CLIENTE = ("customer", "Cliente finale")


class NodoInventory(BaseModel):
    """Nodo nella rete multi-echelon"""

    nodo_id: str
    nome: str
    livello: LivelloEchelon
    capacita_max: int
    stock_attuale: int
    demand_rate: float
    lead_time_fornitori: Dict[str, int]  # lead time dai fornitori/nodi superiori
    costi_trasporto: Dict[str, float]  # costi verso nodi inferiori
    nodi_figli: List[str]  # nodi che servire
    nodi_genitori: List[str]  # nodi che forniscono


class MultiEchelonOptimizer:
    """Ottimizzatore inventory per reti multi-echelon"""

    def __init__(self, rete_nodi: Dict[str, NodoInventory]):
        self.rete = rete_nodi
        self.matrice_costi = self._calcola_matrice_costi()

    def _calcola_matrice_costi(self) -> Dict[str, Dict[str, float]]:
        """Calcola matrice costi trasporto tra tutti i nodi"""
        matrice = {}

        for nodo_id, nodo in self.rete.items():
            matrice[nodo_id] = {}
            for figlio_id in nodo.nodi_figli:
                if figlio_id in nodo.costi_trasporto:
                    matrice[nodo_id][figlio_id] = nodo.costi_trasporto[figlio_id]

        return matrice

    def calcola_safety_stock_echelon(
        self,
        nodo_id: str,
        service_level_target: float,
        variabilita_domanda: float,
        variabilita_lead_time: float = 0.1,
    ) -> Dict[str, float]:
        """
        Calcola safety stock ottimale considerando pooling risk su rete

        Formula multi-echelon: SS = z * σ * √(LT + Review_Period) * √(1 - ρ)
        dove ρ è il coefficiente di correlazione tra nodi
        """
        nodo = self.rete[nodo_id]

        # Z-score per service level
        z_score = stats.norm.ppf(service_level_target)

        # Lead time medio ponderato
        if nodo.lead_time_fornitori:
            lead_time_medio = np.mean(list(nodo.lead_time_fornitori.values()))
        else:
            lead_time_medio = 14  # Default 2 settimane

        # Beneficio risk pooling (riduzione variabilità per aggregazione)
        if len(nodo.nodi_figli) > 1:
            pooling_factor = 1 / np.sqrt(len(nodo.nodi_figli))  # Central Limit Theorem
        else:
            pooling_factor = 1.0

        # Safety stock base
        safety_stock_base = (
            z_score * variabilita_domanda * np.sqrt(lead_time_medio) * pooling_factor
        )

        # Safety stock considerando livello echelon
        if nodo.livello == LivelloEchelon.CENTRALE:
            # Centrale: pooling benefit massimo, ma deve servire tutta la rete
            ss_multiplier = 0.8  # Ridotto per pooling
        elif nodo.livello == LivelloEchelon.REGIONALE:
            # Regionale: bilanciamento tra pooling e responsività
            ss_multiplier = 1.0
        else:  # LOCALE
            # Locale: serve direttamente clienti, safety stock più alto
            ss_multiplier = 1.2

        safety_stock_finale = safety_stock_base * ss_multiplier

        return {
            "safety_stock": round(safety_stock_finale),
            "pooling_factor": pooling_factor,
            "lead_time_medio": lead_time_medio,
            "z_score": z_score,
            "beneficio_pooling_pct": (1 - pooling_factor) * 100,
        }

    def ottimizza_allocation(
        self,
        stock_disponibile_centrale: int,
        richieste_nodi: Dict[str, int],
        priorita_nodi: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Ottimizza allocazione stock dal centrale verso nodi regionali/locali
        usando fair share con priorità
        """
        if priorita_nodi is None:
            priorita_nodi = {nodo_id: 1.0 for nodo_id in richieste_nodi.keys()}

        richiesta_totale = sum(richieste_nodi.values())

        if stock_disponibile_centrale >= richiesta_totale:
            # Stock sufficiente per tutti
            allocazioni = richieste_nodi.copy()
            fill_rate_medio = 1.0

        else:
            # Stock insufficiente: fair share ponderato per priorità
            priorita_totale = sum(priorita_nodi.values())
            allocazioni = {}

            for nodo_id, richiesta in richieste_nodi.items():
                peso = priorita_nodi.get(nodo_id, 1.0) / priorita_totale
                allocazione_proporzionale = stock_disponibile_centrale * peso

                # Non superare mai la richiesta
                allocazioni[nodo_id] = min(richiesta, allocazione_proporzionale)

            # Ricalcola se ci sono residui da redistribuire
            stock_allocato = sum(allocazioni.values())
            stock_residuo = stock_disponibile_centrale - stock_allocato

            if stock_residuo > 0:
                # Redistribuisci residuo a nodi che possono ancora ricevere
                for nodo_id in richieste_nodi:
                    gap = richieste_nodi[nodo_id] - allocazioni[nodo_id]
                    if gap > 0 and stock_residuo > 0:
                        extra = min(gap, stock_residuo)
                        allocazioni[nodo_id] += extra
                        stock_residuo -= extra

            fill_rate_medio = sum(allocazioni.values()) / richiesta_totale

        # Calcola metriche per ogni nodo
        dettaglio_nodi = {}
        for nodo_id in richieste_nodi:
            dettaglio_nodi[nodo_id] = {
                "richiesta": richieste_nodi[nodo_id],
                "allocato": allocazioni.get(nodo_id, 0),
                "fill_rate": allocazioni.get(nodo_id, 0) / richieste_nodi[nodo_id]
                if richieste_nodi[nodo_id] > 0
                else 0,
                "priorita": priorita_nodi.get(nodo_id, 1.0),
                "shortage": max(0, richieste_nodi[nodo_id] - allocazioni.get(nodo_id, 0)),
            }

        return {
            "allocazioni": allocazioni,
            "fill_rate_medio": fill_rate_medio,
            "stock_residuo": max(0, stock_disponibile_centrale - sum(allocazioni.values())),
            "dettaglio_nodi": dettaglio_nodi,
            "richiesta_totale": richiesta_totale,
            "efficienza_utilizzo": sum(allocazioni.values()) / stock_disponibile_centrale
            if stock_disponibile_centrale > 0
            else 0,
        }

    def lateral_transshipment(
        self,
        nodo_shortage: str,
        quantita_necessaria: int,
        costo_transshipment_km: float = 0.5,
        distanze: Dict[str, Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calcola ottimo lateral transshipment tra nodi stesso livello
        per risolvere stockout
        """
        nodo_target = self.rete[nodo_shortage]
        candidati_donatori = []

        # Trova nodi stesso livello con stock eccedente
        for nodo_id, nodo in self.rete.items():
            if (
                nodo_id != nodo_shortage
                and nodo.livello == nodo_target.livello
                and nodo.stock_attuale > 0
            ):
                # Calcola stock eccedente (sopra safety stock stimato)
                safety_stock_stimato = nodo.demand_rate * 7  # 1 settimana
                stock_eccedente = max(0, nodo.stock_attuale - safety_stock_stimato)

                if stock_eccedente > 0:
                    # Calcola costo trasporto
                    if distanze and nodo_shortage in distanze.get(nodo_id, {}):
                        distanza = distanze[nodo_id][nodo_shortage]
                        costo_trasporto = distanza * costo_transshipment_km
                    else:
                        costo_trasporto = 100  # Costo default

                    candidati_donatori.append(
                        {
                            "nodo_id": nodo_id,
                            "stock_eccedente": stock_eccedente,
                            "costo_trasporto_unitario": costo_trasporto,
                            "costo_totale": costo_trasporto * quantita_necessaria,
                        }
                    )

        if not candidati_donatori:
            return {
                "fattibile": False,
                "motivo": "Nessun nodo con stock eccedente disponibile",
                "transshipments": [],
            }

        # Ordina per costo totale crescente
        candidati_donatori.sort(key=lambda x: x["costo_trasporto_unitario"])

        # Pianifica transshipments ottimali
        transshipments = []
        quantita_rimanente = quantita_necessaria

        for candidato in candidati_donatori:
            if quantita_rimanente <= 0:
                break

            quantita_da_trasferire = min(quantita_rimanente, candidato["stock_eccedente"])

            if quantita_da_trasferire > 0:
                transshipments.append(
                    {
                        "da_nodo": candidato["nodo_id"],
                        "a_nodo": nodo_shortage,
                        "quantita": quantita_da_trasferire,
                        "costo_unitario": candidato["costo_trasporto_unitario"],
                        "costo_totale": candidato["costo_trasporto_unitario"]
                        * quantita_da_trasferire,
                    }
                )
                quantita_rimanente -= quantita_da_trasferire

        costo_totale_transshipment = sum(t["costo_totale"] for t in transshipments)
        quantita_coperta = sum(t["quantita"] for t in transshipments)

        return {
            "fattibile": quantita_coperta > 0,
            "quantita_coperta": quantita_coperta,
            "quantita_rimanente_scoperta": quantita_rimanente,
            "copertura_percentuale": quantita_coperta / quantita_necessaria * 100,
            "costo_totale": costo_totale_transshipment,
            "costo_unitario_medio": costo_totale_transshipment / quantita_coperta
            if quantita_coperta > 0
            else 0,
            "transshipments": transshipments,
            "numero_nodi_donatori": len(transshipments),
        }


# =====================================================
# 3. CAPACITY CONSTRAINTS MANAGEMENT
# =====================================================


class TipoCapacita(Enum):
    """Tipi di vincoli di capacità"""

    VOLUME = ("volume", "m³", "Spazio fisico in metri cubi")
    PESO = ("weight", "kg", "Peso massimo sostenibile")
    PALLET_POSITIONS = ("pallet", "pallet", "Numero posizioni pallet")
    BUDGET = ("budget", "€", "Budget acquisti disponibile")
    SKU_COUNT = ("sku", "items", "Numero massimo SKU gestibili")
    HANDLING_CAPACITY = ("handling", "units/day", "Capacità movimentazione giornaliera")


class VincoloCapacita(BaseModel):
    """Definizione vincolo di capacità"""

    tipo: TipoCapacita
    capacita_massima: float
    utilizzo_corrente: float
    unita_misura: str
    costo_per_unita: float = 0.0
    penalita_overflow: float = 0.0  # Costo extra se si supera la capacità


class AttributiProdotto(BaseModel):
    """Attributi fisici prodotto per calcoli capacità"""

    volume_m3: float = 0.0
    peso_kg: float = 0.0
    posizioni_pallet_richieste: float = 0.0
    costo_unitario: float = 0.0
    handling_complexity: float = 1.0  # 1.0 = normale, >1 = più complesso


class CapacityConstrainedOptimizer:
    """Ottimizzatore inventory con vincoli di capacità"""

    def __init__(self, vincoli: Dict[str, VincoloCapacita]):
        self.vincoli = vincoli
        self.prodotti_attributi: Dict[str, AttributiProdotto] = {}

    def aggiorna_attributi_prodotto(self, prodotto_id: str, attributi: AttributiProdotto):
        """Aggiorna attributi fisici di un prodotto"""
        self.prodotti_attributi[prodotto_id] = attributi

    def calcola_utilizzo_capacita(
        self, inventario_pianificato: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcola utilizzo previsto di ogni tipo di capacità

        Args:
            inventario_pianificato: {prodotto_id: quantita_pianificata}
        """
        utilizzi = {}

        for vincolo_id, vincolo in self.vincoli.items():
            utilizzo_totale = 0.0

            for prodotto_id, quantita in inventario_pianificato.items():
                if prodotto_id not in self.prodotti_attributi:
                    continue

                attributi = self.prodotti_attributi[prodotto_id]

                if vincolo.tipo == TipoCapacita.VOLUME:
                    utilizzo_totale += quantita * attributi.volume_m3
                elif vincolo.tipo == TipoCapacita.PESO:
                    utilizzo_totale += quantita * attributi.peso_kg
                elif vincolo.tipo == TipoCapacita.PALLET_POSITIONS:
                    utilizzo_totale += quantita * attributi.posizioni_pallet_richieste
                elif vincolo.tipo == TipoCapacita.BUDGET:
                    utilizzo_totale += quantita * attributi.costo_unitario
                elif vincolo.tipo == TipoCapacita.SKU_COUNT:
                    utilizzo_totale += 1 if quantita > 0 else 0
                elif vincolo.tipo == TipoCapacita.HANDLING_CAPACITY:
                    utilizzo_totale += quantita * attributi.handling_complexity / 365  # Giornaliero

            percentuale_utilizzo = (
                (utilizzo_totale / vincolo.capacita_massima * 100)
                if vincolo.capacita_massima > 0
                else 0
            )
            spazio_disponibile = max(0, vincolo.capacita_massima - utilizzo_totale)

            utilizzi[vincolo_id] = {
                "utilizzo_assoluto": utilizzo_totale,
                "capacita_massima": vincolo.capacita_massima,
                "percentuale_utilizzo": percentuale_utilizzo,
                "spazio_disponibile": spazio_disponibile,
                "overflow": max(0, utilizzo_totale - vincolo.capacita_massima),
                "status": "OK" if utilizzo_totale <= vincolo.capacita_massima else "OVERFLOW",
            }

        return utilizzi

    def ottimizza_con_vincoli(
        self,
        richieste_riordino: Dict[str, int],
        priorita_prodotti: Dict[str, float] = None,
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Ottimizza quantità riordino considerando tutti i vincoli di capacità
        usando algoritmo di programmazione lineare semplificato
        """
        if priorita_prodotti is None:
            priorita_prodotti = {pid: 1.0 for pid in richieste_riordino.keys()}

        # Crea lista prodotti ordinata per priorità decrescente
        prodotti_ordinati = sorted(
            richieste_riordino.keys(), key=lambda p: priorita_prodotti.get(p, 0), reverse=True
        )

        # Algoritmo greedy con backtracking
        quantita_approvate = {}
        utilizzi_capacita = {vid: 0.0 for vid in self.vincoli.keys()}

        for prodotto_id in prodotti_ordinati:
            quantita_richiesta = richieste_riordino[prodotto_id]
            quantita_massima_fattibile = quantita_richiesta

            if prodotto_id not in self.prodotti_attributi:
                # Se non abbiamo attributi, approva quantità ridotta
                quantita_approvate[prodotto_id] = min(quantita_richiesta, quantita_richiesta // 2)
                continue

            attributi = self.prodotti_attributi[prodotto_id]

            # Verifica vincoli uno per uno
            for vincolo_id, vincolo in self.vincoli.items():
                utilizzo_attuale = utilizzi_capacita[vincolo_id]
                spazio_rimanente = vincolo.capacita_massima - utilizzo_attuale

                if spazio_rimanente <= 0:
                    quantita_massima_fattibile = 0
                    break

                # Calcola quantità massima per questo vincolo
                if vincolo.tipo == TipoCapacita.VOLUME:
                    if attributi.volume_m3 > 0:
                        max_per_vincolo = int(spazio_rimanente / attributi.volume_m3)
                        quantita_massima_fattibile = min(
                            quantita_massima_fattibile, max_per_vincolo
                        )

                elif vincolo.tipo == TipoCapacita.PESO:
                    if attributi.peso_kg > 0:
                        max_per_vincolo = int(spazio_rimanente / attributi.peso_kg)
                        quantita_massima_fattibile = min(
                            quantita_massima_fattibile, max_per_vincolo
                        )

                elif vincolo.tipo == TipoCapacita.PALLET_POSITIONS:
                    if attributi.posizioni_pallet_richieste > 0:
                        max_per_vincolo = int(
                            spazio_rimanente / attributi.posizioni_pallet_richieste
                        )
                        quantita_massima_fattibile = min(
                            quantita_massima_fattibile, max_per_vincolo
                        )

                elif vincolo.tipo == TipoCapacita.BUDGET:
                    if attributi.costo_unitario > 0:
                        max_per_vincolo = int(spazio_rimanente / attributi.costo_unitario)
                        quantita_massima_fattibile = min(
                            quantita_massima_fattibile, max_per_vincolo
                        )

                elif vincolo.tipo == TipoCapacita.SKU_COUNT:
                    if spazio_rimanente < 1:
                        quantita_massima_fattibile = 0
                    # else mantieni quantità richiesta

            # Approva quantità fattibile
            quantita_finale = max(0, quantita_massima_fattibile)
            quantita_approvate[prodotto_id] = quantita_finale

            # Aggiorna utilizzi
            if quantita_finale > 0:
                for vincolo_id, vincolo in self.vincoli.items():
                    if vincolo.tipo == TipoCapacita.VOLUME:
                        utilizzi_capacita[vincolo_id] += quantita_finale * attributi.volume_m3
                    elif vincolo.tipo == TipoCapacita.PESO:
                        utilizzi_capacita[vincolo_id] += quantita_finale * attributi.peso_kg
                    elif vincolo.tipo == TipoCapacita.PALLET_POSITIONS:
                        utilizzi_capacita[vincolo_id] += (
                            quantita_finale * attributi.posizioni_pallet_richieste
                        )
                    elif vincolo.tipo == TipoCapacita.BUDGET:
                        utilizzi_capacita[vincolo_id] += quantita_finale * attributi.costo_unitario
                    elif vincolo.tipo == TipoCapacita.SKU_COUNT:
                        utilizzi_capacita[vincolo_id] += 1

        # Calcola metriche risultato
        quantita_totale_richiesta = sum(richieste_riordino.values())
        quantita_totale_approvata = sum(quantita_approvate.values())
        fill_rate = (
            quantita_totale_approvata / quantita_totale_richiesta
            if quantita_totale_richiesta > 0
            else 0
        )

        prodotti_rifiutati = [
            pid for pid, qty in quantita_approvate.items() if qty < richieste_riordino[pid]
        ]

        return {
            "quantita_approvate": quantita_approvate,
            "fill_rate": fill_rate,
            "utilizzi_finali": utilizzi_capacita,
            "prodotti_completamente_rifiutati": [
                pid for pid, qty in quantita_approvate.items() if qty == 0
            ],
            "prodotti_parzialmente_approvati": [
                pid for pid, qty in quantita_approvate.items() if 0 < qty < richieste_riordino[pid]
            ],
            "vincoli_saturati": [
                vid
                for vid, utilizzo in utilizzi_capacita.items()
                if utilizzo >= self.vincoli[vid].capacita_massima * 0.95
            ],
        }

    def suggerisci_espansione_capacita(
        self, richieste_non_soddisfatte: Dict[str, int], orizzonte_mesi: int = 12
    ) -> Dict[str, Any]:
        """
        Suggerisce espansioni di capacità per soddisfare domanda non evasa
        """
        espansioni_necessarie = {}

        # Calcola capacità extra necessaria per tipo
        for vincolo_id, vincolo in self.vincoli.items():
            capacita_extra_necessaria = 0.0

            for prodotto_id, quantita_mancante in richieste_non_soddisfatte.items():
                if prodotto_id not in self.prodotti_attributi:
                    continue

                attributi = self.prodotti_attributi[prodotto_id]

                if vincolo.tipo == TipoCapacita.VOLUME:
                    capacita_extra_necessaria += quantita_mancante * attributi.volume_m3
                elif vincolo.tipo == TipoCapacita.PESO:
                    capacita_extra_necessaria += quantita_mancante * attributi.peso_kg
                elif vincolo.tipo == TipoCapacita.PALLET_POSITIONS:
                    capacita_extra_necessaria += (
                        quantita_mancante * attributi.posizioni_pallet_richieste
                    )
                elif vincolo.tipo == TipoCapacita.BUDGET:
                    capacita_extra_necessaria += quantita_mancante * attributi.costo_unitario

            if capacita_extra_necessaria > 0:
                # Calcola ROI investimento
                costo_espansione = capacita_extra_necessaria * vincolo.costo_per_unita
                valore_vendite_abilitate = sum(
                    richieste_non_soddisfatte[pid]
                    * self.prodotti_attributi[pid].costo_unitario
                    * 0.3  # 30% margine stimato
                    for pid in richieste_non_soddisfatte
                    if pid in self.prodotti_attributi
                )

                payback_mesi = (
                    (costo_espansione / valore_vendite_abilitate * orizzonte_mesi)
                    if valore_vendite_abilitate > 0
                    else 999
                )

                espansioni_necessarie[vincolo_id] = {
                    "capacita_extra_necessaria": capacita_extra_necessaria,
                    "capacita_attuale": vincolo.capacita_massima,
                    "aumento_percentuale": capacita_extra_necessaria
                    / vincolo.capacita_massima
                    * 100,
                    "costo_investimento": costo_espansione,
                    "valore_vendite_abilitate_annuo": valore_vendite_abilitate,
                    "payback_mesi": payback_mesi,
                    "roi_annuo": (valore_vendite_abilitate / costo_espansione * 100)
                    if costo_espansione > 0
                    else 0,
                }

        # Ordina per ROI decrescente
        espansioni_ordinate = dict(
            sorted(espansioni_necessarie.items(), key=lambda x: x[1]["roi_annuo"], reverse=True)
        )

        return {
            "espansioni_consigliate": espansioni_ordinate,
            "investimento_totale": sum(
                exp["costo_investimento"] for exp in espansioni_necessarie.values()
            ),
            "roi_medio_ponderato": sum(
                exp["roi_annuo"] * exp["costo_investimento"]
                for exp in espansioni_necessarie.values()
            )
            / sum(exp["costo_investimento"] for exp in espansioni_necessarie.values())
            if espansioni_necessarie
            else 0,
        }


# =====================================================
# 4. KITTING & BUNDLE OPTIMIZATION
# =====================================================


class TipoComponente(Enum):
    """Tipi di componenti in un kit"""

    MASTER = ("master", "Componente principale del kit")
    STANDARD = ("standard", "Componente standard sostituibile")
    OPTIONAL = ("optional", "Componente opzionale")
    CONSUMABLE = ("consumable", "Consumabile da rifornire")


class ComponenteKit(BaseModel):
    """Definizione componente di un kit"""

    componente_id: str
    nome: str
    tipo: TipoComponente
    quantita_per_kit: int
    costo_unitario: float
    lead_time: int
    criticalita: float = 1.0  # 0-1, 1 = critico
    sostituibili: List[str] = []  # Lista ID componenti sostituibili


class DefinzioneKit(BaseModel):
    """Definizione completa di un kit"""

    kit_id: str
    nome: str
    componenti: List[ComponenteKit]
    prezzo_vendita_kit: float
    margine_target: float
    domanda_storica_kit: List[float]
    can_sell_components_separately: bool = True


class KittingOptimizer:
    """Ottimizzatore per gestione kit e bundle"""

    def __init__(self, definizioni_kit: Dict[str, DefinzioneKit]):
        self.kit_catalog = definizioni_kit
        self.inventory_componenti: Dict[str, int] = {}  # {componente_id: stock}

    def aggiorna_inventory_componente(self, componente_id: str, stock: int):
        """Aggiorna livello stock di un componente"""
        self.inventory_componenti[componente_id] = stock

    def calcola_kit_assemblabili(self, kit_id: str) -> Dict[str, Any]:
        """
        Calcola quanti kit possono essere assemblati con inventory corrente
        """
        if kit_id not in self.kit_catalog:
            return {"errore": f"Kit {kit_id} non trovato"}

        kit_def = self.kit_catalog[kit_id]
        limitazioni = {}
        kit_max_assemblabili = float("inf")

        for componente in kit_def.componenti:
            stock_disponibile = self.inventory_componenti.get(componente.componente_id, 0)
            kit_possibili_da_componente = stock_disponibile // componente.quantita_per_kit

            if componente.tipo == TipoComponente.OPTIONAL:
                # Componenti opzionali non limitano l'assemblaggio
                continue

            if kit_possibili_da_componente < kit_max_assemblabili:
                kit_max_assemblabili = kit_possibili_da_componente
                limitazioni[componente.componente_id] = {
                    "stock_disponibile": stock_disponibile,
                    "quantita_per_kit": componente.quantita_per_kit,
                    "kit_possibili": kit_possibili_da_componente,
                    "shortage": max(0, componente.quantita_per_kit - stock_disponibile),
                }

        # Considera componenti sostituibili
        componenti_mancanti = []
        for comp_id, info in limitazioni.items():
            if info["kit_possibili"] == 0:
                componente = next(c for c in kit_def.componenti if c.componente_id == comp_id)
                if componente.sostituibili:
                    # Verifica se componenti sostituibili possono risolvere
                    for sostituto_id in componente.sostituibili:
                        stock_sostituto = self.inventory_componenti.get(sostituto_id, 0)
                        if stock_sostituto >= componente.quantita_per_kit:
                            kit_max_assemblabili = max(1, kit_max_assemblabili)
                            break
                    else:
                        componenti_mancanti.append(comp_id)
                else:
                    componenti_mancanti.append(comp_id)

        if kit_max_assemblabili == float("inf"):
            kit_max_assemblabili = 1000  # Limite pratico

        return {
            "kit_assemblabili": int(max(0, kit_max_assemblabili)),
            "componente_limitante": min(limitazioni.items(), key=lambda x: x[1]["kit_possibili"])[0]
            if limitazioni
            else None,
            "componenti_mancanti": componenti_mancanti,
            "limitazioni_dettaglio": limitazioni,
            "valore_inventory_impegnato": sum(
                comp.quantita_per_kit * comp.costo_unitario * kit_max_assemblabili
                for comp in kit_def.componenti
                if comp.tipo != TipoComponente.OPTIONAL
            ),
        }

    def ottimizza_kit_vs_componenti(
        self,
        kit_id: str,
        forecast_kit: np.ndarray,
        forecast_componenti_separati: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Decide se conviene vendere come kit o componenti separati
        basato su profitabilità e disponibilità
        """
        kit_def = self.kit_catalog[kit_id]

        # Calcola profitabilità kit vs componenti separati
        costo_totale_componenti = sum(
            comp.costo_unitario * comp.quantita_per_kit for comp in kit_def.componenti
        )
        margine_kit = kit_def.prezzo_vendita_kit - costo_totale_componenti
        margine_kit_pct = (
            margine_kit / kit_def.prezzo_vendita_kit if kit_def.prezzo_vendita_kit > 0 else 0
        )

        # Calcola ricavi da vendita componenti separati
        ricavi_componenti_separati = 0
        margini_componenti_separati = 0

        for componente in kit_def.componenti:
            if componente.componente_id in forecast_componenti_separati:
                domanda_separata = np.mean(forecast_componenti_separati[componente.componente_id])
                prezzo_vendita_separato = componente.costo_unitario * 1.4  # 40% markup stimato
                ricavo_annuo = domanda_separata * 365 * prezzo_vendita_separato
                margine_annuo = ricavo_annuo - (domanda_separata * 365 * componente.costo_unitario)

                ricavi_componenti_separati += ricavo_annuo
                margini_componenti_separati += margine_annuo

        # Calcola ricavi da vendita kit
        domanda_media_kit = np.mean(forecast_kit)
        ricavo_kit_annuo = domanda_media_kit * 365 * kit_def.prezzo_vendita_kit
        margine_kit_annuo = domanda_media_kit * 365 * margine_kit

        # Analisi disponibilità
        kit_info = self.calcola_kit_assemblabili(kit_id)

        # Decisione strategica
        if margine_kit_annuo > margini_componenti_separati * 1.1:  # 10% premium per kit
            strategia_consigliata = "Kit preferito"
            focus_principale = "kit"
        elif kit_info["kit_assemblabili"] < domanda_media_kit * 30:  # Meno di 1 mese stock
            strategia_consigliata = "Componenti separati (disponibilità kit limitata)"
            focus_principale = "componenti"
        else:
            strategia_consigliata = "Strategia mista"
            focus_principale = "misto"

        return {
            "strategia_consigliata": strategia_consigliata,
            "focus_principale": focus_principale,
            "analisi_finanziaria": {
                "margine_kit_annuo": margine_kit_annuo,
                "margine_componenti_annuo": margini_componenti_separati,
                "differenza_margine": margine_kit_annuo - margini_componenti_separati,
                "roi_kit_vs_componenti": (margine_kit_annuo / margini_componenti_separati)
                if margini_componenti_separati > 0
                else 0,
            },
            "analisi_disponibilita": kit_info,
            "raccomandazioni_procurement": self._genera_raccomandazioni_procurement(
                kit_id, kit_info, domanda_media_kit
            ),
        }

    def _genera_raccomandazioni_procurement(
        self, kit_id: str, kit_info: Dict, domanda_media_giornaliera: float
    ) -> List[str]:
        """Genera raccomandazioni di approvvigionamento per componenti kit"""
        raccomandazioni = []
        kit_def = self.kit_catalog[kit_id]

        giorni_copertura_target = 30

        for componente in kit_def.componenti:
            stock_attuale = self.inventory_componenti.get(componente.componente_id, 0)
            consumo_giornaliero = domanda_media_giornaliera * componente.quantita_per_kit
            giorni_copertura = (
                stock_attuale / consumo_giornaliero if consumo_giornaliero > 0 else 999
            )

            if giorni_copertura < 7:
                raccomandazioni.append(
                    f"URGENTE: Riordinare {componente.nome} (copertura: {giorni_copertura:.1f} giorni)"
                )
            elif giorni_copertura < 15:
                raccomandazioni.append(f"ATTENZIONE: Pianificare riordino {componente.nome}")
            elif (
                componente.tipo == TipoComponente.MASTER
                and giorni_copertura < giorni_copertura_target
            ):
                quantita_riordino = (giorni_copertura_target * consumo_giornaliero) - stock_attuale
                raccomandazioni.append(
                    f"Riordinare {quantita_riordino:.0f} unità di {componente.nome}"
                )

        # Controllo componenti sostituibili
        componenti_critici = [c for c in kit_def.componenti if c.criticalita > 0.8]
        for componente in componenti_critici:
            if not componente.sostituibili:
                raccomandazioni.append(
                    f"RISCHIO: {componente.nome} è critico ma non ha sostituibili"
                )

        return raccomandazioni

    def pianifica_disassembly(
        self, kit_id: str, quantita_kit_da_disfare: int, domanda_componenti: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Pianifica disassemblaggio kit per liberare componenti richiesti
        """
        kit_def = self.kit_catalog[kit_id]

        # Calcola componenti liberati
        componenti_liberati = {}
        valore_recuperato = 0

        for componente in kit_def.componenti:
            quantita_liberata = quantita_kit_da_disfare * componente.quantita_per_kit
            componenti_liberati[componente.componente_id] = quantita_liberata

            # Valore recuperato se il componente ha domanda separata
            domanda_comp = domanda_componenti.get(componente.componente_id, 0)
            if domanda_comp > 0:
                prezzo_vendita_comp = componente.costo_unitario * 1.4  # 40% markup
                valore_recuperato += (
                    min(quantita_liberata, domanda_comp * 365) * prezzo_vendita_comp
                )

        # Costo opportunità (perdita vendita kit)
        costo_opportunita = quantita_kit_da_disfare * (
            kit_def.prezzo_vendita_kit
            - sum(comp.costo_unitario * comp.quantita_per_kit for comp in kit_def.componenti)
        )

        # Analisi convenienza
        convenienza = valore_recuperato - costo_opportunita

        return {
            "componenti_liberati": componenti_liberati,
            "valore_recuperato": valore_recuperato,
            "costo_opportunita": costo_opportunita,
            "convenienza_netta": convenienza,
            "raccomandazione": "PROCEDI" if convenienza > 0 else "SCONSIGLIATO",
            "ratio_convenienza": valore_recuperato / costo_opportunita
            if costo_opportunita > 0
            else float("inf"),
        }


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
        "nome": "Carrozzina Standard CRZ001",
        "prezzo_unitario": 280,
        "margine_lordo": 84,  # 30% margine
        "lead_time_giorni": 15,
        "spazio_mq": 0.5,
        "criticita": "ALTO",
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

    vendite_df = pd.DataFrame(
        {"data": pd.date_range(start="2024-01-01", periods=giorni), "quantita": vendite}
    )

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
        lead_time_days=prodotto["lead_time_giorni"],
        service_level=0.95,  # 95% service level
        criticality_factor=1.2,  # Prodotto critico
        seasonality_factor=1.1,  # Leggera stagionalità
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
        unit_cost=prodotto["prezzo_unitario"],
    )

    reorder_point = calculator.calculate_reorder_point(
        demand_mean=vendite.mean(),
        lead_time_days=prodotto["lead_time_giorni"],
        safety_stock=safety_results["dynamic_safety_stock"],
    )

    print(f"EOQ (Quantità Ordine Economica): {eoq} unità")
    print(f"Reorder Point: {reorder_point:.0f} unità")
    print(f"Ordini annui previsti: {vendite.sum() / eoq:.1f}")

    # 5. Analisi Costi Totali
    print("\n[5] OTTIMIZZAZIONE COSTI TOTALI")
    print("-" * 40)

    costi = CostiGiacenza()
    analyzer = TotalCostAnalyzer(costi)

    optimal = analyzer.find_optimal_inventory_level(
        demand_forecast=vendite[-30:],  # Ultimi 30 giorni
        unit_cost=prodotto["prezzo_unitario"],
        gross_margin=prodotto["margine_lordo"],
        space_per_unit=prodotto["spazio_mq"],
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
    max_stock = eoq + safety_results["dynamic_safety_stock"]

    alert_system = InventoryAlertSystem()
    analisi_rischio = alert_system.check_inventory_status(
        current_stock=stock_corrente,
        safety_stock=safety_results["dynamic_safety_stock"],
        reorder_point=reorder_point,
        max_stock=max_stock,
        daily_demand=vendite.mean(),
        lead_time_days=prodotto["lead_time_giorni"],
    )

    print(f"Stock Corrente: {stock_corrente} unità")
    print(
        f"Livello Alert: {analisi_rischio.livello_alert.value[0].upper()} - {analisi_rischio.livello_alert.value[1]}"
    )
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
    sales_data.columns = ["date", "quantity"]

    inventory_data = pd.DataFrame(
        {
            "date": pd.date_range(start="2024-01-01", periods=30),
            "stock_level": np.random.normal(stock_corrente, 50, 30),
        }
    )

    costs_data = {
        "unit_cost": prodotto["prezzo_unitario"],
        "gross_margin": prodotto["margine_lordo"] / prodotto["prezzo_unitario"],
        "payment_terms": 30,
        "supplier_terms": 45,
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

    # 10. CLASSIFICAZIONE SLOW/FAST MOVING
    print("\n[10] ANALISI SLOW/FAST MOVING")
    print("-" * 40)

    # Classifica il prodotto
    classifier = MovementClassifier()
    turnover = kpis["inventory_turnover"]
    categoria_movimento = classifier.classify_by_movement(turnover)
    classe_xyz = classifier.classify_xyz(pd.Series(vendite))

    print(f"Classificazione Movimento: {categoria_movimento.value[1]} (Turnover: {turnover}x)")
    print(
        f"Classificazione Variabilità: {classe_xyz.value[1]} (CV: {vendite.std() / vendite.mean():.2f})"
    )

    # 11. OTTIMIZZAZIONE SLOW/FAST MOVING
    print("\n[11] OTTIMIZZAZIONE SPECIFICA PER CATEGORIA")
    print("-" * 40)

    optimizer = SlowFastOptimizer(costi)

    # Politica attuale per confronto
    current_policy = {
        "safety_stock": safety_results["dynamic_safety_stock"],
        "eoq": eoq,
        "reorder_point": reorder_point,
        "avg_inventory": stock_corrente,
    }

    # Confronta strategie
    comparison = optimizer.compare_strategies(
        vendite[-30:], prodotto["prezzo_unitario"], prodotto["lead_time_giorni"], current_policy
    )

    print("\nConfronto Strategie:")
    print(comparison.to_string(index=False))

    # 12. STRATEGIA RACCOMANDATA
    print("\n[12] STRATEGIA RACCOMANDATA PER CATEGORIA")
    print("-" * 40)

    # Simula classe ABC (assumiamo classe A per questo esempio)
    classe_abc = ClassificazioneABC.A

    strategia = classifier.get_strategy_by_classification(
        categoria_movimento, classe_abc, classe_xyz
    )

    print(f"Strategia: {strategia['strategia']}")
    print(f"Service Level Target: {strategia['service_level']:.0%}")
    print(f"Periodo Revisione: {strategia['review_period']}")
    print(f"Politica Ordini: {strategia['ordering_policy']}")
    print(f"Fattore Safety Stock: {strategia['safety_stock_factor']}")

    print("\n" + "=" * 60)
    print("ANALISI COMPLETATA CON SUCCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    esempio_bilanciamento_completo()
