"""
What-If Scenario Simulator per Dashboard.

Simulatore interattivo per analisi scenari business e decision making.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ScenarioType(Enum):
    """Tipi di scenario predefiniti."""
    CUSTOM = "Scenario Personalizzato"
    MARKETING_BOOST = "Campagna Marketing" 
    ECONOMIC_CRISIS = "Crisi Economica"
    SUPPLIER_ISSUE = "Problema Fornitore"
    SEASONAL_PEAK = "Picco Stagionale"
    PRICE_WAR = "Guerra Prezzi"
    NEW_COMPETITOR = "Nuovo Competitor"
    BLACK_FRIDAY = "Black Friday"


@dataclass
class ScenarioParameters:
    """Parametri per simulazione scenario."""
    # Demand drivers
    marketing_boost: float = 0.0  # % incremento da marketing
    price_change: float = 0.0     # % cambio prezzo
    seasonality_factor: float = 1.0  # moltiplicatore stagionalità
    
    # Supply constraints  
    supplier_reliability: float = 95.0  # % affidabilità fornitore
    lead_time_change: int = 0       # giorni variazione lead time
    capacity_limit: float = 100.0   # % capacità produttiva
    
    # Economic factors
    inflation_rate: float = 3.0     # % inflazione annua
    exchange_rate: float = 0.0      # % variazione cambio
    interest_rate: float = 2.0      # % tasso interesse
    
    # External shocks
    competitor_impact: float = 0.0  # % impatto competitor
    regulatory_impact: float = 0.0  # % impatto normativo
    weather_impact: float = 0.0     # % impatto meteo


@dataclass 
class ScenarioResults:
    """Risultati simulazione scenario."""
    revenue_impact: float
    revenue_change_pct: float
    inventory_investment: float
    inventory_change_pct: float
    service_level: float
    service_risk: float
    profit_impact: float
    cash_flow_impact: float
    break_even_days: int
    roi_3months: float
    recommendations: List[str]


class WhatIfScenarioSimulator:
    """Simulatore What-If per analisi scenari business."""
    
    def __init__(self):
        """Inizializza il simulatore."""
        self.base_forecast = None
        self.base_metrics = {}
        self.scenario_cache = {}
        
    def create_scenario_ui(self) -> ScenarioParameters:
        """Crea interfaccia utente per configurazione scenario."""
        
        st.subheader("🎯 What-If Scenario Simulator")
        st.markdown("Simula l'impatto di diversi fattori sui tuoi risultati business")
        
        # Selezione scenario predefinito
        col1, col2 = st.columns([2, 1])
        
        with col1:
            scenario_type = st.selectbox(
                "Scenario Predefinito",
                options=list(ScenarioType),
                format_func=lambda x: x.value,
                help="Seleziona uno scenario predefinito o personalizza i parametri"
            )
        
        with col2:
            if st.button("🔄 Reset Parametri"):
                st.rerun()
        
        # Carica parametri predefiniti
        if scenario_type != ScenarioType.CUSTOM:
            params = self._get_predefined_scenario(scenario_type)
        else:
            params = ScenarioParameters()
        
        # Interfaccia controlli
        st.markdown("---")
        
        # Organizza controlli in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Demand Drivers", 
            "🏭 Supply Chain", 
            "💰 Economics", 
            "⚡ External Shocks"
        ])
        
        with tab1:
            st.markdown("**Fattori che influenzano la domanda**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                params.marketing_boost = st.slider(
                    "Marketing Campaign Impact",
                    min_value=-50, max_value=200, 
                    value=int(params.marketing_boost),
                    step=5,
                    help="% incremento domanda da campagne marketing",
                    format="%d%%"
                )
            
            with col2:
                params.price_change = st.slider(
                    "Price Change",
                    min_value=-30, max_value=30,
                    value=int(params.price_change),
                    step=1,
                    help="% cambio prezzo prodotto (elasticità -1.5)",
                    format="%d%%"
                )
            
            with col3:
                params.seasonality_factor = st.slider(
                    "Seasonal Multiplier", 
                    min_value=0.5, max_value=2.5,
                    value=params.seasonality_factor,
                    step=0.1,
                    help="Moltiplicatore per effetti stagionali"
                )
        
        with tab2:
            st.markdown("**Vincoli della supply chain**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                params.supplier_reliability = st.slider(
                    "Supplier Reliability",
                    min_value=70, max_value=100,
                    value=int(params.supplier_reliability),
                    step=1,
                    help="% affidabilità consegne fornitore",
                    format="%d%%"
                )
            
            with col2:
                params.lead_time_change = st.slider(
                    "Lead Time Change",
                    min_value=-7, max_value=21,
                    value=params.lead_time_change,
                    step=1,
                    help="Giorni variazione lead time medio",
                    format="%d giorni"
                )
            
            with col3:
                params.capacity_limit = st.slider(
                    "Production Capacity",
                    min_value=50, max_value=150,
                    value=int(params.capacity_limit),
                    step=5,
                    help="% capacità produttiva disponibile",
                    format="%d%%"
                )
        
        with tab3:
            st.markdown("**Fattori macroeconomici**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                params.inflation_rate = st.slider(
                    "Inflation Rate",
                    min_value=0.0, max_value=15.0,
                    value=params.inflation_rate,
                    step=0.5,
                    help="% inflazione annua",
                    format="%.1f%%"
                )
            
            with col2:
                params.exchange_rate = st.slider(
                    "Exchange Rate Change", 
                    min_value=-25, max_value=25,
                    value=int(params.exchange_rate),
                    step=1,
                    help="% variazione tasso di cambio",
                    format="%d%%"
                )
            
            with col3:
                params.interest_rate = st.slider(
                    "Interest Rate",
                    min_value=0.0, max_value=10.0,
                    value=params.interest_rate,
                    step=0.25,
                    help="% tasso interesse (costo capitale)",
                    format="%.2f%%"
                )
        
        with tab4:
            st.markdown("**Shock esterni e eventi**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                params.competitor_impact = st.slider(
                    "Competitor Impact",
                    min_value=-40, max_value=10,
                    value=int(params.competitor_impact),
                    step=5,
                    help="% impatto nuovo competitor/prodotto",
                    format="%d%%"
                )
            
            with col2:
                params.regulatory_impact = st.slider(
                    "Regulatory Changes",
                    min_value=-30, max_value=20,
                    value=int(params.regulatory_impact),
                    step=5, 
                    help="% impatto cambiamenti normativi",
                    format="%d%%"
                )
            
            with col3:
                params.weather_impact = st.slider(
                    "Weather/Seasonal Events",
                    min_value=-25, max_value=50,
                    value=int(params.weather_impact),
                    step=5,
                    help="% impatto eventi meteo/stagionali",
                    format="%d%%"
                )
        
        return params
    
    def _get_predefined_scenario(self, scenario_type: ScenarioType) -> ScenarioParameters:
        """Restituisce parametri per scenari predefiniti."""
        
        scenarios = {
            ScenarioType.MARKETING_BOOST: ScenarioParameters(
                marketing_boost=150.0,
                price_change=-10.0,
                seasonality_factor=1.2
            ),
            ScenarioType.ECONOMIC_CRISIS: ScenarioParameters(
                marketing_boost=-20.0,
                price_change=5.0,
                inflation_rate=8.0,
                interest_rate=6.0,
                seasonality_factor=0.8
            ),
            ScenarioType.SUPPLIER_ISSUE: ScenarioParameters(
                supplier_reliability=70.0,
                lead_time_change=14,
                capacity_limit=60.0
            ),
            ScenarioType.SEASONAL_PEAK: ScenarioParameters(
                seasonality_factor=2.2,
                marketing_boost=50.0,
                capacity_limit=120.0
            ),
            ScenarioType.PRICE_WAR: ScenarioParameters(
                price_change=-25.0,
                competitor_impact=-30.0,
                marketing_boost=80.0
            ),
            ScenarioType.NEW_COMPETITOR: ScenarioParameters(
                competitor_impact=-25.0,
                marketing_boost=40.0,
                price_change=-15.0
            ),
            ScenarioType.BLACK_FRIDAY: ScenarioParameters(
                marketing_boost=300.0,
                price_change=-40.0,
                seasonality_factor=1.8,
                capacity_limit=80.0
            )
        }
        
        return scenarios.get(scenario_type, ScenarioParameters())
    
    def run_scenario_simulation(
        self, 
        params: ScenarioParameters,
        base_forecast: pd.Series,
        base_metrics: Dict[str, Any]
    ) -> Tuple[pd.Series, ScenarioResults]:
        """
        Esegue simulazione scenario completa.
        
        Args:
            params: Parametri scenario
            base_forecast: Previsioni baseline
            base_metrics: Metriche baseline
            
        Returns:
            Tuple[pd.Series, ScenarioResults]: Forecast scenario e risultati
        """
        
        # 1. Calcola impatti sulla domanda
        demand_multiplier = self._calculate_demand_impact(params)
        
        # 2. Applica vincoli supply chain
        supply_constraint = self._calculate_supply_constraints(params)
        
        # 3. Calcola impatto economico
        cost_multiplier = self._calculate_economic_impact(params)
        
        # 4. Genera forecast scenario
        scenario_forecast = base_forecast * demand_multiplier * supply_constraint
        
        # 5. Calcola metriche impatto
        results = self._calculate_scenario_impact(
            base_forecast, scenario_forecast, 
            base_metrics, params, cost_multiplier
        )
        
        return scenario_forecast, results
    
    def _calculate_demand_impact(self, params: ScenarioParameters) -> float:
        """Calcola impatto complessivo sulla domanda."""
        
        # Marketing impact (diretto)
        marketing_factor = (100 + params.marketing_boost) / 100
        
        # Price elasticity (elasticità -1.5 tipica)
        price_elasticity = 1 + (params.price_change / 100) * (-1.5)
        
        # Seasonal factor
        seasonal_factor = params.seasonality_factor
        
        # Competitor impact
        competitor_factor = (100 + params.competitor_impact) / 100
        
        # Regulatory impact  
        regulatory_factor = (100 + params.regulatory_impact) / 100
        
        # Weather/external events
        weather_factor = (100 + params.weather_impact) / 100
        
        # Combine all factors
        total_demand_multiplier = (
            marketing_factor * 
            price_elasticity * 
            seasonal_factor * 
            competitor_factor * 
            regulatory_factor * 
            weather_factor
        )
        
        return max(0.1, total_demand_multiplier)  # Minimo 10% della domanda base
    
    def _calculate_supply_constraints(self, params: ScenarioParameters) -> float:
        """Calcola vincoli supply chain."""
        
        # Reliability impact
        reliability_factor = params.supplier_reliability / 100
        
        # Lead time impact (maggiore lead time = maggiore variabilità)
        if params.lead_time_change > 0:
            leadtime_factor = 1 - (params.lead_time_change * 0.02)  # -2% per giorno extra
        else:
            leadtime_factor = 1 + abs(params.lead_time_change * 0.01)  # +1% per giorno risparmiato
        
        # Capacity constraint
        capacity_factor = min(1.0, params.capacity_limit / 100)
        
        return reliability_factor * leadtime_factor * capacity_factor
    
    def _calculate_economic_impact(self, params: ScenarioParameters) -> float:
        """Calcola moltiplicatori costi economici."""
        
        # Inflation impact sui costi
        inflation_factor = 1 + (params.inflation_rate / 100)
        
        # Exchange rate impact (assume 30% costi in valuta estera)
        fx_factor = 1 + (params.exchange_rate / 100) * 0.3
        
        # Interest rate impact su holding costs
        interest_factor = 1 + (params.interest_rate / 100) * 0.5
        
        return inflation_factor * fx_factor * interest_factor
    
    def _calculate_scenario_impact(
        self,
        base_forecast: pd.Series,
        scenario_forecast: pd.Series,
        base_metrics: Dict[str, Any],
        params: ScenarioParameters,
        cost_multiplier: float
    ) -> ScenarioResults:
        """Calcola metriche di impatto dello scenario."""
        
        # Assumi prezzi e costi base realistici
        base_price = base_metrics.get('unit_price', 150.0)
        base_cost = base_metrics.get('unit_cost', 90.0)
        base_volume = base_forecast.sum()
        scenario_volume = scenario_forecast.sum()
        
        # Price adjustment per scenario
        scenario_price = base_price * (1 + params.price_change / 100)
        scenario_cost = base_cost * cost_multiplier
        
        # Revenue impact
        base_revenue = base_volume * base_price
        scenario_revenue = scenario_volume * scenario_price
        revenue_impact = scenario_revenue - base_revenue
        revenue_change_pct = (revenue_impact / base_revenue) * 100
        
        # Inventory investment (assume 2 months coverage)
        base_inventory_value = (base_volume / len(base_forecast)) * 60 * base_cost
        scenario_inventory_value = (scenario_volume / len(scenario_forecast)) * 60 * scenario_cost
        inventory_investment = scenario_inventory_value - base_inventory_value
        inventory_change_pct = (inventory_investment / base_inventory_value) * 100
        
        # Service level impact
        base_service_level = base_metrics.get('service_level', 92.0)
        
        # Service level decreases with supply constraints and demand volatility
        service_impact = (
            (params.supplier_reliability - 95) * 0.2 +  
            (params.lead_time_change * -0.5) +
            (max(0, abs(revenue_change_pct) - 20) * -0.1)
        )
        scenario_service_level = max(70, base_service_level + service_impact)
        service_risk = base_service_level - scenario_service_level
        
        # Profit impact
        base_profit = base_volume * (base_price - base_cost)
        scenario_profit = scenario_volume * (scenario_price - scenario_cost)
        profit_impact = scenario_profit - base_profit
        
        # Cash flow impact (inventory investment impact)
        cash_flow_impact = revenue_impact - inventory_investment
        
        # Break-even calculation
        if profit_impact > 0:
            break_even_days = max(1, int(abs(inventory_investment) / (profit_impact / 90)))
        else:
            break_even_days = 999  # Non raggiungibile
        
        # ROI 3 months
        three_month_profit = profit_impact * (90 / len(scenario_forecast))
        roi_3months = (three_month_profit / max(abs(inventory_investment), 1000)) * 100
        
        # Raccomandazioni
        recommendations = self._generate_recommendations(params, revenue_change_pct, service_risk)
        
        return ScenarioResults(
            revenue_impact=revenue_impact,
            revenue_change_pct=revenue_change_pct,
            inventory_investment=inventory_investment,
            inventory_change_pct=inventory_change_pct,
            service_level=scenario_service_level,
            service_risk=service_risk,
            profit_impact=profit_impact,
            cash_flow_impact=cash_flow_impact,
            break_even_days=break_even_days,
            roi_3months=roi_3months,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self, 
        params: ScenarioParameters, 
        revenue_change_pct: float,
        service_risk: float
    ) -> List[str]:
        """Genera raccomandazioni basate sui risultati scenario."""
        
        recommendations = []
        
        # Revenue-based recommendations
        if revenue_change_pct > 50:
            recommendations.append("🚀 Scenario ad alto potenziale: pianifica scale-up produzione")
            recommendations.append("📦 Aumenta safety stock per evitare stockout durante picco")
        elif revenue_change_pct < -20:
            recommendations.append("⚠️ Scenario negativo: implementa piani di contingenza")
            recommendations.append("💰 Riduci inventory exposure per limitare cash flow impact")
        
        # Supply chain recommendations  
        if params.supplier_reliability < 85:
            recommendations.append("🏭 Reliability bassa: attiva fornitori backup immediata")
        
        if params.lead_time_change > 10:
            recommendations.append("⏱️ Lead time estesi: aumenta safety stock del 25%")
        
        # Service level recommendations
        if service_risk > 5:
            recommendations.append("📋 Rischio service level: implenta monitoring giornaliero")
        
        # Economic recommendations
        if params.inflation_rate > 6:
            recommendations.append("💸 Alta inflazione: negozia contratti prezzo fisso")
        
        # Marketing recommendations
        if params.marketing_boost > 100:
            recommendations.append("📢 Campaign intensive: prepara fulfillment per volume extra")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("✅ Scenario bilanciato: procedi con piani attuali")
        
        return recommendations[:5]  # Max 5 raccomandazioni
    
    def create_scenario_visualization(
        self,
        base_forecast: pd.Series,
        scenario_forecast: pd.Series, 
        results: ScenarioResults
    ) -> go.Figure:
        """Crea visualizzazione comparativa scenario."""
        
        # Crea subplot con metriche multiple
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Demand Forecast Comparison',
                'Revenue Impact Over Time', 
                'Service Level vs Cost',
                'ROI Timeline'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # 1. Forecast comparison
        fig.add_trace(
            go.Scatter(
                x=base_forecast.index,
                y=base_forecast.values,
                name='Baseline Forecast',
                line=dict(color='blue', width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=scenario_forecast.index,
                y=scenario_forecast.values,
                name='Scenario Forecast',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Fill area between curves
        fig.add_trace(
            go.Scatter(
                x=scenario_forecast.index,
                y=scenario_forecast.values,
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)' if results.revenue_change_pct > 0 else 'rgba(0,0,255,0.1)',
                name='Impact Zone',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # 2. Revenue impact over time (cumulative)
        cumulative_base = base_forecast.cumsum() * 150  # Assume €150 per unit
        cumulative_scenario = scenario_forecast.cumsum() * 150 * (1 + results.revenue_change_pct/100)
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_base.index,
                y=cumulative_base.values,
                name='Base Revenue',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_scenario.index, 
                y=cumulative_scenario.values,
                name='Scenario Revenue',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
        
        # 3. Service level vs cost scatter
        scenarios_data = {
            'Current': (92, 100),
            'Scenario': (results.service_level, 100 + results.inventory_change_pct),
            'Optimistic': (96, 110),
            'Pessimistic': (85, 90)
        }
        
        for name, (service, cost) in scenarios_data.items():
            color = 'red' if name == 'Scenario' else 'blue'
            size = 15 if name == 'Scenario' else 10
            
            fig.add_trace(
                go.Scatter(
                    x=[service],
                    y=[cost],
                    name=name,
                    mode='markers',
                    marker=dict(color=color, size=size),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # 4. ROI timeline
        timeline_months = ['Month 1', 'Month 2', 'Month 3', 'Month 6', 'Month 12']
        roi_progression = [
            results.roi_3months * 0.3,
            results.roi_3months * 0.7, 
            results.roi_3months,
            results.roi_3months * 1.8,
            results.roi_3months * 3.2
        ]
        
        fig.add_trace(
            go.Bar(
                x=timeline_months,
                y=roi_progression,
                name='ROI %',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            title_text="Scenario Analysis Dashboard",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right", 
                x=1
            )
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Timeline", row=1, col=1)
        fig.update_yaxes(title_text="Demand Units", row=1, col=1)
        
        fig.update_xaxes(title_text="Timeline", row=1, col=2)
        fig.update_yaxes(title_text="Revenue (€)", row=1, col=2)
        
        fig.update_xaxes(title_text="Service Level (%)", row=2, col=1)
        fig.update_yaxes(title_text="Cost Index", row=2, col=1)
        
        fig.update_xaxes(title_text="Timeline", row=2, col=2)
        fig.update_yaxes(title_text="ROI (%)", row=2, col=2)
        
        return fig
    
    def display_scenario_results(self, results: ScenarioResults):
        """Visualizza risultati scenario in formato dashboard."""
        
        st.markdown("### 📊 Scenario Impact Analysis")
        
        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Revenue Impact",
                f"€{results.revenue_impact:,.0f}",
                delta=f"{results.revenue_change_pct:+.1f}%",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Inventory Investment", 
                f"€{results.inventory_investment:,.0f}",
                delta=f"{results.inventory_change_pct:+.1f}%",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Service Level",
                f"{results.service_level:.1f}%",
                delta=f"{-results.service_risk:+.1f}%",
                delta_color="normal"
            )
        
        with col4:
            st.metric(
                "3M ROI",
                f"{results.roi_3months:+.1f}%",
                help="Return on Investment 3 mesi"
            )
        
        # Metriche secondarie
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Profit Impact", f"€{results.profit_impact:,.0f}")
        
        with col2:
            st.metric("Cash Flow Impact", f"€{results.cash_flow_impact:,.0f}")
        
        with col3:
            break_even_text = f"{results.break_even_days} giorni" if results.break_even_days < 365 else "N/A"
            st.metric("Break Even", break_even_text)
        
        # Raccomandazioni
        if results.recommendations:
            st.markdown("### 💡 Raccomandazioni Strategiche")
            for i, rec in enumerate(results.recommendations, 1):
                st.markdown(f"{i}. {rec}")


def create_sample_base_data():
    """Crea dati base per testing simulator."""
    
    # Base forecast
    dates = pd.date_range(start='2025-01-15', periods=90, freq='D')
    base_values = 1000 + np.random.normal(0, 100, 90) + np.sin(np.arange(90) * 2 * np.pi / 30) * 50
    base_forecast = pd.Series(base_values, index=dates, name='base_demand')
    
    # Base metrics
    base_metrics = {
        'unit_price': 150.0,
        'unit_cost': 90.0, 
        'service_level': 92.0,
        'total_value': 125000,
        'coverage_days': 45
    }
    
    return base_forecast, base_metrics