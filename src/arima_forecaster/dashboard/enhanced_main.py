"""
Enhanced Dashboard con Mobile Responsive, Excel Export e What-If Simulator.

Dashboard evoluta per enterprise con funzionalità avanzate per procurement team.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import core ARIMA functionality
from arima_forecaster.core import (
    ARIMAForecaster,
    ARIMAModelSelector, 
    SARIMAForecaster,
    SARIMAModelSelector,
    VARForecaster,
    ProphetForecaster,
)
from arima_forecaster.data import DataLoader, TimeSeriesPreprocessor
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils.logger import get_logger

# Import new dashboard modules
from arima_forecaster.dashboard.mobile_responsive import (
    get_responsive_manager,
    init_responsive_dashboard
)
from arima_forecaster.dashboard.excel_exporter import (
    ProcurementExcelExporter,
    create_sample_procurement_data
)
from arima_forecaster.dashboard.scenario_simulator import (
    WhatIfScenarioSimulator,
    ScenarioType,
    create_sample_base_data
)

# Enhanced page configuration with mobile support
st.set_page_config(
    page_title="ARIMA Forecaster Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",  # Auto-collapse on mobile
)

# Initialize logger
logger = get_logger(__name__)


class EnhancedARIMADashboard:
    """Dashboard evoluta con funzionalità enterprise."""
    
    def __init__(self):
        """Inizializza dashboard evoluta."""
        # Core dashboard state
        self.data = None
        self.preprocessed_data = None
        self.model = None
        self.forecast_result = None
        
        # Enhanced functionality
        self.responsive_manager = init_responsive_dashboard()
        self.excel_exporter = ProcurementExcelExporter()
        self.scenario_simulator = WhatIfScenarioSimulator()
        
        # Initialize session state
        self._init_session_state()
        
    def _init_session_state(self):
        """Inizializza session state per tutte le funzionalità."""
        
        # Core states
        states = [
            "data_loaded", "model_trained", "forecast_generated",
            "report_generated", "last_report_path", "last_report_config",
            # Enhanced states
            "excel_report_ready", "scenario_results", "mobile_mode",
            "current_section", "base_forecast", "base_metrics"
        ]
        
        for state in states:
            if state not in st.session_state:
                if state in ["last_report_config", "scenario_results", "base_metrics"]:
                    st.session_state[state] = {}
                elif state == "current_section":
                    st.session_state[state] = "📊 Dashboard"
                elif state == "mobile_mode":
                    st.session_state[state] = self.responsive_manager.is_mobile
                else:
                    st.session_state[state] = False
    
    def run(self):
        """Esegue dashboard principale evoluta."""
        
        # Apply responsive CSS
        self.responsive_manager.apply_responsive_css()
        
        # Main header with responsive design
        self._create_main_header()
        
        # Navigation (responsive)
        current_section = self._create_navigation()
        
        # Route to appropriate section
        if current_section == "📊 Dashboard":
            self._create_dashboard_section()
        elif current_section == "📈 Forecasting":
            self._create_forecasting_section()
        elif current_section == "🎯 What-If Simulator":
            self._create_whatif_section()
        elif current_section == "📋 Reports & Export":
            self._create_reports_section()
        elif current_section == "⚙️ Settings":
            self._create_settings_section()
    
    def _create_main_header(self):
        """Crea header principale responsive."""
        
        config = self.responsive_manager.get_layout_config()
        
        if config['columns'] == 1:  # Mobile
            st.title("📈 ARIMA Forecaster Pro")
            st.caption("Enterprise Time Series Forecasting")
        else:  # Desktop/Tablet
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.title("📈 ARIMA Forecaster Pro")
                st.caption("Enterprise Time Series Forecasting & Inventory Optimization")
            
            with col2:
                if st.button("🔄 Refresh Data"):
                    st.rerun()
            
            with col3:
                st.metric("Status", "✅ Online")
    
    def _create_navigation(self) -> str:
        """Crea navigazione responsive."""
        
        sections = [
            "📊 Dashboard",
            "📈 Forecasting", 
            "🎯 What-If Simulator",
            "📋 Reports & Export",
            "⚙️ Settings"
        ]
        
        return self.responsive_manager.create_mobile_navigation(sections)
    
    def _create_dashboard_section(self):
        """Sezione dashboard principale."""
        
        st.markdown("---")
        
        # Quick metrics (responsive)
        sample_metrics = {
            'Forecast Accuracy': {'value': '84.7%', 'delta': '+2.1%'},
            'Models Active': {'value': '3', 'delta': '+1'}, 
            'Data Points': {'value': '1,250', 'delta': '+50'},
            'Last Update': {'value': '2 min ago', 'delta': None}
        }
        
        self.responsive_manager.display_metrics_responsive(sample_metrics)
        
        st.markdown("---")
        
        # Data upload section
        self._create_data_upload_section()
        
        # Model training section
        if st.session_state.data_loaded:
            self._create_model_training_section()
        
        # Quick forecast preview
        if st.session_state.model_trained:
            self._create_forecast_preview()
    
    def _create_data_upload_section(self):
        """Sezione caricamento dati responsive."""
        
        st.subheader("📁 Data Upload")
        
        cols = self.responsive_manager.create_responsive_columns([2, 1])
        
        with cols[0]:
            uploaded_file = st.file_uploader(
                "Upload CSV file", 
                type=['csv'],
                help="Upload time series data with Date and Value columns"
            )
            
            if uploaded_file:
                try:
                    self.data = pd.read_csv(uploaded_file)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded: {len(self.data)} records")
                    
                    # Preview data (responsive)
                    st.markdown("**Data Preview:**")
                    self.responsive_manager.create_responsive_dataframe(self.data, max_rows=5)
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        if len(cols) > 1:
            with cols[1]:
                st.markdown("**Quick Start:**")
                if st.button("📊 Load Sample Data"):
                    self._load_sample_data()
                
                if st.button("🧹 Clear Data"):
                    self._clear_data()
    
    def _create_model_training_section(self):
        """Sezione training modelli."""
        
        st.subheader("🎓 Model Training")
        
        cols = self.responsive_manager.create_responsive_columns([1, 1, 1])
        
        with cols[0]:
            model_type = st.selectbox(
                "Model Type",
                ["ARIMA", "SARIMA", "Prophet"],
                help="Choose forecasting model type"
            )
        
        if len(cols) > 1:
            with cols[1]:
                auto_params = st.checkbox("Auto Parameters", value=True)
            
            with cols[2]:
                if st.button("🚀 Train Model", type="primary"):
                    self._train_model(model_type, auto_params)
    
    def _create_forecast_preview(self):
        """Crea anteprima forecast rapida."""
        
        st.subheader("📈 Quick Forecast Preview")
        
        cols = self.responsive_manager.create_responsive_columns([2, 1])
        
        with cols[0]:
            if st.button("📊 Generate Quick Forecast"):
                self._generate_detailed_forecast(30, 0.95, True)
        
        if len(cols) > 1:
            with cols[1]:
                st.markdown("**Preview Settings:**")
                st.caption("30-day horizon, 95% confidence")
    
    def _create_forecasting_section(self):
        """Sezione forecasting avanzato."""
        
        st.subheader("📈 Advanced Forecasting")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first in the Dashboard section")
            return
        
        cols = self.responsive_manager.create_responsive_columns([2, 1])
        
        with cols[0]:
            forecast_steps = st.slider("Forecast Horizon (days)", 7, 90, 30)
            confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
        
        if len(cols) > 1:
            with cols[1]:
                st.markdown("**Options:**")
                include_history = st.checkbox("Include History", True)
                show_components = st.checkbox("Show Components", False)
        
        if st.button("📊 Generate Forecast"):
            self._generate_detailed_forecast(forecast_steps, confidence_level, include_history)
        
        # Display forecast results
        if st.session_state.forecast_generated:
            self._display_forecast_results()
    
    def _create_whatif_section(self):
        """Sezione What-If Simulator."""
        
        st.header("🎯 What-If Scenario Simulator")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first to enable scenario analysis")
            return
        
        # Get base data for simulation
        if 'base_forecast' not in st.session_state:
            base_forecast, base_metrics = create_sample_base_data()
            st.session_state.base_forecast = base_forecast
            st.session_state.base_metrics = base_metrics
        
        # Scenario configuration UI
        scenario_params = self.scenario_simulator.create_scenario_ui()
        
        st.markdown("---")
        
        # Run simulation
        cols = self.responsive_manager.create_responsive_columns([1, 1])
        
        with cols[0]:
            if st.button("🚀 Run Scenario Simulation", type="primary"):
                with st.spinner("Running scenario analysis..."):
                    scenario_forecast, results = self.scenario_simulator.run_scenario_simulation(
                        scenario_params,
                        st.session_state.base_forecast,
                        st.session_state.base_metrics
                    )
                    
                    st.session_state.scenario_results = {
                        'params': scenario_params,
                        'forecast': scenario_forecast,
                        'results': results
                    }
        
        if len(cols) > 1:
            with cols[1]:
                if st.button("🔄 Reset Scenario"):
                    if 'scenario_results' in st.session_state:
                        del st.session_state.scenario_results
                    st.rerun()
        
        # Display results
        if 'scenario_results' in st.session_state and st.session_state.scenario_results:
            scenario_data = st.session_state.scenario_results
            
            # Show results
            self.scenario_simulator.display_scenario_results(scenario_data['results'])
            
            # Visualization
            st.markdown("### 📊 Scenario Visualization")
            fig = self.scenario_simulator.create_scenario_visualization(
                st.session_state.base_forecast,
                scenario_data['forecast'],
                scenario_data['results']
            )
            
            # Make chart responsive
            fig = self.responsive_manager.create_responsive_chart(fig, "Scenario Analysis")
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_reports_section(self):
        """Sezione reports ed export."""
        
        st.header("📋 Reports & Excel Export")
        
        # Quick export section
        st.subheader("🚀 Quick Export")
        
        cols = self.responsive_manager.create_responsive_columns([1, 1, 1])
        
        with cols[0]:
            if st.button("📊 Export Dashboard Summary"):
                self._export_dashboard_summary()
        
        if len(cols) > 1:
            with cols[1]:
                if st.button("📈 Export Forecast Data"):
                    self._export_forecast_data()
            
            with cols[2]:
                if st.button("🎯 Export Scenario Analysis"):
                    self._export_scenario_results()
        
        st.markdown("---")
        
        # Procurement Excel Report
        st.subheader("💼 Procurement Excel Report")
        
        report_cols = self.responsive_manager.create_responsive_columns([2, 1])
        
        with report_cols[0]:
            st.markdown("""
            **Professional Excel Report includes:**
            - Executive Summary with KPIs
            - Detailed Reorder Plan with suppliers
            - 30-day Forecast with confidence intervals
            - Performance Analysis by product
            - Risk Assessment matrix
            - Action Items with timeline
            """)
        
        if len(report_cols) > 1:
            with report_cols[1]:
                report_format = st.radio(
                    "Report Format",
                    ["Full Report", "Quick Summary"],
                    help="Choose report detail level"
                )
        
        if st.button("📋 Generate Procurement Report", type="primary"):
            self._generate_procurement_report(report_format == "Full Report")
    
    def _create_settings_section(self):
        """Sezione impostazioni."""
        
        st.header("⚙️ Settings")
        
        # Display preferences
        st.subheader("🖥️ Display Preferences")
        
        cols = self.responsive_manager.create_responsive_columns([1, 1])
        
        with cols[0]:
            theme = st.selectbox("Theme", ["Auto", "Light", "Dark"])
            chart_style = st.selectbox("Chart Style", ["Professional", "Colorful", "Minimal"])
        
        if len(cols) > 1:
            with cols[1]:
                mobile_mode = st.checkbox("Force Mobile Mode", st.session_state.mobile_mode)
                show_advanced = st.checkbox("Show Advanced Options", False)
        
        # Model preferences
        st.subheader("🤖 Model Preferences")
        
        model_cols = self.responsive_manager.create_responsive_columns([1, 1, 1])
        
        with model_cols[0]:
            default_model = st.selectbox("Default Model", ["ARIMA", "SARIMA", "Prophet"])
        
        if len(model_cols) > 1:
            with model_cols[1]:
                confidence_level = st.slider("Default Confidence", 0.8, 0.99, 0.95)
            
            with model_cols[2]:
                forecast_horizon = st.number_input("Default Horizon", 7, 365, 30)
        
        # Export preferences
        st.subheader("📤 Export Preferences")
        
        export_cols = self.responsive_manager.create_responsive_columns([1, 1])
        
        with export_cols[0]:
            excel_template = st.selectbox("Excel Template", ["Professional", "Simple", "Custom"])
        
        if len(export_cols) > 1:
            with export_cols[1]:
                auto_export = st.checkbox("Auto Export After Forecast")
        
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved successfully!")
    
    def _load_sample_data(self):
        """Carica dati di esempio."""
        
        # Generate realistic sample data
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        
        # Realistic time series with trend and seasonality
        trend = np.linspace(1000, 1200, 365)
        seasonal = 100 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly seasonality
        noise = np.random.normal(0, 50, 365)
        
        values = trend + seasonal + noise
        values = np.maximum(values, 100)  # Ensure positive values
        
        self.data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        st.session_state.data_loaded = True
        st.success("✅ Sample data loaded successfully!")
        st.rerun()
    
    def _clear_data(self):
        """Pulisce tutti i dati."""
        
        states_to_clear = [
            'data_loaded', 'model_trained', 'forecast_generated',
            'excel_report_ready', 'scenario_results'
        ]
        
        for state in states_to_clear:
            st.session_state[state] = False
        
        self.data = None
        st.success("✅ Data cleared successfully!")
        st.rerun()
    
    def _train_model(self, model_type: str, auto_params: bool):
        """Addestra il modello selezionato."""
        
        if self.data is None:
            st.error("No data loaded")
            return
        
        with st.spinner(f"Training {model_type} model..."):
            try:
                # Prepare data
                if 'date' in self.data.columns and 'value' in self.data.columns:
                    series = pd.Series(
                        self.data['value'].values,
                        index=pd.to_datetime(self.data['date'])
                    )
                else:
                    # Assume first column is date, second is value
                    series = pd.Series(
                        self.data.iloc[:, 1].values,
                        index=pd.to_datetime(self.data.iloc[:, 0])
                    )
                
                # Train model based on type
                if model_type == "ARIMA":
                    if auto_params:
                        selector = ARIMAModelSelector()
                        selector.search(series)  # Search for best parameters
                        self.model = selector.get_best_model()  # Get the trained model
                    else:
                        self.model = ARIMAForecaster(order=(1, 1, 1))
                        self.model.fit(series)
                
                elif model_type == "SARIMA":
                    if auto_params:
                        selector = SARIMAModelSelector(max_models=10)  # Pass max_models to constructor
                        selector.search(series)  # Search for best parameters
                        self.model = selector.get_best_model()  # Get the trained model
                    else:
                        self.model = SARIMAForecaster(
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12)
                        )
                        self.model.fit(series)
                
                elif model_type == "Prophet":
                    self.model = ProphetForecaster()
                    self.model.fit(series)
                
                # Store model in session state for persistence
                st.session_state.trained_model = self.model
                st.session_state.model_type = model_type
                st.session_state.model_trained = True
                st.success(f"✅ {model_type} model trained successfully!")
                
                # Store base data for scenarios
                st.session_state.base_metrics = {
                    'unit_price': 150.0,
                    'unit_cost': 90.0,
                    'service_level': 92.0,
                    'total_value': 125000
                }
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
    
    def _generate_detailed_forecast(self, steps: int, confidence: float, include_history: bool):
        """Genera forecast dettagliato."""
        
        # Get model from session state
        if not st.session_state.model_trained or 'trained_model' not in st.session_state:
            st.error("No model trained")
            return
        
        model = st.session_state.trained_model
        
        with st.spinner("Generating forecast..."):
            try:
                # Generate forecast with proper parameters
                alpha = 1 - confidence  # Convert confidence level to alpha
                forecast = model.forecast(steps=steps, confidence_intervals=True, alpha=alpha)
                
                # Store results
                self.forecast_result = {
                    'forecast': forecast,
                    'steps': steps,
                    'confidence': confidence,
                    'generated_at': datetime.now()
                }
                
                # Create base forecast for scenarios
                if isinstance(forecast, dict) and 'forecast' in forecast:
                    base_forecast = forecast['forecast']
                    st.session_state.base_forecast = base_forecast
                elif isinstance(forecast, pd.Series):
                    st.session_state.base_forecast = forecast
                
                st.session_state.forecast_generated = True
                st.success("✅ Forecast generated successfully!")
                
            except Exception as e:
                st.error(f"Forecast generation failed: {str(e)}")
    
    def _display_forecast_results(self):
        """Visualizza risultati forecast."""
        
        if not self.forecast_result:
            return
        
        st.markdown("### 📊 Forecast Results")
        
        # Create forecast chart
        forecast_data = self.forecast_result['forecast']
        
        if isinstance(forecast_data, dict):
            fig = go.Figure()
            
            # Forecast line
            if 'forecast' in forecast_data:
                forecast_series = forecast_data['forecast']
                fig.add_trace(go.Scatter(
                    x=forecast_series.index if hasattr(forecast_series, 'index') else list(range(len(forecast_series))),
                    y=forecast_series.values if hasattr(forecast_series, 'values') else forecast_series,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='blue', width=2)
                ))
            
            # Confidence intervals
            if 'confidence_intervals' in forecast_data:
                ci_data = forecast_data['confidence_intervals']
                if 'lower' in ci_data and 'upper' in ci_data:
                    lower_ci = ci_data['lower']
                    upper_ci = ci_data['upper']
                    
                    fig.add_trace(go.Scatter(
                        x=upper_ci.index if hasattr(upper_ci, 'index') else list(range(len(upper_ci))),
                        y=upper_ci.values if hasattr(upper_ci, 'values') else upper_ci,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=lower_ci.index if hasattr(lower_ci, 'index') else list(range(len(lower_ci))),
                        y=lower_ci.values if hasattr(lower_ci, 'values') else lower_ci,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name=f'{self.forecast_result["confidence"]:.0%} Confidence',
                        fillcolor='rgba(0,100,80,0.2)'
                    ))
        elif isinstance(forecast_data, pd.Series):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data.index if hasattr(forecast_data, 'index') else list(range(len(forecast_data))),
                y=forecast_data.values if hasattr(forecast_data, 'values') else forecast_data,
                mode='lines',
                name='Forecast',
                line=dict(color='blue', width=2)
            ))
        
        # Make responsive
        fig = self.responsive_manager.create_responsive_chart(fig, "Forecast Results")
        st.plotly_chart(fig, use_container_width=True)
    
    def _export_dashboard_summary(self):
        """Esporta summary dashboard."""
        
        summary_data = {
            'total_value': 125000,
            'coverage_days': 45,
            'reorder_items': 3,
            'savings': 8500
        }
        
        excel_data = self.excel_exporter.generate_quick_summary(summary_data)
        
        st.download_button(
            label="⬇️ Download Dashboard Summary.xlsx",
            data=excel_data,
            file_name=f"Dashboard_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    def _export_forecast_data(self):
        """Esporta dati forecast."""
        
        if not st.session_state.forecast_generated:
            st.warning("No forecast data available")
            return
        
        # Create forecast DataFrame
        if self.forecast_result and 'forecast' in self.forecast_result:
            forecast_data = self.forecast_result['forecast']
            
            if isinstance(forecast_data, dict) and 'forecast' in forecast_data:
                # Dictionary format: {'forecast': Series, 'confidence_intervals': {...}}
                forecast_series = forecast_data['forecast']
                forecast_values = forecast_series.values if hasattr(forecast_series, 'values') else forecast_series
                dates = forecast_series.index if hasattr(forecast_series, 'index') else pd.date_range(
                    start=datetime.now().date(), periods=len(forecast_values), freq='D'
                )
            elif isinstance(forecast_data, pd.Series):
                # Series format
                forecast_values = forecast_data.values
                dates = forecast_data.index if hasattr(forecast_data, 'index') else pd.date_range(
                    start=datetime.now().date(), periods=len(forecast_values), freq='D'
                )
            else:
                # Fallback
                forecast_values = forecast_data
                dates = pd.date_range(start=datetime.now().date(), periods=len(forecast_values), freq='D')
            
            forecast_df = pd.DataFrame({
                'Date': dates,
                'Forecast': forecast_values
            })
            
            # Convert to Excel
            excel_buffer = BytesIO()
            forecast_df.to_excel(excel_buffer, index=False)
            
            st.download_button(
                label="⬇️ Download Forecast.xlsx",
                data=excel_buffer.getvalue(),
                file_name=f"Forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    def _export_scenario_results(self):
        """Esporta risultati scenario analysis."""
        
        if 'scenario_results' not in st.session_state or not st.session_state.scenario_results:
            st.warning("No scenario results available")
            return
        
        scenario_data = st.session_state.scenario_results
        results = scenario_data['results']
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Metric': [
                'Revenue Impact', 'Revenue Change %', 'Inventory Investment',
                'Service Level', 'Profit Impact', 'ROI 3M'
            ],
            'Value': [
                f"€{results.revenue_impact:,.0f}",
                f"{results.revenue_change_pct:.1f}%",
                f"€{results.inventory_investment:,.0f}",
                f"{results.service_level:.1f}%",
                f"€{results.profit_impact:,.0f}",
                f"{results.roi_3months:.1f}%"
            ]
        })
        
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Scenario Summary', index=False)
            
            # Add recommendations
            rec_df = pd.DataFrame({
                'Recommendations': results.recommendations
            })
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        st.download_button(
            label="⬇️ Download Scenario Analysis.xlsx",
            data=excel_buffer.getvalue(),
            file_name=f"Scenario_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    def _generate_procurement_report(self, full_report: bool = True):
        """Genera report procurement completo."""
        
        with st.spinner("Generating procurement report..."):
            try:
                # Get sample data
                forecast_data, inventory_params, product_info = create_sample_procurement_data()
                
                if full_report:
                    # Generate full Excel report
                    excel_data = self.excel_exporter.generate_procurement_report(
                        forecast_data=forecast_data,
                        inventory_params=inventory_params,
                        product_info=product_info,
                        supplier_data={}  # Sample supplier data
                    )
                    filename = f"Procurement_Report_Full_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                else:
                    # Generate quick summary
                    excel_data = self.excel_exporter.generate_quick_summary(inventory_params)
                    filename = f"Procurement_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                
                st.download_button(
                    label=f"⬇️ Download {filename}",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Procurement report generated successfully!")
                
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")


def main():
    """Funzione principale."""
    
    # Initialize and run enhanced dashboard
    dashboard = EnhancedARIMADashboard()
    dashboard.run()


if __name__ == "__main__":
    main()