"""
Main Streamlit dashboard for ARIMA forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arima_forecaster.core import ARIMAForecaster, SARIMAForecaster, VARForecaster
from arima_forecaster.core import ARIMAModelSelector, SARIMAModelSelector
from arima_forecaster.data import DataLoader, TimeSeriesPreprocessor
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils.logger import get_logger

# Page configuration
st.set_page_config(
    page_title="ARIMA Forecaster Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_logger(__name__)


class ARIMADashboard:
    """Main dashboard class."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.data = None
        self.preprocessed_data = None
        self.model = None
        self.forecast_result = None
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'forecast_generated' not in st.session_state:
            st.session_state.forecast_generated = False
        if 'report_generated' not in st.session_state:
            st.session_state.report_generated = False
        if 'last_report_path' not in st.session_state:
            st.session_state.last_report_path = None
        if 'last_report_config' not in st.session_state:
            st.session_state.last_report_config = {}
    
    def run(self):
        """Run the main dashboard."""
        st.title("📈 Dashboard ARIMA Forecaster")
        st.markdown("Interactive dashboard for time series forecasting with ARIMA models")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Data Upload", "Data Exploration", "Model Training", "Forecasting", "Model Diagnostics", "Report Generation"]
        )
        
        # Route to appropriate page
        if page == "Data Upload":
            self.data_upload_page()
        elif page == "Data Exploration":
            self.data_exploration_page()
        elif page == "Model Training":
            self.model_training_page()
        elif page == "Forecasting":
            self.forecasting_page()
        elif page == "Model Diagnostics":
            self.diagnostics_page()
        elif page == "Report Generation":
            self.report_generation_page()
    
    def data_upload_page(self):
        """Data upload and preprocessing page."""
        st.header("📁 Caricamento Dati e Preprocessing")
        
        # Data upload section
        st.subheader("Caricamento Dati")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with timestamp and value columns"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.success("Dati caricati con successo!")
                
                # Show data info
                st.subheader("Panoramica Dati")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Show first few rows
                st.subheader("Anteprima Dati")
                st.dataframe(df.head(10))
                
                # Column selection
                st.subheader("Configurazione Colonne")
                timestamp_col = st.selectbox("Select timestamp column", df.columns)
                value_col = st.selectbox("Select value column", df.columns)
                
                if st.button("Process Data"):
                    try:
                        # Create time series
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                        series = pd.Series(
                            df[value_col].values,
                            index=df[timestamp_col],
                            name=value_col
                        )
                        
                        # Store in session state
                        st.session_state.data = series
                        st.session_state.data_loaded = True
                        
                        st.success("Dati elaborati con successo!")
                        st.info("Vai su 'Data Exploration' per esplorare i tuoi dati.")
                        
                    except Exception as e:
                        st.error(f"Errore nell'elaborazione dati: {e}")
                        
            except Exception as e:
                st.error(f"Errore nel caricamento file: {e}")
        
        # Sample data option
        st.subheader("Oppure Usa Dati di Esempio")
        if st.button("Generate Sample Data"):
            # Generate sample time series data
            dates = pd.date_range('2020-01-01', periods=365, freq='D')
            np.random.seed(42)
            trend = np.linspace(100, 150, 365)
            seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)
            noise = np.random.normal(0, 5, 365)
            values = trend + seasonal + noise
            
            series = pd.Series(values, index=dates, name='sample_data')
            
            st.session_state.data = series
            st.session_state.data_loaded = True
            
            st.success("Dati di esempio generati!")
            st.info("Vai su 'Data Exploration' per esplorare i dati di esempio.")
    
    def data_exploration_page(self):
        """Data exploration and visualization page."""
        st.header("🔍 Esplorazione Dati")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Per favore carica prima i dati nella pagina 'Data Upload'.")
            return
        
        data = st.session_state.data
        
        # Data statistics
        st.subheader("Statistiche Dati")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Count", len(data))
        with col2:
            st.metric("Mean", f"{data.mean():.2f}")
        with col3:
            st.metric("Std", f"{data.std():.2f}")
        with col4:
            st.metric("Missing", data.isnull().sum())
        
        # Time series plot
        st.subheader("Grafico Serie Temporale")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=data.name or 'Values',
            line=dict(color='blue', width=1)
        ))
        fig.update_layout(
            title="Original Time Series",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Preprocessing options
        st.subheader("Preprocessing")
        preprocessor = TimeSeriesPreprocessor()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values handling
            st.write("**Missing Values**")
            missing_method = st.selectbox(
                "Handle missing values",
                ['none', 'interpolate', 'forward_fill', 'backward_fill', 'drop']
            )
            
            # Outlier detection
            st.write("**Outliers**")
            outlier_detection = st.checkbox("Remove outliers")
            outlier_method = st.selectbox(
                "Outlier method",
                ['iqr', 'zscore', 'modified_zscore'],
                disabled=not outlier_detection
            )
        
        with col2:
            # Stationarity
            st.write("**Stationarity**")
            make_stationary = st.checkbox("Make stationary")
            stationary_method = st.selectbox(
                "Stationary method",
                ['difference', 'log_difference'],
                disabled=not make_stationary
            )
        
        if st.button("Apply Preprocessing"):
            try:
                processed_data, metadata = preprocessor.preprocess_pipeline(
                    data.copy(),
                    handle_missing=missing_method != 'none',
                    missing_method=missing_method if missing_method != 'none' else 'interpolate',
                    remove_outliers_flag=outlier_detection,
                    outlier_method=outlier_method,
                    make_stationary_flag=make_stationary,
                    stationarity_method=stationary_method
                )
                
                st.session_state.preprocessed_data = processed_data
                
                # Show comparison
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Original Data", "Preprocessed Data"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(x=data.index, y=data.values, name="Original"),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=processed_data.index, y=processed_data.values, name="Preprocessed"),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("Preprocessing completato!")
                
            except Exception as e:
                st.error(f"Preprocessing fallito: {e}")
        
        # Show stationarity test results
        if st.button("Check Stationarity"):
            from statsmodels.tsa.stattools import adfuller
            
            current_data = st.session_state.get('preprocessed_data', data)
            
            result = adfuller(current_data.dropna())
            
            st.subheader("Test di Dickey-Fuller Aumentato")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ADF Statistic", f"{result[0]:.4f}")
                st.metric("P-value", f"{result[1]:.4f}")
            with col2:
                st.metric("Critical Value (5%)", f"{result[4]['5%']:.4f}")
                is_stationary = result[1] < 0.05
                st.metric("Stationary", "Yes" if is_stationary else "No")
    
    def model_training_page(self):
        """Model training page."""
        st.header("🤖 Addestramento Modello")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Per favore carica prima i dati nella pagina 'Data Upload'.")
            return
        
        # Get data
        data = st.session_state.get('preprocessed_data', st.session_state.data)
        
        # Model selection
        st.subheader("Configurazione Modello")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select model type",
                ['ARIMA', 'SARIMA', 'Auto-ARIMA', 'Auto-SARIMA']
            )
            
            if model_type in ['ARIMA', 'SARIMA']:
                st.subheader("Parametri Manuali")
                p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
                d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
                q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
                
                if model_type == 'SARIMA':
                    st.write("**Seasonal Parameters**")
                    P = st.number_input("Seasonal AR (P)", min_value=0, max_value=2, value=1)
                    D = st.number_input("Seasonal Diff (D)", min_value=0, max_value=1, value=1)
                    Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=2, value=1)
                    s = st.number_input("Seasonal period (s)", min_value=2, max_value=365, value=12)
        
        with col2:
            if model_type.startswith('Auto'):
                st.subheader("Parametri Auto-selezione")
                max_p = st.number_input("Max AR order", min_value=0, max_value=5, value=3)
                max_d = st.number_input("Max Differencing", min_value=0, max_value=2, value=2)
                max_q = st.number_input("Max MA order", min_value=0, max_value=5, value=3)
                
                if model_type == 'Auto-SARIMA':
                    max_P = st.number_input("Max Seasonal AR", min_value=0, max_value=2, value=1)
                    max_D = st.number_input("Max Seasonal Diff", min_value=0, max_value=1, value=1)
                    max_Q = st.number_input("Max Seasonal MA", min_value=0, max_value=2, value=1)
                    seasonal_periods = st.multiselect(
                        "Seasonal periods to test",
                        [4, 7, 12, 24, 52, 365],
                        default=[12]
                    )
                
                ic = st.selectbox("Information Criterion", ['aic', 'bic', 'hqic'])
                max_models = st.number_input("Max models to test", min_value=10, max_value=200, value=50)
        
        # Train model
        if st.button("Train Model", type="primary"):
            try:
                with st.spinner("Training model..."):
                    
                    if model_type == 'ARIMA':
                        model = ARIMAForecaster(order=(p, d, q))
                        model.fit(data)
                        
                    elif model_type == 'SARIMA':
                        model = SARIMAForecaster(
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s)
                        )
                        model.fit(data)
                        
                    elif model_type == 'Auto-ARIMA':
                        selector = ARIMAModelSelector(
                            p_range=(0, max_p),
                            d_range=(0, max_d),
                            q_range=(0, max_q),
                            information_criterion=ic,
                            max_models=max_models
                        )
                        selector.search(data, verbose=False)
                        model = selector.get_best_model()
                        
                        # Show selection results
                        st.subheader("Selezione Miglior Modello")
                        st.write(f"Selected order: {selector.best_order}")
                        results_df = selector.get_results_summary(10)
                        st.dataframe(results_df)
                        
                    elif model_type == 'Auto-SARIMA':
                        selector = SARIMAModelSelector(
                            p_range=(0, max_p),
                            d_range=(0, max_d),
                            q_range=(0, max_q),
                            P_range=(0, max_P),
                            D_range=(0, max_D),
                            Q_range=(0, max_Q),
                            seasonal_periods=seasonal_periods,
                            information_criterion=ic,
                            max_models=max_models
                        )
                        selector.search(data, verbose=False)
                        model = selector.get_best_model()
                        
                        # Show selection results
                        st.subheader("Selezione Miglior Modello")
                        st.write(f"Selected order: {selector.best_order}")
                        st.write(f"Selected seasonal order: {selector.best_seasonal_order}")
                        results_df = selector.get_results_summary(10)
                        st.dataframe(results_df)
                    
                    # Store model in session state
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    
                    # Show model information
                    st.success("Modello addestrato con successo!")
                    
                    model_info = model.get_model_info()
                    st.subheader("Informazioni Modello")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("AIC", f"{model_info.get('aic', 0):.2f}")
                    with col2:
                        st.metric("BIC", f"{model_info.get('bic', 0):.2f}")
                    with col3:
                        st.metric("HQIC", f"{model_info.get('hqic', 0):.2f}")
                    
                    # Show fitted values vs actual
                    st.subheader("Adattamento Modello")
                    
                    fitted_values = model.predict()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data.values,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=fitted_values.index,
                        y=fitted_values.values,
                        mode='lines',
                        name='Fitted',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Actual vs Fitted Values",
                        xaxis_title="Time",
                        yaxis_title="Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Addestramento modello fallito: {e}")
                logger.error(f"Model training failed: {e}")
    
    def forecasting_page(self):
        """Forecasting page."""
        st.header("📊 Forecasting")
        
        if not st.session_state.get('model_trained', False):
            st.warning("Per favore addestra prima un modello nella pagina 'Model Training'.")
            return
        
        model = st.session_state.model
        data = st.session_state.get('preprocessed_data', st.session_state.data)
        
        # Forecast parameters
        st.subheader("Configurazione Forecast")
        
        col1, col2 = st.columns(2)
        with col1:
            forecast_steps = st.number_input(
                "Number of forecast steps",
                min_value=1,
                max_value=100,
                value=10
            )
            
        with col2:
            confidence_level = st.slider(
                "Confidence level",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
        
        show_intervals = st.checkbox("Show confidence intervals", value=True)
        
        # Generate forecast
        if st.button("Generate Forecast", type="primary"):
            try:
                with st.spinner("Generating forecast..."):
                    
                    if show_intervals:
                        forecast, conf_int = model.forecast(
                            steps=forecast_steps,
                            alpha=1-confidence_level,
                            return_conf_int=True
                        )
                    else:
                        forecast = model.forecast(
                            steps=forecast_steps,
                            confidence_intervals=False
                        )
                        conf_int = None
                    
                    st.session_state.forecast_result = {
                        'forecast': forecast,
                        'conf_int': conf_int,
                        'confidence_level': confidence_level
                    }
                    st.session_state.forecast_generated = True
                    
                    # Plot forecast
                    st.subheader("Risultati Forecast")
                    
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data.index[-50:],  # Show last 50 points
                        y=data.values[-50:],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast.values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Confidence intervals
                    if conf_int is not None:
                        fig.add_trace(go.Scatter(
                            x=forecast.index,
                            y=conf_int.iloc[:, 1],
                            mode='lines',
                            name=f'Upper {confidence_level:.1%}',
                            line=dict(color='rgba(255,0,0,0.3)', width=0),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast.index,
                            y=conf_int.iloc[:, 0],
                            mode='lines',
                            name=f'Confidence Interval',
                            line=dict(color='rgba(255,0,0,0.3)', width=0),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.2)'
                        ))
                    
                    fig.update_layout(
                        title="Time Series Forecast",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show forecast values
                    st.subheader("Valori Forecast")
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast.index,
                        'Forecast': forecast.values
                    })
                    
                    if conf_int is not None:
                        forecast_df['Lower Bound'] = conf_int.iloc[:, 0].values
                        forecast_df['Upper Bound'] = conf_int.iloc[:, 1].values
                    
                    st.dataframe(forecast_df)
                    
                    # Download forecast
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv,
                        file_name="forecast.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Generazione forecast fallita: {e}")
                logger.error(f"Forecast generation failed: {e}")
    
    def diagnostics_page(self):
        """Model diagnostics page."""
        st.header("🔧 Diagnostica Modello")
        
        if not st.session_state.get('model_trained', False):
            st.warning("Per favore addestra prima un modello nella pagina 'Model Training'.")
            return
        
        model = st.session_state.model
        data = st.session_state.get('preprocessed_data', st.session_state.data)
        
        st.subheader("Sommario Modello")
        model_info = model.get_model_info()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AIC", f"{model_info.get('aic', 0):.2f}")
        with col2:
            st.metric("BIC", f"{model_info.get('bic', 0):.2f}")
        with col3:
            st.metric("HQIC", f"{model_info.get('hqic', 0):.2f}")
        with col4:
            st.metric("Log Likelihood", f"{model_info.get('llf', 0):.2f}")
        
        # Residuals analysis
        if hasattr(model, 'fitted_model'):
            st.subheader("Analisi Residui")
            
            try:
                residuals = model.fitted_model.resid
                
                # Residuals plot
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Residuals vs Time", "Residuals Distribution", 
                                   "Q-Q Plot", "ACF of Residuals"),
                    vertical_spacing=0.1
                )
                
                # Residuals vs time
                fig.add_trace(
                    go.Scatter(x=residuals.index, y=residuals.values, mode='lines', name="Residuals"),
                    row=1, col=1
                )
                
                # Residuals histogram
                fig.add_trace(
                    go.Histogram(x=residuals.values, name="Distribution", nbinsx=30),
                    row=1, col=2
                )
                
                # Q-Q plot (simplified)
                from scipy import stats
                qq_data = stats.probplot(residuals.dropna(), dist="norm")
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name="Q-Q"),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[1][0] + qq_data[1][1] * qq_data[0][0],
                              mode='lines', name="Q-Q Line"),
                    row=2, col=1
                )
                
                # ACF of residuals
                try:
                    from statsmodels.tsa.stattools import acf
                    acf_vals = acf(residuals.dropna(), nlags=20)
                    fig.add_trace(
                        go.Bar(x=list(range(21)), y=acf_vals, name="ACF"),
                        row=2, col=2
                    )
                except Exception:
                    pass
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Residuals statistics
                st.subheader("Statistiche Residui")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{residuals.mean():.4f}")
                with col2:
                    st.metric("Std Dev", f"{residuals.std():.4f}")
                with col3:
                    st.metric("Skewness", f"{residuals.skew():.4f}")
                with col4:
                    st.metric("Kurtosis", f"{residuals.kurtosis():.4f}")
                
                # Statistical tests
                st.subheader("Test Statistici")
                
                # Normality test
                try:
                    stat, p_value = stats.jarque_bera(residuals.dropna())
                    st.write(f"**Jarque-Bera Normality Test:**")
                    st.write(f"- Test Statistic: {stat:.4f}")
                    st.write(f"- P-value: {p_value:.4f}")
                    st.write(f"- Residuals are {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)")
                except Exception as e:
                    st.write("Could not perform normality test")
                
                # Ljung-Box test for autocorrelation
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_result = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
                    st.write(f"**Ljung-Box Test (Residual Autocorrelation):**")
                    st.write(f"- P-value (lag 10): {lb_result['lb_pvalue'].iloc[9]:.4f}")
                    st.write(f"- No autocorrelation: {'Yes' if lb_result['lb_pvalue'].iloc[9] > 0.05 else 'No'} (α=0.05)")
                except Exception as e:
                    st.write("Could not perform Ljung-Box test")
                    
            except Exception as e:
                st.error(f"Impossibile generare analisi residui: {e}")
    
    def report_generation_page(self):
        """Report generation page with proper state management."""
        st.header("📄 Generazione Report")
        
        if not st.session_state.get('model_trained', False):
            st.warning("Per favore addestra prima un modello nella pagina 'Model Training'.")
            return
        
        st.markdown("""
        Generate comprehensive reports for your trained model including:
        - Model summary and configuration
        - Performance metrics and diagnostics  
        - Forecast visualizations and analysis
        - Statistical tests and validation
        """)
        
        # Check if report was already generated
        if st.session_state.get('report_generated', False) and st.session_state.get('last_report_path'):
            self._show_generated_report()
        else:
            self._show_report_configuration()
    
    def _show_report_configuration(self):
        """Show report configuration form."""
        st.subheader("Configurazione Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Report Title", 
                value="Time Series Analysis Report",
                help="Custom title for your report"
            )
            
            output_filename = st.text_input(
                "Output Filename",
                value="",
                help="Leave empty for auto-generated filename"
            )
            
            format_type = st.selectbox(
                "Output Format",
                ["html", "pdf", "docx"],
                index=0,
                help="Choose the output format for your report"
            )
        
        with col2:
            include_diagnostics = st.checkbox(
                "Include Model Diagnostics", 
                value=True,
                help="Include residual analysis, statistical tests, and model validation"
            )
            
            include_forecast = st.checkbox(
                "Include Forecast Analysis", 
                value=True,
                help="Include forecast visualizations and confidence intervals"
            )
            
            if include_forecast:
                forecast_steps = st.number_input(
                    "Forecast Steps",
                    min_value=1,
                    max_value=100,
                    value=12,
                    help="Number of forecast steps to include in the report"
                )
            else:
                forecast_steps = 12
        
        # Additional options
        st.subheader("Opzioni Avanzate")
        
        col1, col2 = st.columns(2)
        with col1:
            if format_type == "pdf":
                st.info("📋 L'export PDF richiede l'installazione di LaTeX")
            elif format_type == "docx":
                st.info("📄 L'export DOCX richiede pandoc")
            else:
                st.info("🌐 Il formato HTML è raccomandato per la visualizzazione web")
        
        with col2:
            auto_open = st.checkbox(
                "Auto-open report after generation",
                value=True,
                help="Automatically open the report in your default browser/application"
            )
        
        # Generate report button
        st.subheader("Genera Report")
        
        if st.button("🚀 Generate Report", type="primary", use_container_width=True):
            self._generate_report(
                report_title, output_filename, format_type, 
                include_diagnostics, include_forecast, forecast_steps, auto_open
            )
    
    def _generate_report(self, report_title, output_filename, format_type, 
                        include_diagnostics, include_forecast, forecast_steps, auto_open):
        """Generate the report and handle the process."""
        try:
            with st.spinner("Generating comprehensive report... This may take a few minutes."):
                
                # Use the model's built-in report generation
                model = st.session_state.model
                
                # Set output filename if not provided
                if not output_filename.strip():
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"dashboard_report_{timestamp}"
                
                # Generate plots data if we have forecast data
                plots_data = None
                if st.session_state.get('forecast_generated', False) and include_forecast:
                    plots_data = self._create_forecast_plot(output_filename)
                
                # Generate the report
                report_path = model.generate_report(
                    plots_data=plots_data,
                    report_title=report_title,
                    output_filename=output_filename,
                    format_type=format_type,
                    include_diagnostics=include_diagnostics,
                    include_forecast=include_forecast,
                    forecast_steps=forecast_steps
                )
                
                # Store in session state
                st.session_state.report_generated = True
                st.session_state.last_report_path = report_path
                st.session_state.last_report_config = {
                    'format_type': format_type,
                    'auto_open': auto_open,
                    'report_title': report_title
                }
                
                # Auto-open if requested
                if auto_open:
                    self._auto_open_report(report_path)
                
                st.success("🎉 Report generato con successo!")
                st.rerun()  # Refresh to show the generated report section
                
        except Exception as e:
            st.error(f"❌ Generazione report fallita: {e}")
            logger.error(f"Report generation failed: {e}")
            self._show_troubleshooting(format_type)
    
    def _create_forecast_plot(self, output_filename):
        """Create forecast plot for report."""
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path
            
            # Get data
            data = st.session_state.get('preprocessed_data', st.session_state.data)
            forecast_result = st.session_state.get('forecast_result', {})
            
            if forecast_result:
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical data (last 50 points)
                ax.plot(data.index[-50:], data.values[-50:], 
                       label='Historical', color='blue', linewidth=1.5)
                
                # Plot forecast
                forecast = forecast_result['forecast']
                ax.plot(forecast.index, forecast.values, 
                       label='Forecast', color='red', linewidth=2, marker='o', markersize=4)
                
                # Plot confidence intervals if available
                conf_int = forecast_result.get('conf_int')
                if conf_int is not None:
                    confidence_level = forecast_result.get('confidence_level', 0.95)
                    ax.fill_between(forecast.index, 
                                   conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                                   alpha=0.3, color='red', 
                                   label=f'{confidence_level:.0%} Confidence Interval')
                
                ax.set_title('Time Series Forecast')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Save plot
                plots_dir = Path("outputs/plots")
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plots_dir / f"{output_filename}_dashboard_plot.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return {'main_plot': str(plot_path)}
                
        except Exception as e:
            st.warning(f"Impossibile creare grafico per report: {e}")
            return None
    
    def _auto_open_report(self, report_path):
        """Attempt to auto-open the report."""
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                subprocess.run(["start", str(report_path)], shell=True, check=False)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(report_path)], check=False)
            else:  # Linux
                subprocess.run(["xdg-open", str(report_path)], check=False)
            
            st.success("✅ Report aperto automaticamente!")
            
        except Exception as e:
            st.info(f"💡 Apertura automatica fallita, ma report generato: {report_path}")
    
    def _show_generated_report(self):
        """Show the already generated report with actions."""
        st.success("🎉 Report generato con successo!")
        
        report_path = st.session_state.last_report_path
        config = st.session_state.last_report_config
        format_type = config.get('format_type', 'html')
        
        # Show report info
        st.subheader("Informazioni Report")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Format", format_type.upper())
        with col2:
            if Path(report_path).exists():
                report_size = Path(report_path).stat().st_size / (1024 * 1024)
                st.metric("File Size", f"{report_size:.1f} MB")
            else:
                st.metric("File Size", "N/A")
        with col3:
            st.metric("Location", "outputs/reports/")
        
        # Display report path
        st.info(f"📁 Report salvato in: `{report_path}`")
        
        # Action buttons
        if Path(report_path).exists():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with open(report_path, "rb") as file:
                    st.download_button(
                        label=f"📥 Download {format_type.upper()} Report",
                        data=file.read(),
                        file_name=Path(report_path).name,
                        mime=f"application/{format_type}" if format_type != "html" else "text/html",
                        use_container_width=True
                    )
            
            with col2:
                if st.button(f"🌐 Open {format_type.upper()} Report", use_container_width=True):
                    self._auto_open_report(report_path)
            
            with col3:
                if st.button("🔄 Generate New Report", use_container_width=True):
                    # Reset state to show configuration again
                    st.session_state.report_generated = False
                    st.session_state.last_report_path = None
                    st.rerun()
            
            # Preview for HTML
            if format_type == "html":
                self._show_html_preview(report_path)
        
        self._show_tips(format_type)
    
    def _show_html_preview(self, report_path):
        """Show HTML preview with proper state management."""
        st.subheader("Anteprima Report")
        
        # Use a unique key for the radio button to maintain state
        preview_key = f"preview_option_{hash(report_path)}"
        preview_option = st.radio(
            "Preview type:",
            ["Summary Only", "Full Report"],
            index=0,
            key=preview_key,
            help="Summary shows key info, Full Report shows complete HTML content"
        )
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            if preview_option == "Summary Only":
                self._show_html_summary(html_content, report_path)
            else:
                self._show_full_html_preview(html_content)
                
        except Exception as e:
            st.error(f"Impossibile mostrare anteprima: {e}")
            st.markdown(f"""
            **Alternative options:**
            - Use the **Download** button to save the report
            - Use the **Open** button to view in your browser
            - File location: `{report_path}`
            """)
    
    def _show_html_summary(self, html_content, report_path):
        """Show HTML summary information."""
        st.write("**Report Summary:**")
        
        # Extract key information from HTML
        import re
        
        # Try to extract title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        if title_match:
            st.write(f"📋 **Title:** {title_match.group(1)}")
        
        # Check for presence of sections
        sections = []
        if "Model Information" in html_content or "Informazioni Modello" in html_content:
            sections.append("✅ Model Information")
        if "Forecast" in html_content or "forecast" in html_content:
            sections.append("✅ Forecast Analysis") 
        if "Diagnostics" in html_content or "Diagnostica" in html_content:
            sections.append("✅ Model Diagnostics")
        if "table" in html_content.lower():
            sections.append("✅ Data Tables")
        if "img" in html_content.lower() or "plot" in html_content.lower():
            sections.append("✅ Visualizations")
        
        if sections:
            st.write("**Report Sections:**")
            for section in sections:
                st.write(f"- {section}")
        
        # File info
        file_size = Path(report_path).stat().st_size / 1024
        st.write(f"📁 **File Size:** {file_size:.1f} KB")
        st.write(f"📂 **Location:** {report_path}")
        
        st.info("💡 Clicca 'Full Report' sopra per vedere l'anteprima HTML completa")
    
    def _show_full_html_preview(self, html_content):
        """Show full HTML preview."""
        st.warning("⚠️ Report grandi potrebbero richiedere tempo per caricare. Se l'anteprima è lenta, usa i pulsanti download/open.")
        
        # Limit HTML size for performance
        max_size = 1024 * 1024 * 2  # 2MB limit
        if len(html_content) > max_size:
            st.warning("Report molto grande. Mostrando anteprima troncata...")
            html_content = html_content[:max_size] + "\n<!-- Content truncated for preview -->"
        
        st.components.v1.html(
            html_content,
            height=800,
            scrolling=True
        )
    
    def _show_tips(self, format_type):
        """Show tips based on format type."""
        st.subheader("💡 Suggerimenti")
        if format_type == "html":
            st.markdown("""
            - HTML reports are interactive and work best for web viewing
            - Share the HTML file for collaborative review
            - Use browser print function to create PDF if needed
            """)
        elif format_type == "pdf":
            st.markdown("""
            - PDF reports are perfect for presentations and formal documentation
            - Requires LaTeX installation for full functionality
            - Best for printing and archival purposes
            """)
        elif format_type == "docx":
            st.markdown("""
            - DOCX reports can be edited in Microsoft Word
            - Perfect for collaborative editing and custom formatting
            - Requires pandoc for advanced features
            """)
    
    def _show_troubleshooting(self, format_type):
        """Show troubleshooting information."""
        st.subheader("🔧 Risoluzione Problemi")
        if format_type == "pdf":
            st.markdown("""
            **PDF generation issues:**
            - Install LaTeX: `winget install MiKTeX.MiKTeX` (Windows)
            - Or try HTML format instead
            """)
        elif format_type == "docx":
            st.markdown("""
            **DOCX generation issues:**
            - Install pandoc: `winget install --id JohnMacFarlane.Pandoc`
            - Or try HTML format instead
            """)
        else:
            st.markdown("""
            **General issues:**
            - Ensure model is properly trained
            - Check that forecast data is available if including forecasts
            - Try with minimal options (HTML format, basic settings)
            """)
        
        # Show debug info
        with st.expander("🔍 Debug Information"):
            st.write("**Session State:**")
            debug_info = {
                "Data Loaded": st.session_state.get('data_loaded', False),
                "Model Trained": st.session_state.get('model_trained', False), 
                "Forecast Generated": st.session_state.get('forecast_generated', False),
                "Report Generated": st.session_state.get('report_generated', False)
            }
            
            if st.session_state.get('model_trained', False):
                model = st.session_state.model
                model_info = model.get_model_info()
                debug_info.update({
                    "Model Type": type(model).__name__,
                    "Model Order": model_info.get('order', 'N/A'),
                    "AIC": f"{model_info.get('aic', 0):.2f}",
                    "BIC": f"{model_info.get('bic', 0):.2f}"
                })
            
            st.json(debug_info)


def main():
    """Main function to run the dashboard."""
    dashboard = ARIMADashboard()
    dashboard.run()


if __name__ == "__main__":
    main()