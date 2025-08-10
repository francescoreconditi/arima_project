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
    
    def run(self):
        """Run the main dashboard."""
        st.title("📈 ARIMA Forecaster Dashboard")
        st.markdown("Interactive dashboard for time series forecasting with ARIMA models")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Data Upload", "Data Exploration", "Model Training", "Forecasting", "Model Diagnostics"]
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
    
    def data_upload_page(self):
        """Data upload and preprocessing page."""
        st.header("📁 Data Upload and Preprocessing")
        
        # Data upload section
        st.subheader("Upload Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with timestamp and value columns"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
                
                # Show data info
                st.subheader("Data Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Show first few rows
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                # Column selection
                st.subheader("Column Configuration")
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
                        
                        st.success("Data processed successfully!")
                        st.info("Go to 'Data Exploration' to explore your data.")
                        
                    except Exception as e:
                        st.error(f"Error processing data: {e}")
                        
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        # Sample data option
        st.subheader("Or Use Sample Data")
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
            
            st.success("Sample data generated!")
            st.info("Go to 'Data Exploration' to explore the sample data.")
    
    def data_exploration_page(self):
        """Data exploration and visualization page."""
        st.header("🔍 Data Exploration")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Please upload data first in the 'Data Upload' page.")
            return
        
        data = st.session_state.data
        
        # Data statistics
        st.subheader("Data Statistics")
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
        st.subheader("Time Series Plot")
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
                processed_data = preprocessor.preprocess_pipeline(
                    data.copy(),
                    missing_values=missing_method if missing_method != 'none' else None,
                    outlier_detection=outlier_method if outlier_detection else None,
                    make_stationary=stationary_method if make_stationary else None
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
                
                st.success("Preprocessing completed!")
                
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
        
        # Show stationarity test results
        if st.button("Check Stationarity"):
            from statsmodels.tsa.stattools import adfuller
            
            current_data = st.session_state.get('preprocessed_data', data)
            
            result = adfuller(current_data.dropna())
            
            st.subheader("Augmented Dickey-Fuller Test")
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
        st.header("🤖 Model Training")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Please upload data first in the 'Data Upload' page.")
            return
        
        # Get data
        data = st.session_state.get('preprocessed_data', st.session_state.data)
        
        # Model selection
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select model type",
                ['ARIMA', 'SARIMA', 'Auto-ARIMA', 'Auto-SARIMA']
            )
            
            if model_type in ['ARIMA', 'SARIMA']:
                st.subheader("Manual Parameters")
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
                st.subheader("Auto-selection Parameters")
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
                        st.subheader("Best Model Selection")
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
                        st.subheader("Best Model Selection")
                        st.write(f"Selected order: {selector.best_order}")
                        st.write(f"Selected seasonal order: {selector.best_seasonal_order}")
                        results_df = selector.get_results_summary(10)
                        st.dataframe(results_df)
                    
                    # Store model in session state
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    
                    # Show model information
                    st.success("Model trained successfully!")
                    
                    model_info = model.get_model_info()
                    st.subheader("Model Information")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("AIC", f"{model_info.get('aic', 0):.2f}")
                    with col2:
                        st.metric("BIC", f"{model_info.get('bic', 0):.2f}")
                    with col3:
                        st.metric("HQIC", f"{model_info.get('hqic', 0):.2f}")
                    
                    # Show fitted values vs actual
                    st.subheader("Model Fit")
                    
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
                st.error(f"Model training failed: {e}")
                logger.error(f"Model training failed: {e}")
    
    def forecasting_page(self):
        """Forecasting page."""
        st.header("📊 Forecasting")
        
        if not st.session_state.get('model_trained', False):
            st.warning("Please train a model first in the 'Model Training' page.")
            return
        
        model = st.session_state.model
        data = st.session_state.get('preprocessed_data', st.session_state.data)
        
        # Forecast parameters
        st.subheader("Forecast Configuration")
        
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
                    st.subheader("Forecast Results")
                    
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
                    st.subheader("Forecast Values")
                    
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
                st.error(f"Forecast generation failed: {e}")
                logger.error(f"Forecast generation failed: {e}")
    
    def diagnostics_page(self):
        """Model diagnostics page."""
        st.header("🔧 Model Diagnostics")
        
        if not st.session_state.get('model_trained', False):
            st.warning("Please train a model first in the 'Model Training' page.")
            return
        
        model = st.session_state.model
        data = st.session_state.get('preprocessed_data', st.session_state.data)
        
        st.subheader("Model Summary")
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
            st.subheader("Residuals Analysis")
            
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
                st.subheader("Residuals Statistics")
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
                st.subheader("Statistical Tests")
                
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
                st.error(f"Could not generate residuals analysis: {e}")


def main():
    """Main function to run the dashboard."""
    dashboard = ARIMADashboard()
    dashboard.run()


if __name__ == "__main__":
    main()