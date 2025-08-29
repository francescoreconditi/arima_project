"""
Main Streamlit dashboard for ARIMA forecasting.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arima_forecaster.core import (
    ARIMAForecaster,
    ARIMAModelSelector,
    SARIMAForecaster,
    SARIMAModelSelector,
    SARIMAXAutoSelector,
    SARIMAXForecaster,
    SARIMAXModelSelector,
    VARForecaster,
    ProphetForecaster,
    ProphetModelSelector,
)
from arima_forecaster.data import DataLoader, TimeSeriesPreprocessor
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils.exog_diagnostics import ExogDiagnostics
from arima_forecaster.utils.logger import get_logger
from arima_forecaster.utils.preprocessing import (
    ExogenousPreprocessor,
    analyze_feature_relationships,
)

# Page configuration
st.set_page_config(
    page_title="ARIMA Forecaster Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
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
        if "data_loaded" not in st.session_state:
            st.session_state.data_loaded = False
        if "model_trained" not in st.session_state:
            st.session_state.model_trained = False
        if "forecast_generated" not in st.session_state:
            st.session_state.forecast_generated = False
        if "report_generated" not in st.session_state:
            st.session_state.report_generated = False
        if "last_report_path" not in st.session_state:
            st.session_state.last_report_path = None
        if "last_report_config" not in st.session_state:
            st.session_state.last_report_config = {}

    def run(self):
        """Run the main dashboard."""
        st.title("📈 Dashboard ARIMA Forecaster")
        st.markdown(
            "Interactive dashboard for time series forecasting with ARIMA, SARIMA, SARIMAX, and VAR models"
        )

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            [
                "Data Upload",
                "Data Exploration",
                "Advanced Exog Analysis",
                "Model Training",
                "Forecasting",
                "Model Diagnostics",
                "Report Generation",
            ],
        )

        # Route to appropriate page
        if page == "Data Upload":
            self.data_upload_page()
        elif page == "Data Exploration":
            self.data_exploration_page()
        elif page == "Advanced Exog Analysis":
            self.advanced_exog_analysis_page()
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
            type=["csv"],
            help="Upload a CSV file with timestamp and value columns",
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

                # Additional columns for SARIMAX
                other_columns = [col for col in df.columns if col not in [timestamp_col, value_col]]
                if other_columns:
                    st.info(
                        f"Found {len(other_columns)} additional columns that can be used as exogenous variables for SARIMAX models."
                    )
                    show_additional = st.checkbox("Preview additional columns", value=True)
                    if show_additional:
                        st.write("Additional columns:", other_columns)
                        st.dataframe(df[other_columns].head())

                if st.button("Process Data"):
                    try:
                        # Create time series
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                        series = pd.Series(
                            df[value_col].values, index=df[timestamp_col], name=value_col
                        )

                        # Store in session state
                        st.session_state.data = series
                        st.session_state.original_data = (
                            df  # Store original dataframe for exogenous variables
                        )
                        st.session_state.timestamp_col = timestamp_col  # Store column names
                        st.session_state.value_col = value_col
                        st.session_state.data_loaded = True

                        st.success("Dati elaborati con successo!")
                        st.info("Vai su 'Data Exploration' per esplorare i tuoi dati.")

                    except Exception as e:
                        st.error(f"Errore nell'elaborazione dati: {e}")

            except Exception as e:
                st.error(f"Errore nel caricamento file: {e}")

        # Sample data option
        st.subheader("Oppure Usa Dati di Esempio")
        sample_type = st.selectbox(
            "Select sample data type",
            ["Simple Time Series", "Time Series with Exogenous Variables"],
        )

        if st.button("Generate Sample Data"):
            # Generate sample time series data
            dates = pd.date_range("2020-01-01", periods=365, freq="D")
            np.random.seed(42)
            trend = np.linspace(100, 150, 365)
            seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)
            noise = np.random.normal(0, 5, 365)
            values = trend + seasonal + noise

            if sample_type == "Simple Time Series":
                series = pd.Series(values, index=dates, name="sample_data")

                st.session_state.data = series
                st.session_state.timestamp_col = "timestamp"  # Per dati di esempio semplici
                st.session_state.value_col = "sample_data"
                st.session_state.data_loaded = True

                st.success("Dati di esempio generati!")
                st.info("Vai su 'Data Exploration' per esplorare i dati di esempio.")

            else:  # Time Series with Exogenous Variables
                # Generate exogenous variables
                temperature = (
                    20
                    + 10 * np.sin(2 * np.pi * np.arange(365) / 365.25)
                    + np.random.normal(0, 2, 365)
                )
                marketing_spend = 1000 + 50 * np.arange(365) + np.random.normal(0, 200, 365)
                day_of_week = [(i % 7) for i in range(365)]

                # Create DataFrame with all variables
                df_sample = pd.DataFrame(
                    {
                        "timestamp": dates,
                        "value": values,
                        "temperature": temperature,
                        "marketing_spend": marketing_spend,
                        "day_of_week": day_of_week,
                    }
                )

                series = pd.Series(values, index=dates, name="sample_data")

                st.session_state.data = series
                st.session_state.original_data = df_sample
                st.session_state.timestamp_col = "timestamp"  # Per dati di esempio
                st.session_state.value_col = "value"
                st.session_state.data_loaded = True

                st.success("Dati di esempio con variabili esogene generati!")
                st.info(
                    "Ora puoi usare modelli SARIMAX con le variabili esogene: temperature, marketing_spend, day_of_week"
                )
                st.info("Vai su 'Data Exploration' per esplorare i dati di esempio.")

    def data_exploration_page(self):
        """Data exploration and visualization page."""
        st.header("🔍 Esplorazione Dati")

        if not st.session_state.get("data_loaded", False):
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
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                mode="lines",
                name=data.name or "Values",
                line=dict(color="blue", width=1),
            )
        )
        fig.update_layout(
            title="Original Time Series",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
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
                ["none", "interpolate", "forward_fill", "backward_fill", "drop"],
            )

            # Outlier detection
            st.write("**Outliers**")
            outlier_detection = st.checkbox("Remove outliers")
            outlier_method = st.selectbox(
                "Outlier method",
                ["iqr", "zscore", "modified_zscore"],
                disabled=not outlier_detection,
            )

        with col2:
            # Stationarity
            st.write("**Stationarity**")
            make_stationary = st.checkbox("Make stationary")
            stationary_method = st.selectbox(
                "Stationary method", ["difference", "log_difference"], disabled=not make_stationary
            )

        if st.button("Apply Preprocessing"):
            try:
                processed_data, metadata = preprocessor.preprocess_pipeline(
                    data.copy(),
                    handle_missing=missing_method != "none",
                    missing_method=missing_method if missing_method != "none" else "interpolate",
                    remove_outliers_flag=outlier_detection,
                    outlier_method=outlier_method,
                    make_stationary_flag=make_stationary,
                    stationarity_method=stationary_method,
                )

                st.session_state.preprocessed_data = processed_data

                # Show comparison
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=("Original Data", "Preprocessed Data"),
                    vertical_spacing=0.1,
                )

                fig.add_trace(
                    go.Scatter(x=data.index, y=data.values, name="Original"), row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=processed_data.index, y=processed_data.values, name="Preprocessed"
                    ),
                    row=2,
                    col=1,
                )

                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.success("Preprocessing completato!")

            except Exception as e:
                st.error(f"Preprocessing fallito: {e}")

        # Show stationarity test results
        if st.button("Check Stationarity"):
            from statsmodels.tsa.stattools import adfuller

            current_data = st.session_state.get("preprocessed_data", data)

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

        if not st.session_state.get("data_loaded", False):
            st.warning("Per favore carica prima i dati nella pagina 'Data Upload'.")
            return

        # Get data
        data = st.session_state.get("preprocessed_data", st.session_state.data)

        # Model selection
        st.subheader("Configurazione Modello")

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "Select model type",
                [
                    "ARIMA",
                    "SARIMA",
                    "SARIMAX",
                    "Advanced SARIMAX",
                    "Prophet",
                    "Auto-ARIMA",
                    "Auto-SARIMA",
                    "Auto-SARIMAX",
                    "Auto-Prophet",
                ],
            )

            if model_type in ["ARIMA", "SARIMA", "SARIMAX"]:
                st.subheader("Parametri Manuali")
                p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
                d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
                q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)

                if model_type in ["SARIMA", "SARIMAX"]:
                    st.write("**Seasonal Parameters**")
                    P = st.number_input("Seasonal AR (P)", min_value=0, max_value=2, value=1)
                    D = st.number_input("Seasonal Diff (D)", min_value=0, max_value=1, value=1)
                    Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=2, value=1)
                    s = st.number_input("Seasonal period (s)", min_value=2, max_value=365, value=12)

            elif model_type == "Advanced SARIMAX":
                st.subheader("⭐ Advanced Exogenous Handling")
                st.info(
                    "Automatic feature selection, preprocessing, and diagnostics for SARIMAX with multiple exogenous variables"
                )

                # Parametri base SARIMA
                st.write("**Base SARIMA Parameters**")
                p = st.number_input("AR order (p)", min_value=0, max_value=3, value=1, key="adv_p")
                d = st.number_input(
                    "Differencing (d)", min_value=0, max_value=2, value=1, key="adv_d"
                )
                q = st.number_input("MA order (q)", min_value=0, max_value=3, value=1, key="adv_q")

                st.write("**Seasonal Parameters**")
                P = st.number_input(
                    "Seasonal AR (P)", min_value=0, max_value=2, value=1, key="adv_P"
                )
                D = st.number_input(
                    "Seasonal Diff (D)", min_value=0, max_value=1, value=1, key="adv_D"
                )
                Q = st.number_input(
                    "Seasonal MA (Q)", min_value=0, max_value=2, value=1, key="adv_Q"
                )
                s = st.number_input(
                    "Seasonal period (s)", min_value=2, max_value=365, value=12, key="adv_s"
                )

                # Advanced Exog parameters
                st.write("**Advanced Exog Settings**")
                max_features = st.number_input(
                    "Max features to select",
                    min_value=3,
                    max_value=20,
                    value=8,
                    key="adv_max_features",
                )
                selection_method = st.selectbox(
                    "Feature selection method",
                    ["stepwise", "lasso", "elastic_net", "f_test"],
                    key="adv_selection",
                )

                preprocessing_method = st.selectbox(
                    "Preprocessing method",
                    ["auto", "robust", "standard", "minmax"],
                    key="adv_preprocessing",
                )

                feature_engineering = st.multiselect(
                    "Feature engineering",
                    ["lags", "differences", "interactions"],
                    default=["lags"],
                    key="adv_engineering",
                )

                if model_type == "SARIMAX" or model_type == "Advanced SARIMAX":
                    st.write("**Exogenous Variables**")
                    st.info(
                        "SARIMAX requires exogenous variables. Upload data with additional columns for exogenous variables."
                    )

                    # Check if we have additional columns that could be exogenous
                    if (
                        hasattr(st.session_state, "original_data")
                        and len(st.session_state.original_data.columns) > 2
                    ):
                        # Escludi le colonne timestamp e value (usando i nomi reali salvati)
                        exclude_cols = [
                            st.session_state.get("timestamp_col", "timestamp"),
                            st.session_state.get("value_col", "value"),
                        ]
                        available_columns = [
                            col
                            for col in st.session_state.original_data.columns
                            if col not in exclude_cols
                        ]
                        exog_variables = st.multiselect(
                            "Select exogenous variables",
                            available_columns,
                            default=available_columns[:3]
                            if len(available_columns) >= 3
                            else available_columns,
                        )
                    else:
                        st.warning(
                            "No additional columns found for exogenous variables. Please upload data with extra columns."
                        )
                        exog_variables = []

            elif model_type == "Prophet":
                st.subheader("📊 Prophet Parameters")
                st.info(
                    "Facebook Prophet automatically handles trends, seasonality, and holidays"
                )
                
                # Growth type
                growth_type = st.selectbox(
                    "Growth type",
                    ["linear", "logistic", "flat"],
                    index=0,
                    help="Type of trend growth: linear for continuous growth, logistic for saturating growth, flat for no trend"
                )
                
                # Seasonality settings
                st.write("**Seasonality Settings**")
                yearly_seasonality = st.selectbox(
                    "Yearly seasonality",
                    ["auto", True, False],
                    index=0,
                    help="Whether to include yearly seasonal effects"
                )
                
                weekly_seasonality = st.selectbox(
                    "Weekly seasonality", 
                    ["auto", True, False],
                    index=0,
                    help="Whether to include weekly seasonal effects"
                )
                
                daily_seasonality = st.selectbox(
                    "Daily seasonality",
                    ["auto", True, False],
                    index=0,
                    help="Whether to include daily seasonal effects"
                )
                
                # Holiday settings
                st.write("**Holiday Settings**")
                include_holidays = st.checkbox(
                    "Include holidays",
                    value=True,
                    help="Whether to include country-specific holidays"
                )
                
                country_holidays = None
                if include_holidays:
                    country_holidays = st.selectbox(
                        "Country for holidays",
                        ["IT", "US", "UK", "DE", "FR", "ES"],
                        index=0,
                        help="Country code for holiday calendar"
                    )
                
                # Advanced settings
                with st.expander("Advanced Prophet Settings"):
                    seasonality_mode = st.selectbox(
                        "Seasonality mode",
                        ["additive", "multiplicative"],
                        index=0,
                        help="Whether seasonality is additive or multiplicative"
                    )
                    
                    changepoint_prior_scale = st.slider(
                        "Changepoint prior scale",
                        min_value=0.001,
                        max_value=0.5,
                        value=0.05,
                        step=0.001,
                        help="Flexibility of the trend (higher = more flexible)"
                    )
                    
                    seasonality_prior_scale = st.slider(
                        "Seasonality prior scale",
                        min_value=0.01,
                        max_value=50.0,
                        value=10.0,
                        step=0.1,
                        help="Flexibility of the seasonality (higher = more flexible)"
                    )
                    
                    holidays_prior_scale = st.slider(
                        "Holidays prior scale",
                        min_value=0.01,
                        max_value=50.0,
                        value=10.0,
                        step=0.1,
                        help="Flexibility of holiday effects (higher = more flexible)"
                    )

        with col2:
            if model_type.startswith("Auto"):
                st.subheader("Parametri Auto-selezione")
                max_p = st.number_input("Max AR order", min_value=0, max_value=5, value=3)
                max_d = st.number_input("Max Differencing", min_value=0, max_value=2, value=2)
                max_q = st.number_input("Max MA order", min_value=0, max_value=5, value=3)

                if model_type in ["Auto-SARIMA", "Auto-SARIMAX"]:
                    max_P = st.number_input("Max Seasonal AR", min_value=0, max_value=2, value=1)
                    max_D = st.number_input("Max Seasonal Diff", min_value=0, max_value=1, value=1)
                    max_Q = st.number_input("Max Seasonal MA", min_value=0, max_value=2, value=1)
                    seasonal_periods = st.multiselect(
                        "Seasonal periods to test", [4, 7, 12, 24, 52, 365], default=[12]
                    )

                if model_type == "Auto-SARIMAX":
                    st.write("**Exogenous Variables**")
                    st.info("Auto-SARIMAX requires exogenous variables for automatic selection.")

                    # Check if we have additional columns that could be exogenous
                    if (
                        hasattr(st.session_state, "original_data")
                        and len(st.session_state.original_data.columns) > 2
                    ):
                        # Escludi le colonne timestamp e value (usando i nomi reali salvati)
                        exclude_cols = [
                            st.session_state.get("timestamp_col", "timestamp"),
                            st.session_state.get("value_col", "value"),
                        ]
                        available_columns = [
                            col
                            for col in st.session_state.original_data.columns
                            if col not in exclude_cols
                        ]
                        auto_exog_variables = st.multiselect(
                            "Select exogenous variables for auto-selection",
                            available_columns,
                            default=available_columns[:3]
                            if len(available_columns) >= 3
                            else available_columns,
                        )
                    else:
                        st.warning(
                            "No additional columns found for exogenous variables. Please upload data with extra columns."
                        )
                        auto_exog_variables = []

                if model_type == "Auto-Prophet":
                    st.write("**Prophet Auto Settings**")
                    st.info("Auto-Prophet will test multiple parameter combinations automatically.")
                    
                    # Prophet specific auto parameters
                    test_growth_types = st.multiselect(
                        "Growth types to test",
                        ["linear", "logistic", "flat"],
                        default=["linear", "logistic"]
                    )
                    
                    test_seasonality_modes = st.multiselect(
                        "Seasonality modes to test",
                        ["additive", "multiplicative"],
                        default=["additive"]
                    )
                    
                    test_countries = st.multiselect(
                        "Countries for holidays to test",
                        ["IT", "US", "UK", "DE", "FR", "ES", "None"],
                        default=["IT", "None"],
                        help="Test different holiday calendars, including no holidays"
                    )

                ic = st.selectbox("Information Criterion", ["aic", "bic", "hqic"])
                max_models = st.number_input(
                    "Max models to test", min_value=10, max_value=200, value=50
                )

        # Train model
        if st.button("Train Model", type="primary"):
            try:
                with st.spinner("Training model..."):
                    if model_type == "ARIMA":
                        model = ARIMAForecaster(order=(p, d, q))
                        model.fit(data)

                    elif model_type == "SARIMA":
                        model = SARIMAForecaster(order=(p, d, q), seasonal_order=(P, D, Q, s))
                        model.fit(data)

                    elif model_type == "SARIMAX":
                        if not exog_variables:
                            st.error("Please select at least one exogenous variable for SARIMAX.")
                            return

                        # Prepare exogenous data con preprocessing robusto
                        try:
                            exog_data = st.session_state.original_data[exog_variables].copy()
                            if len(exog_data) != len(data):
                                st.error(
                                    f"Mismatch dimensioni: serie ha {len(data)} osservazioni, variabili esogene hanno {len(exog_data)} osservazioni"
                                )
                                return
                            exog_data.index = data.index

                            # Applica preprocessing alle variabili esogene
                            try:
                                from arima_forecaster.utils.preprocessing import (
                                    ExogenousPreprocessor,
                                    suggest_preprocessing_method,
                                    validate_exog_data,
                                )
                            except ImportError:
                                st.error(
                                    "Modulo preprocessing non disponibile - usando dati originali"
                                )
                                exog_data = st.session_state.original_data[exog_variables].copy()
                                exog_data.index = data.index
                                raise Exception("Preprocessing non disponibile")

                            # Valida dati esogeni
                            is_valid, error_msg = validate_exog_data(exog_data, len(data))
                            if not is_valid:
                                st.error(f"Problemi con variabili esogene: {error_msg}")
                                return

                            # Suggerisci e applica preprocessing
                            preprocessing_method = suggest_preprocessing_method(exog_data)
                            preprocessor = ExogenousPreprocessor(
                                method=preprocessing_method, 
                                handle_outliers=False,
                                detect_multicollinearity=False,
                                stationarity_test=False
                            )
                            exog_data_processed = preprocessor.fit_transform(exog_data)

                            # Salva il preprocessor per il forecast
                            st.session_state.exog_preprocessor = preprocessor

                            # Informa l'utente del preprocessing applicato
                            st.info(
                                f"🔧 Preprocessing automatico applicato: {preprocessing_method.upper()}"
                            )
                            with st.expander("Dettagli preprocessing"):
                                stats = preprocessor.get_stats()
                                for var, var_stats in stats.items():
                                    st.write(f"**{var}**: {var_stats}")

                            # Usa i dati processati
                            exog_data = exog_data_processed

                        except KeyError as e:
                            st.error(f"Variabile esogena non trovata: {e}")
                            return
                        except Exception as e:
                            st.warning(f"Errore preprocessing: {e} - usando dati originali")
                            # Fallback ai dati originali
                            exog_data = st.session_state.original_data[exog_variables].copy()
                            exog_data.index = data.index

                        model = SARIMAXForecaster(order=(p, d, q), seasonal_order=(P, D, Q, s))
                        model.fit(data, exog=exog_data)

                        # Store exogenous variables for forecasting
                        st.session_state.exog_variables = exog_variables
                        st.session_state.exog_data = exog_data

                    elif model_type == "Advanced SARIMAX":
                        if not exog_variables:
                            st.error(
                                "Please select at least one exogenous variable for Advanced SARIMAX."
                            )
                            return

                        st.info("🚀 Training Advanced SARIMAX with automatic feature selection...")

                        # Prepare exogenous data
                        try:
                            exog_data = st.session_state.original_data[exog_variables].copy()
                            if len(exog_data) != len(data):
                                st.error(
                                    f"Dimension mismatch: series has {len(data)} observations, exog has {len(exog_data)}"
                                )
                                return
                            
                            # Allineamento robusto degli indici
                            exog_data.index = data.index
                            
                            # Verifica finale allineamento
                            if not data.index.equals(exog_data.index):
                                st.error("Impossibile allineare gli indici delle serie temporali")
                                return

                            # Create Advanced SARIMAX selector
                            model = SARIMAXAutoSelector(
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                max_features=max_features,
                                selection_method=selection_method,
                                feature_engineering=feature_engineering,
                                preprocessing_method=preprocessing_method,
                                validation_split=0.2,
                            )

                            # Training with automatic feature selection
                            model.fit_with_exog(data, exog_data)

                            # Display feature analysis
                            feature_analysis = model.get_feature_analysis()

                            st.success("🎯 Advanced SARIMAX training completed!")

                            # Display selection summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Original Features",
                                    feature_analysis["selection_summary"]["original_features"],
                                )
                            with col2:
                                st.metric(
                                    "Selected Features",
                                    feature_analysis["selection_summary"]["selected_features"],
                                )
                            with col3:
                                st.metric(
                                    "Selection Ratio",
                                    f"{feature_analysis['selection_summary']['selection_ratio']:.1%}",
                                )

                            # Display selected features
                            with st.expander("Selected Features Details"):
                                st.write("**Selected Features:**")
                                for feat in feature_analysis["feature_details"][
                                    "selected_features"
                                ]:
                                    st.write(f"• {feat}")

                                if feature_analysis["feature_details"]["feature_importance"]:
                                    st.write("**Feature Importance:**")
                                    importance_data = []
                                    for feat, importance in feature_analysis["feature_details"][
                                        "feature_importance"
                                    ].items():
                                        importance_data.append(
                                            {"Feature": feat, "Importance": abs(importance)}
                                        )

                                    importance_df = pd.DataFrame(importance_data).sort_values(
                                        "Importance", ascending=False
                                    )
                                    st.dataframe(importance_df)

                            # Store for forecasting
                            st.session_state.exog_variables = exog_variables
                            st.session_state.exog_data = exog_data
                            st.session_state.selected_features = model.selected_features

                        except Exception as e:
                            st.error(f"Advanced SARIMAX training failed: {str(e)}")
                            return

                    elif model_type == "Auto-ARIMA":
                        try:
                            selector = ARIMAModelSelector(
                                p_range=(0, max_p),
                                d_range=(0, max_d),
                                q_range=(0, max_q),
                                information_criterion=ic,
                            )
                            selector.search(data, verbose=False)
                            model = selector.get_best_model()

                            if model is None:
                                st.error(
                                    "Nessun modello ARIMA valido trovato. Prova a modificare i parametri di ricerca."
                                )
                                return
                        except Exception as e:
                            st.error(f"Errore durante la selezione Auto-ARIMA: {str(e)}")
                            return

                        # Show selection results
                        st.subheader("Selezione Miglior Modello")
                        st.write(f"Selected order: {selector.best_order}")
                        results_df = selector.get_results_summary(10)
                        st.dataframe(results_df)

                    elif model_type == "Auto-SARIMA":
                        selector = SARIMAModelSelector(
                            p_range=(0, max_p),
                            d_range=(0, max_d),
                            q_range=(0, max_q),
                            P_range=(0, max_P),
                            D_range=(0, max_D),
                            Q_range=(0, max_Q),
                            seasonal_periods=seasonal_periods,
                            information_criterion=ic,
                            max_models=max_models,
                        )
                        selector.search(data, verbose=False)
                        model = selector.get_best_model()

                        # Show selection results
                        st.subheader("Selezione Miglior Modello")
                        st.write(f"Selected order: {selector.best_order}")
                        st.write(f"Selected seasonal order: {selector.best_seasonal_order}")
                        results_df = selector.get_results_summary(10)
                        st.dataframe(results_df)

                    elif model_type == "Auto-SARIMAX":
                        if not auto_exog_variables:
                            st.error(
                                "Please select at least one exogenous variable for Auto-SARIMAX."
                            )
                            return

                        # Prepare exogenous data con preprocessing robusto
                        try:
                            exog_data = st.session_state.original_data[auto_exog_variables].copy()
                            if len(exog_data) != len(data):
                                st.error(
                                    f"Mismatch dimensioni: serie ha {len(data)} osservazioni, variabili esogene hanno {len(exog_data)} osservazioni"
                                )
                                return
                            exog_data.index = data.index

                            # Applica preprocessing alle variabili esogene
                            try:
                                from arima_forecaster.utils.preprocessing import (
                                    ExogenousPreprocessor,
                                    suggest_preprocessing_method,
                                    validate_exog_data,
                                )
                            except ImportError:
                                st.error(
                                    "Modulo preprocessing non disponibile - usando dati originali"
                                )
                                exog_data = st.session_state.original_data[exog_variables].copy()
                                exog_data.index = data.index
                                raise Exception("Preprocessing non disponibile")

                            # Valida dati esogeni
                            is_valid, error_msg = validate_exog_data(exog_data, len(data))
                            if not is_valid:
                                st.error(f"Problemi con variabili esogene: {error_msg}")
                                return

                            # Suggerisci e applica preprocessing
                            preprocessing_method = suggest_preprocessing_method(exog_data)
                            preprocessor = ExogenousPreprocessor(
                                method=preprocessing_method, 
                                handle_outliers=False,
                                detect_multicollinearity=False,
                                stationarity_test=False
                            )
                            exog_data_processed = preprocessor.fit_transform(exog_data)

                            # Salva il preprocessor per il forecast
                            st.session_state.exog_preprocessor = preprocessor

                            # Informa l'utente del preprocessing applicato
                            st.info(
                                f"🔧 Preprocessing automatico applicato: {preprocessing_method.upper()}"
                            )
                            with st.expander("Dettagli preprocessing"):
                                stats = preprocessor.get_stats()
                                for var, var_stats in stats.items():
                                    st.write(f"**{var}**: {var_stats}")

                            # Usa i dati processati
                            exog_data = exog_data_processed

                        except KeyError as e:
                            st.error(f"Variabile esogena non trovata: {e}")
                            return
                        except Exception as e:
                            st.warning(f"Errore preprocessing: {e} - usando dati originali")
                            # Fallback ai dati originali
                            exog_data = st.session_state.original_data[auto_exog_variables].copy()
                            exog_data.index = data.index

                        selector = SARIMAXModelSelector(
                            p_range=(0, max_p),
                            d_range=(0, max_d),
                            q_range=(0, max_q),
                            P_range=(0, max_P),
                            D_range=(0, max_D),
                            Q_range=(0, max_Q),
                            seasonal_periods=seasonal_periods,
                            information_criterion=ic,
                            max_models=max_models,
                            exog_names=auto_exog_variables,
                        )

                        with st.spinner(
                            f"Testando {max_models} combinazioni di modelli SARIMAX..."
                        ):
                            selector.search(data, exog=exog_data, verbose=False)
                            model = selector.get_best_model()

                        # Verifica che il modello sia stato trovato
                        if model is None:
                            st.error(
                                "Nessun modello SARIMAX valido trovato. Prova a modificare i parametri di ricerca."
                            )
                            return

                        # Show selection results
                        st.subheader("Selezione Miglior Modello")
                        st.write(f"Selected order: {selector.best_order}")
                        st.write(f"Selected seasonal order: {selector.best_seasonal_order}")
                        st.write(f"Exogenous variables: {auto_exog_variables}")
                        results_df = selector.get_results_summary(10)
                        if not results_df.empty:
                            st.dataframe(results_df)

                        # Store exogenous variables for forecasting
                        st.session_state.exog_variables = auto_exog_variables
                        st.session_state.exog_data = exog_data

                    elif model_type == "Prophet":
                        # Check if Prophet is available
                        try:
                            # Prepare Prophet parameters
                            prophet_params = {
                                'growth': growth_type,
                                'yearly_seasonality': yearly_seasonality,
                                'weekly_seasonality': weekly_seasonality,
                                'daily_seasonality': daily_seasonality,
                                'seasonality_mode': seasonality_mode,
                                'changepoint_prior_scale': changepoint_prior_scale,
                                'seasonality_prior_scale': seasonality_prior_scale,
                                'holidays_prior_scale': holidays_prior_scale,
                            }
                            
                            if include_holidays and country_holidays:
                                prophet_params['country_holidays'] = country_holidays

                            model = ProphetForecaster(**prophet_params)
                            model.fit(data)

                            st.success("✅ Modello Prophet addestrato con successo!")
                            
                            # Prophet specific info
                            st.subheader("Prophet Model Info")
                            st.write(f"**Growth**: {growth_type}")
                            st.write(f"**Seasonality Mode**: {seasonality_mode}")
                            st.write(f"**Yearly Seasonality**: {yearly_seasonality}")
                            st.write(f"**Weekly Seasonality**: {weekly_seasonality}")
                            st.write(f"**Daily Seasonality**: {daily_seasonality}")
                            if include_holidays and country_holidays:
                                st.write(f"**Holidays**: {country_holidays}")

                        except ImportError:
                            st.error(
                                "Facebook Prophet non disponibile. Installa con: pip install prophet"
                            )
                            return
                        except Exception as e:
                            st.error(f"Errore nell'addestramento Prophet: {e}")
                            return

                    elif model_type == "Auto-Prophet":
                        # Check if Prophet is available
                        try:
                            # Prepare search parameters - Fix: growth_modes nel costruttore
                            search_params = {
                                'growth_modes': test_growth_types,  # Fix: growth_modes non growth_types
                                'seasonality_modes': test_seasonality_modes,
                                'max_models': max_models
                            }

                            selector = ProphetModelSelector(**search_params)
                            
                            # country_holidays va nel search(), non nel costruttore
                            country_holiday = None
                            if test_countries and len(test_countries) > 0:
                                first_country = test_countries[0]
                                country_holiday = first_country if first_country != "None" else None
                            
                            with st.spinner(f"Testando {max_models} combinazioni di modelli Prophet..."):
                                selector.search(data, country_holidays=country_holiday)
                                model = selector.get_best_model()

                            # Verifica che il modello sia stato trovato
                            if model is None:
                                st.error(
                                    "Nessun modello Prophet valido trovato. Prova a modificare i parametri di ricerca."
                                )
                                return

                            # Show selection results
                            st.subheader("Selezione Miglior Modello Prophet")
                            best_params = selector.get_best_params()
                            st.write("**Best Parameters:**")
                            for param, value in best_params.items():
                                st.write(f"- **{param}**: {value}")
                            
                            # Show results summary if available
                            try:
                                results_df = selector.get_results_summary(10)
                                if not results_df.empty:
                                    st.subheader("Top 10 Models")
                                    st.dataframe(results_df)
                            except:
                                pass

                        except ImportError:
                            st.error(
                                "Facebook Prophet non disponibile. Installa con: pip install prophet"
                            )
                            return
                        except Exception as e:
                            st.error(f"Errore nell'auto-selezione Prophet: {e}")
                            return

                    # Store model in session state
                    st.session_state.model = model
                    st.session_state.model_trained = True

                    # Show model information (skip for Prophet as it was already shown)
                    if model_type not in ["Prophet", "Auto-Prophet"]:
                        st.success("Modello addestrato con successo!")

                        model_info = model.get_model_info()
                        st.subheader("Informazioni Modello")

                        # Check if model has traditional metrics (ARIMA/SARIMA models)
                        if 'aic' in model_info:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("AIC", f"{model_info.get('aic', 0):.2f}")
                            with col2:
                                st.metric("BIC", f"{model_info.get('bic', 0):.2f}")
                            with col3:
                                st.metric("HQIC", f"{model_info.get('hqic', 0):.2f}")
                        else:
                            # For models without traditional metrics, show available info
                            for key, value in model_info.items():
                                st.write(f"**{key}**: {value}")

                    # Show fitted values vs actual
                    st.subheader("Adattamento Modello")

                    fitted_values = model.predict()

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data.values,
                            mode="lines",
                            name="Actual",
                            line=dict(color="blue"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fitted_values.index,
                            y=fitted_values.values,
                            mode="lines",
                            name="Fitted",
                            line=dict(color="red", dash="dash"),
                        )
                    )

                    fig.update_layout(
                        title="Actual vs Fitted Values", xaxis_title="Time", yaxis_title="Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Addestramento modello fallito: {e}")
                logger.error(f"Model training failed: {e}")

    def forecasting_page(self):
        """Forecasting page."""
        st.header("📊 Forecasting")

        if not st.session_state.get("model_trained", False):
            st.warning("Per favore addestra prima un modello nella pagina 'Model Training'.")
            return

        model = st.session_state.model
        data = st.session_state.get("preprocessed_data", st.session_state.data)

        # Forecast parameters
        st.subheader("Configurazione Forecast")

        col1, col2 = st.columns(2)
        with col1:
            forecast_steps = st.number_input(
                "Number of forecast steps", min_value=1, max_value=100, value=10
            )

        with col2:
            confidence_level = st.slider(
                "Confidence level", min_value=0.8, max_value=0.99, value=0.95, step=0.01
            )

        show_intervals = st.checkbox("Show confidence intervals", value=True)

        # SARIMAX specific: Handle exogenous variables for forecasting
        exog_future = None
        if hasattr(st.session_state, "exog_variables") and st.session_state.exog_variables:
            st.subheader("Exogenous Variables for Forecast")
            st.info("SARIMAX models require future values of exogenous variables for forecasting.")

            # Options for providing future exogenous values
            exog_method = st.radio(
                "How to provide future exogenous values?",
                ["Manual Input", "Extend Last Values", "Linear Trend", "Upload CSV"],
            )

            if exog_method == "Manual Input":
                st.write("Enter future values for each exogenous variable:")
                exog_future_dict = {}
                for var in st.session_state.exog_variables:
                    exog_future_dict[var] = st.number_input(
                        f"Future value for {var} (constant for all forecast steps)",
                        value=float(st.session_state.exog_data[var].iloc[-1]),
                    )

                # Create DataFrame with repeated values
                exog_future = pd.DataFrame(
                    {var: [value] * forecast_steps for var, value in exog_future_dict.items()}
                )

            elif exog_method == "Extend Last Values":
                # Use last known values
                last_values = st.session_state.exog_data.iloc[-1]
                exog_future = pd.DataFrame(
                    {
                        var: [last_values[var]] * forecast_steps
                        for var in st.session_state.exog_variables
                    }
                )
                st.write("Using last known values:", last_values.to_dict())

            elif exog_method == "Linear Trend":
                st.write("Extending with linear trend based on last 5 observations:")
                exog_future_dict = {}
                for var in st.session_state.exog_variables:
                    # Calculate simple linear trend
                    recent_values = st.session_state.exog_data[var].iloc[-5:].values
                    trend = np.mean(np.diff(recent_values))
                    start_value = st.session_state.exog_data[var].iloc[-1]

                    future_values = [start_value + trend * (i + 1) for i in range(forecast_steps)]
                    exog_future_dict[var] = future_values
                    st.write(f"{var}: trend = {trend:.4f}, starting from {start_value:.4f}")

                exog_future = pd.DataFrame(exog_future_dict)

            elif exog_method == "Upload CSV":
                uploaded_exog = st.file_uploader(
                    "Upload CSV with future exogenous values",
                    type=["csv"],
                    help=f"CSV deve contenere almeno {forecast_steps} righe e le colonne: {', '.join(st.session_state.exog_variables)}. Se ha più righe, verranno usate solo le prime {forecast_steps}.",
                )

                if uploaded_exog is not None:
                    try:
                        exog_future = pd.read_csv(uploaded_exog)
                        st.write("Uploaded exogenous data preview:", exog_future.head())

                        # Validate columns
                        missing_cols = [
                            col
                            for col in st.session_state.exog_variables
                            if col not in exog_future.columns
                        ]
                        if missing_cols:
                            st.error(f"CSV manca le seguenti colonne richieste: {missing_cols}")
                            st.info(f"Colonne richieste: {st.session_state.exog_variables}")
                            st.info(f"Colonne trovate nel CSV: {list(exog_future.columns)}")
                            exog_future = None
                        elif len(exog_future) < forecast_steps:
                            st.error(f"CSV ha insufficienti righe per l'orizzonte di forecast.")
                            st.info(
                                f"Righe richieste: {forecast_steps}, Righe trovate: {len(exog_future)}"
                            )
                            exog_future = None
                        else:
                            # Seleziona solo le colonne necessarie
                            exog_future = exog_future[st.session_state.exog_variables]

                            # Se ci sono più righe del necessario, prendi solo le prime N
                            if len(exog_future) > forecast_steps:
                                total_rows = len(exog_future)
                                exog_future = exog_future.head(forecast_steps)
                                st.warning(
                                    f"⚠️ CSV aveva {total_rows} righe, utilizzate solo le prime {forecast_steps} per il forecast"
                                )

                            st.success(
                                f"✓ CSV caricato correttamente con {len(exog_future)} righe per {forecast_steps} steps di forecast"
                            )
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
                        exog_future = None

        # Generate forecast
        if st.button("Generate Forecast", type="primary"):
            try:
                with st.spinner("Generating forecast..."):
                    # Check if SARIMAX requires exogenous variables
                    if (
                        hasattr(st.session_state, "exog_variables")
                        and st.session_state.exog_variables
                    ):
                        if exog_future is None:
                            st.error(
                                "Please provide future exogenous variables for SARIMAX forecast."
                            )
                            return

                        # Applica lo stesso preprocessing ai dati futuri
                        if (
                            hasattr(st.session_state, "exog_preprocessor")
                            and st.session_state.exog_preprocessor
                        ):
                            try:
                                st.info("🔧 Applicando preprocessing ai dati futuri...")

                                # Debug: mostra range prima e dopo
                                st.write("**Debug - Dati futuri prima del preprocessing:**")
                                for var in st.session_state.exog_variables:
                                    if var in exog_future.columns:
                                        min_val, max_val = (
                                            exog_future[var].min(),
                                            exog_future[var].max(),
                                        )
                                        st.write(f"   {var}: {min_val:.2f} - {max_val:.2f}")

                                exog_future_original = exog_future.copy()
                                exog_future = st.session_state.exog_preprocessor.transform(
                                    exog_future
                                )

                                st.write("**Debug - Dati futuri dopo preprocessing:**")
                                for var in st.session_state.exog_variables:
                                    if var in exog_future.columns:
                                        min_val, max_val = (
                                            exog_future[var].min(),
                                            exog_future[var].max(),
                                        )
                                        st.write(f"   {var}: {min_val:.3f} - {max_val:.3f}")

                                st.success("✓ Preprocessing applicato ai dati futuri")

                            except Exception as e:
                                st.error(f"Errore nel preprocessing dei dati futuri: {e}")
                                logger.error(f"Future data preprocessing failed: {e}")
                                return
                        else:
                            st.warning(
                                "⚠️ Preprocessor non trovato in sessione - usando dati futuri originali"
                            )
                            st.write(
                                "**PROBLEMA**: Questo può causare forecast non coerenti se il modello è stato trainato con preprocessing."
                            )
                            st.write(
                                "**SOLUZIONE**: Rifare il training del modello SARIMAX per salvare il preprocessor."
                            )
                            if not hasattr(st.session_state, "exog_preprocessor"):
                                st.write("Motivo: exog_preprocessor non esiste nella sessione")
                            else:
                                st.write(
                                    f"Motivo: exog_preprocessor è {st.session_state.exog_preprocessor}"
                                )

                            st.info(
                                "💡 Suggerimento: Vai alla sezione 'Model Training' e re-addestra il modello SARIMAX per risolvere il problema."
                            )

                        # Check if model is SARIMAX or SARIMAXAutoSelector (both support exog_future)
                        if isinstance(model, SARIMAXAutoSelector):
                            # SARIMAXAutoSelector uses forecast_with_exog method
                            if show_intervals:
                                forecast_result = model.forecast_with_exog(
                                    steps=forecast_steps,
                                    exog=exog_future,
                                    alpha=1 - confidence_level,
                                    confidence_intervals=True,
                                )
                                # Handle return format
                                if isinstance(forecast_result, tuple):
                                    forecast, conf_int = forecast_result
                                else:
                                    forecast = forecast_result
                                    conf_int = None
                            else:
                                forecast = model.forecast_with_exog(
                                    steps=forecast_steps,
                                    exog=exog_future,
                                    confidence_intervals=False,
                                )
                                conf_int = None
                        elif isinstance(model, SARIMAXForecaster):
                            # Regular SARIMAX model
                            if show_intervals:
                                forecast, conf_int = model.forecast(
                                    steps=forecast_steps,
                                    exog_future=exog_future,
                                    alpha=1 - confidence_level,
                                    return_conf_int=True,
                                )
                            else:
                                forecast = model.forecast(
                                    steps=forecast_steps,
                                    exog_future=exog_future,
                                    confidence_intervals=False,
                                )
                                conf_int = None
                        else:
                            # SARIMA model (no exogenous variables in forecast)
                            if show_intervals:
                                forecast, conf_int = model.forecast(
                                    steps=forecast_steps,
                                    alpha=1 - confidence_level,
                                    return_conf_int=True,
                                )
                            else:
                                forecast = model.forecast(
                                    steps=forecast_steps,
                                    confidence_intervals=False,
                                )
                                conf_int = None
                    else:
                        # Regular ARIMA/SARIMA forecast
                        if show_intervals:
                            forecast, conf_int = model.forecast(
                                steps=forecast_steps,
                                alpha=1 - confidence_level,
                                return_conf_int=True,
                            )
                        else:
                            forecast = model.forecast(
                                steps=forecast_steps, confidence_intervals=False
                            )
                            conf_int = None

                    # Controllo di sanità sui risultati del forecast
                    if (
                        hasattr(st.session_state, "original_data")
                        and "target_column" in st.session_state
                    ):
                        historical_mean = st.session_state.original_data[
                            st.session_state.target_column
                        ].mean()
                        forecast_mean = forecast.mean()
                        ratio = forecast_mean / historical_mean

                        if ratio > 10 or ratio < 0.1:
                            st.error(
                                f"⚠️ **PROBLEMA RILEVATO**: Il forecast sembra non coerente con i dati storici!"
                            )
                            st.write(f"- Media dati storici: {historical_mean:.2f}")
                            st.write(f"- Media forecast: {forecast_mean:.2f}")
                            st.write(f"- Ratio: {ratio:.2f}")

                            if ratio > 10:
                                st.write("**Possibili cause:**")
                                st.write(
                                    "- Manca il preprocessing sui dati futuri (problema più comune)"
                                )
                                st.write("- Errore nella preparazione delle variabili esogene")
                                st.write("- Il modello potrebbe essere sovraadattato")
                                st.info(
                                    "💡 **Soluzione suggerita**: Re-addestrare il modello SARIMAX dalla sezione Training"
                                )

                    st.session_state.forecast_result = {
                        "forecast": forecast,
                        "conf_int": conf_int,
                        "confidence_level": confidence_level,
                    }
                    st.session_state.forecast_generated = True

                    # Plot forecast
                    st.subheader("Risultati Forecast")

                    fig = go.Figure()

                    # Historical data
                    fig.add_trace(
                        go.Scatter(
                            x=data.index[-50:],  # Show last 50 points
                            y=data.values[-50:],
                            mode="lines",
                            name="Historical",
                            line=dict(color="blue"),
                        )
                    )

                    # Forecast
                    fig.add_trace(
                        go.Scatter(
                            x=forecast.index,
                            y=forecast.values,
                            mode="lines+markers",
                            name="Forecast",
                            line=dict(color="red", width=2),
                            marker=dict(size=6),
                        )
                    )

                    # Confidence intervals
                    if conf_int is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=forecast.index,
                                y=conf_int.iloc[:, 1],
                                mode="lines",
                                name=f"Upper {confidence_level:.1%}",
                                line=dict(color="rgba(255,0,0,0.3)", width=0),
                                showlegend=False,
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=forecast.index,
                                y=conf_int.iloc[:, 0],
                                mode="lines",
                                name=f"Confidence Interval",
                                line=dict(color="rgba(255,0,0,0.3)", width=0),
                                fill="tonexty",
                                fillcolor="rgba(255,0,0,0.2)",
                            )
                        )

                    fig.update_layout(
                        title="Time Series Forecast",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show forecast values
                    st.subheader("Valori Forecast")

                    forecast_df = pd.DataFrame(
                        {"Date": forecast.index, "Forecast": forecast.values}
                    )

                    if conf_int is not None:
                        forecast_df["Lower Bound"] = conf_int.iloc[:, 0].values
                        forecast_df["Upper Bound"] = conf_int.iloc[:, 1].values

                    st.dataframe(forecast_df)

                    # Download forecast
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv,
                        file_name="forecast.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Generazione forecast fallita: {e}")
                logger.error(f"Forecast generation failed: {e}")

    def diagnostics_page(self):
        """Model diagnostics page."""
        st.header("🔧 Diagnostica Modello")

        if not st.session_state.get("model_trained", False):
            st.warning("Per favore addestra prima un modello nella pagina 'Model Training'.")
            return

        model = st.session_state.model
        data = st.session_state.get("preprocessed_data", st.session_state.data)

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
        if hasattr(model, "fitted_model"):
            st.subheader("Analisi Residui")

            try:
                residuals = model.fitted_model.resid

                # Residuals plot
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "Residuals vs Time",
                        "Residuals Distribution",
                        "Q-Q Plot",
                        "ACF of Residuals",
                    ),
                    vertical_spacing=0.1,
                )

                # Residuals vs time
                fig.add_trace(
                    go.Scatter(
                        x=residuals.index, y=residuals.values, mode="lines", name="Residuals"
                    ),
                    row=1,
                    col=1,
                )

                # Residuals histogram
                fig.add_trace(
                    go.Histogram(x=residuals.values, name="Distribution", nbinsx=30), row=1, col=2
                )

                # Q-Q plot (simplified)
                from scipy import stats

                qq_data = stats.probplot(residuals.dropna(), dist="norm")
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode="markers", name="Q-Q"),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[1][0] + qq_data[1][1] * qq_data[0][0],
                        mode="lines",
                        name="Q-Q Line",
                    ),
                    row=2,
                    col=1,
                )

                # ACF of residuals
                try:
                    from statsmodels.tsa.stattools import acf

                    acf_vals = acf(residuals.dropna(), nlags=20)
                    fig.add_trace(go.Bar(x=list(range(21)), y=acf_vals, name="ACF"), row=2, col=2)
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
                    st.write(
                        f"- Residuals are {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)"
                    )
                except Exception as e:
                    st.write("Could not perform normality test")

                # Ljung-Box test for autocorrelation
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox

                    lb_result = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
                    st.write(f"**Ljung-Box Test (Residual Autocorrelation):**")
                    st.write(f"- P-value (lag 10): {lb_result['lb_pvalue'].iloc[9]:.4f}")
                    st.write(
                        f"- No autocorrelation: {'Yes' if lb_result['lb_pvalue'].iloc[9] > 0.05 else 'No'} (α=0.05)"
                    )
                except Exception as e:
                    st.write("Could not perform Ljung-Box test")

            except Exception as e:
                st.error(f"Impossibile generare analisi residui: {e}")

    def report_generation_page(self):
        """Report generation page with proper state management."""
        st.header("📄 Generazione Report")

        if not st.session_state.get("model_trained", False):
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
        if st.session_state.get("report_generated", False) and st.session_state.get(
            "last_report_path"
        ):
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
                help="Custom title for your report",
            )

            output_filename = st.text_input(
                "Output Filename", value="", help="Leave empty for auto-generated filename"
            )

            format_type = st.selectbox(
                "Output Format",
                ["html", "pdf", "docx"],
                index=0,
                help="Choose the output format for your report",
            )

        with col2:
            include_diagnostics = st.checkbox(
                "Include Model Diagnostics",
                value=True,
                help="Include residual analysis, statistical tests, and model validation",
            )

            include_forecast = st.checkbox(
                "Include Forecast Analysis",
                value=True,
                help="Include forecast visualizations and confidence intervals",
            )

            if include_forecast:
                forecast_steps = st.number_input(
                    "Forecast Steps",
                    min_value=1,
                    max_value=100,
                    value=12,
                    help="Number of forecast steps to include in the report",
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
                help="Automatically open the report in your default browser/application",
            )

        # Generate report button
        st.subheader("Genera Report")

        if st.button("🚀 Generate Report", type="primary", use_container_width=True):
            self._generate_report(
                report_title,
                output_filename,
                format_type,
                include_diagnostics,
                include_forecast,
                forecast_steps,
                auto_open,
            )

    def _generate_report(
        self,
        report_title,
        output_filename,
        format_type,
        include_diagnostics,
        include_forecast,
        forecast_steps,
        auto_open,
    ):
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
                if st.session_state.get("forecast_generated", False) and include_forecast:
                    plots_data = self._create_forecast_plot(output_filename)

                # Costruisci i parametri base comuni a tutti i modelli
                report_params = {
                    "plots_data": plots_data,
                    "report_title": report_title,
                    "output_filename": output_filename,
                    "format_type": format_type,
                    "include_diagnostics": include_diagnostics,
                    "include_forecast": include_forecast,
                    "forecast_steps": forecast_steps,
                }

                # Aggiungi parametri specifici per SARIMA e SARIMAX
                model_type = type(model).__name__
                if model_type in ["SARIMAForecaster", "SARIMAXForecaster"]:
                    report_params["include_seasonal_decomposition"] = True
                if model_type == "SARIMAXForecaster":
                    report_params["include_exog_analysis"] = True

                # Generate the report
                report_path = model.generate_report(**report_params)

                # Store in session state
                st.session_state.report_generated = True
                st.session_state.last_report_path = report_path
                st.session_state.last_report_config = {
                    "format_type": format_type,
                    "auto_open": auto_open,
                    "report_title": report_title,
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
            from pathlib import Path

            import matplotlib.pyplot as plt

            # Get data
            data = st.session_state.get("preprocessed_data", st.session_state.data)
            forecast_result = st.session_state.get("forecast_result", {})

            if forecast_result:
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot historical data (last 50 points)
                ax.plot(
                    data.index[-50:],
                    data.values[-50:],
                    label="Historical",
                    color="blue",
                    linewidth=1.5,
                )

                # Plot forecast
                forecast = forecast_result["forecast"]
                ax.plot(
                    forecast.index,
                    forecast.values,
                    label="Forecast",
                    color="red",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                # Plot confidence intervals if available
                conf_int = forecast_result.get("conf_int")
                if conf_int is not None:
                    confidence_level = forecast_result.get("confidence_level", 0.95)
                    ax.fill_between(
                        forecast.index,
                        conf_int.iloc[:, 0],
                        conf_int.iloc[:, 1],
                        alpha=0.3,
                        color="red",
                        label=f"{confidence_level:.0%} Confidence Interval",
                    )

                ax.set_title("Time Series Forecast")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Save plot
                plots_dir = Path("outputs/plots")
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plots_dir / f"{output_filename}_dashboard_plot.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                return {"main_plot": str(plot_path)}

        except Exception as e:
            st.warning(f"Impossibile creare grafico per report: {e}")
            return None

    def _auto_open_report(self, report_path):
        """Attempt to auto-open the report."""
        try:
            import platform
            import subprocess

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
        format_type = config.get("format_type", "html")

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
                        use_container_width=True,
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
            help="Summary shows key info, Full Report shows complete HTML content",
        )

        try:
            with open(report_path, "r", encoding="utf-8") as f:
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
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
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
        st.warning(
            "⚠️ Report grandi potrebbero richiedere tempo per caricare. Se l'anteprima è lenta, usa i pulsanti download/open."
        )

        # Limit HTML size for performance
        max_size = 1024 * 1024 * 2  # 2MB limit
        if len(html_content) > max_size:
            st.warning("Report molto grande. Mostrando anteprima troncata...")
            html_content = html_content[:max_size] + "\n<!-- Content truncated for preview -->"

        st.components.v1.html(html_content, height=800, scrolling=True)

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
                "Data Loaded": st.session_state.get("data_loaded", False),
                "Model Trained": st.session_state.get("model_trained", False),
                "Forecast Generated": st.session_state.get("forecast_generated", False),
                "Report Generated": st.session_state.get("report_generated", False),
            }

            if st.session_state.get("model_trained", False):
                model = st.session_state.model
                model_info = model.get_model_info()
                debug_info.update(
                    {
                        "Model Type": type(model).__name__,
                        "Model Order": model_info.get("order", "N/A"),
                        "AIC": f"{model_info.get('aic', 0):.2f}",
                        "BIC": f"{model_info.get('bic', 0):.2f}",
                    }
                )

            st.json(debug_info)

    def advanced_exog_analysis_page(self):
        """⭐ Advanced Exogenous Variables Analysis Page."""
        st.header("⭐ Advanced Exogenous Variables Analysis")
        st.markdown(
            "Comprehensive analysis, preprocessing, and diagnostics for exogenous variables"
        )

        if not st.session_state.get("data_loaded", False):
            st.warning("Please load data first in the 'Data Upload' page.")
            return

        # Check if we have exogenous data
        if (
            not hasattr(st.session_state, "original_data")
            or len(st.session_state.original_data.columns) <= 2
        ):
            st.warning(
                "No exogenous variables found. Please upload data with additional columns for exogenous analysis."
            )
            return

        # Get data
        data = st.session_state.get("preprocessed_data", st.session_state.data)

        # Get available exogenous variables
        exclude_cols = [
            st.session_state.get("timestamp_col", "timestamp"),
            st.session_state.get("value_col", "value"),
        ]
        available_columns = [
            col for col in st.session_state.original_data.columns if col not in exclude_cols
        ]

        st.info(f"Found {len(available_columns)} potential exogenous variables")

        # Variable selection
        st.subheader("1. 🎯 Exogenous Variable Selection")
        selected_exog = st.multiselect(
            "Select exogenous variables for analysis",
            available_columns,
            default=available_columns[: min(8, len(available_columns))],
            help="Select up to 15 variables for comprehensive analysis",
        )

        if not selected_exog:
            st.warning("Please select at least one exogenous variable.")
            return

        # Prepare exog data
        try:
            exog_data = st.session_state.original_data[selected_exog].copy()
            exog_data.index = data.index
        except Exception as e:
            st.error(f"Error preparing exog data: {e}")
            return

        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Relationships Analysis",
                "🔧 Preprocessing Analysis",
                "🧪 Advanced Diagnostics",
                "📈 Feature Selection",
            ]
        )

        with tab1:
            st.subheader("📊 Feature Relationships Analysis")

            if st.button("Analyze Relationships", key="analyze_relationships"):
                with st.spinner("Analyzing feature relationships..."):
                    try:
                        relationships = analyze_feature_relationships(exog_data, data)

                        # Correlation analysis
                        st.write("**Correlation Analysis:**")
                        corr_data = []
                        for var, corr in relationships["correlations"].items():
                            corr_data.append(
                                {
                                    "Variable": var,
                                    "Correlation": corr,
                                    "Abs Correlation": abs(corr),
                                    "Strength": "Strong"
                                    if abs(corr) > 0.5
                                    else "Medium"
                                    if abs(corr) > 0.3
                                    else "Weak",
                                }
                            )

                        corr_df = pd.DataFrame(corr_data).sort_values(
                            "Abs Correlation", ascending=False
                        )
                        st.dataframe(corr_df)

                        # Visualization
                        fig = px.bar(
                            corr_df,
                            x="Variable",
                            y="Correlation",
                            color="Strength",
                            title="Feature Correlations with Target",
                            color_discrete_map={
                                "Strong": "#ff4444",
                                "Medium": "#ffaa44",
                                "Weak": "#44ff44",
                            },
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Recommendations
                        if relationships.get("recommendations"):
                            st.write("**🎯 Recommendations:**")
                            for rec in relationships["recommendations"]:
                                st.write(f"• {rec}")

                    except Exception as e:
                        st.error(f"Relationship analysis failed: {e}")

        with tab2:
            st.subheader("🔧 Preprocessing Analysis")

            # Preprocessing options
            col1, col2 = st.columns(2)
            with col1:
                preprocessing_method = st.selectbox(
                    "Preprocessing method",
                    ["auto", "robust", "standard", "minmax", "none"],
                    key="preprocessing_analysis",
                )

                outlier_method = st.selectbox(
                    "Outlier detection",
                    ["iqr", "zscore", "modified_zscore"],
                    key="outlier_analysis",
                )

            with col2:
                missing_strategy = st.selectbox(
                    "Missing values strategy",
                    ["interpolate", "mean", "median", "knn", "ffill", "bfill"],
                    key="missing_analysis",
                )

                detect_multicollinearity = st.checkbox(
                    "Detect multicollinearity", value=True, key="multicoll_analysis"
                )
                stationarity_test = st.checkbox(
                    "Test stationarity", value=True, key="stationarity_analysis"
                )

            if st.button("Run Preprocessing Analysis", key="run_preprocessing"):
                with st.spinner("Running preprocessing analysis..."):
                    try:
                        # Create advanced preprocessor
                        preprocessor = ExogenousPreprocessor(
                            method=preprocessing_method,
                            handle_outliers=False,  # Temporaneamente disabilitato per compatibilità
                            outlier_method=outlier_method,
                            missing_strategy=missing_strategy,
                            detect_multicollinearity=False,  # Temporaneamente disabilitato per compatibilità  
                            stationarity_test=False,  # Temporaneamente disabilitato per compatibilità
                        )

                        # Fit and transform
                        exog_processed = preprocessor.fit_transform(exog_data)

                        # Show results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Features", len(exog_data.columns))
                        with col2:
                            st.metric("Processed Features", len(exog_processed.columns))
                        with col3:
                            reduction_pct = (
                                1 - len(exog_processed.columns) / len(exog_data.columns)
                            ) * 100
                            st.metric("Reduction", f"{reduction_pct:.1f}%")

                        # Detailed report
                        report = preprocessor.get_preprocessing_report()

                        with st.expander("📋 Detailed Preprocessing Report"):
                            st.write("**Configuration:**")
                            st.json(report["configuration"])

                            if report["transformations_applied"]:
                                st.write("**Transformations Applied:**")
                                for transformation in report["transformations_applied"]:
                                    st.write(f"• {transformation}")

                            if report["multicollinear_features"]:
                                st.write("**Removed Multicollinear Features:**")
                                for feat in report["multicollinear_features"]:
                                    st.write(f"• {feat}")

                        # Store preprocessed data
                        st.session_state.exog_processed = exog_processed
                        st.session_state.exog_preprocessor = preprocessor

                    except Exception as e:
                        st.error(f"Preprocessing analysis failed: {e}")

        with tab3:
            st.subheader("🧪 Advanced Diagnostics")

            if st.button("Run Advanced Diagnostics", key="run_diagnostics"):
                with st.spinner("Running comprehensive diagnostics..."):
                    try:
                        # Create diagnostics engine
                        diagnostics = ExogDiagnostics(
                            max_lag=min(7, len(data) // 10), significance_level=0.05
                        )

                        # Run full diagnostic suite
                        diagnostic_results = diagnostics.full_diagnostic_suite(
                            target_series=data, exog_data=exog_data
                        )

                        # Display summary
                        summary = diagnostic_results.get("summary", {})
                        st.write("**📊 Diagnostic Summary:**")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Overall Assessment",
                                summary.get("overall_assessment", "unknown").upper(),
                            )
                        with col2:
                            st.metric("Features Analyzed", summary.get("feature_count", 0))
                        with col3:
                            st.metric("Critical Issues", len(summary.get("critical_issues", [])))

                        # Detailed results in expanders
                        with st.expander("🔍 Stationarity Analysis"):
                            stationarity = diagnostic_results.get("stationarity", {})
                            stationary_count = sum(
                                1
                                for result in stationarity.values()
                                if result.get("consensus") == "stationary"
                            )

                            st.write(
                                f"**Stationary variables: {stationary_count}/{len(stationarity)}**"
                            )

                            for var, result in stationarity.items():
                                if "consensus" in result:
                                    status = "✅" if result["consensus"] == "stationary" else "❌"
                                    st.write(f"{status} **{var}**: {result['consensus']}")

                        with st.expander("🔗 Causality Analysis"):
                            causality = diagnostic_results.get("causality", {})
                            causal_vars = [
                                var
                                for var, result in causality.items()
                                if result.get("is_causal", False)
                            ]

                            st.write(f"**Causal variables found: {len(causal_vars)}**")

                            for var, result in causality.items():
                                if "is_causal" in result:
                                    status = "🎯" if result["is_causal"] else "◯"
                                    strength = result.get("strength", "unknown")
                                    st.write(
                                        f"{status} **{var}**: {result['interpretation']} ({strength})"
                                    )

                        with st.expander("📈 Feature Importance"):
                            importance = diagnostic_results.get("feature_importance", {})
                            top_features = importance.get("top_features", [])[:5]

                            st.write("**Top Important Features:**")
                            for i, feat in enumerate(top_features, 1):
                                st.write(f"{i}. {feat}")

                        # Recommendations
                        recommendations = diagnostic_results.get("recommendations", [])
                        if recommendations:
                            st.write("**🎯 Recommendations:**")
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"{i}. {rec}")

                        # Store diagnostic results
                        st.session_state.diagnostic_results = diagnostic_results

                    except Exception as e:
                        st.error(f"Advanced diagnostics failed: {e}")

        with tab4:
            st.subheader("📈 Feature Selection Preview")

            # Feature selection parameters
            col1, col2 = st.columns(2)
            with col1:
                max_features = st.number_input(
                    "Max features to select",
                    min_value=2,
                    max_value=len(selected_exog),
                    value=min(8, len(selected_exog)),
                    key="feature_selection_max",
                )
                selection_method = st.selectbox(
                    "Selection method",
                    ["stepwise", "lasso", "elastic_net", "f_test"],
                    key="feature_selection_method",
                )

            with col2:
                feature_engineering = st.multiselect(
                    "Feature engineering",
                    ["lags", "differences", "interactions"],
                    default=["lags"],
                    key="feature_selection_engineering",
                )

                validation_split = st.slider(
                    "Validation split", 0.1, 0.3, 0.2, key="feature_selection_validation"
                )

            if st.button("Preview Feature Selection", key="preview_feature_selection"):
                with st.spinner("Running feature selection preview..."):
                    try:
                        # Create temporary selector for preview
                        selector = SARIMAXAutoSelector(
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12),
                            max_features=max_features,
                            selection_method=selection_method,
                            feature_engineering=feature_engineering,
                            preprocessing_method="robust",
                            validation_split=validation_split,
                        )

                        # Dry run - just feature selection part (più veloce)
                        st.info(
                            "This is a preview - full training available in 'Model Training' section"
                        )

                        # Show what would be selected (simulato)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Input Features", len(selected_exog))
                        with col2:
                            st.metric("Max Features", max_features)
                        with col3:
                            st.metric("Method", selection_method.upper())

                        # Simulate feature selection results
                        st.write("**Feature Selection Configuration:**")
                        config_info = {
                            "Selection Method": selection_method,
                            "Max Features": max_features,
                            "Feature Engineering": feature_engineering,
                            "Validation Split": validation_split,
                            "Input Variables": selected_exog,
                        }
                        st.json(config_info)

                        st.info(
                            "💡 Run 'Advanced SARIMAX' in the Model Training page for full feature selection and training"
                        )

                    except Exception as e:
                        st.error(f"Feature selection preview failed: {e}")


def main():
    """Main function to run the dashboard."""
    dashboard = ARIMADashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
