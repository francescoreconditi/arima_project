"""
Batch Forecasting Web UI - Streamlit Dashboard
Interfaccia web per business users per portfolio analysis automatica

Autore: Claude Code
Data: 2025-09-02
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import json
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Import del nostro sistema
try:
    from arima_forecaster.automl.batch_processor import (
        BatchForecastProcessor,
        BatchProgress,
        BatchTaskResult,
    )
    from arima_forecaster.automl.auto_selector import AutoForecastSelector
    from arima_forecaster.utils.translations import translate as _
except ImportError as e:
    st.error(f"Errore import moduli: {e}")
    st.stop()


class BatchForecastDashboard:
    """
    Dashboard Streamlit per batch forecasting con AutoML
    """

    def __init__(self):
        self.processor = BatchForecastProcessor()
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self):
        """Configurazione pagina Streamlit"""
        st.set_page_config(
            page_title="ARIMA AutoML Batch Forecasting",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS
        st.markdown(
            """
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .success-box {
            padding: 1rem;
            background-color: #d4edda;
            border-color: #c3e6cb;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        .info-box {
            padding: 1rem;
            background-color: #d1ecf1;
            border-color: #bee5eb;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        .warning-box {
            padding: 1rem;
            background-color: #fff3cd;
            border-color: #ffeaa7;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def initialize_session_state(self):
        """Inizializza session state"""
        if "uploaded_data" not in st.session_state:
            st.session_state.uploaded_data = {}
        if "batch_results" not in st.session_state:
            st.session_state.batch_results = {}
        if "processing_status" not in st.session_state:
            st.session_state.processing_status = "idle"
        if "progress_data" not in st.session_state:
            st.session_state.progress_data = None

    def render_header(self):
        """Render header principale"""
        st.markdown(
            '<h1 class="main-header">📈 ARIMA AutoML Batch Forecasting</h1>', unsafe_allow_html=True
        )
        st.markdown(
            """
        <div class="info-box">
        <strong>🚀 One-Click Portfolio Forecasting</strong><br>
        Carica i tuoi dati CSV, il sistema AutoML seleziona automaticamente il modello ottimale per ogni serie temporale.
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        """Render sidebar configurazioni"""
        st.sidebar.header("⚙️ Configurazioni")

        st.sidebar.subheader("📊 Parametri Forecasting")
        forecast_steps = st.sidebar.slider(
            "Periodi da predire",
            min_value=7,
            max_value=365,
            value=30,
            step=7,
            help="Numero di giorni/periodi futuri da prevedere",
        )

        validation_split = (
            st.sidebar.slider(
                "Split validazione (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentuale dati per validazione modello",
            )
            / 100
        )

        st.sidebar.subheader("🔧 Parametri AutoML")
        max_models = st.sidebar.selectbox(
            "Max modelli da testare",
            options=[3, 5, 8, 10],
            index=1,
            help="Numero massimo di modelli da valutare per serie",
        )

        timeout_per_model = st.sidebar.slider(
            "Timeout per modello (sec)",
            min_value=30,
            max_value=300,
            value=60,
            step=30,
            help="Tempo massimo training per modello",
        )

        st.sidebar.subheader("⚡ Performance")
        enable_parallel = st.sidebar.checkbox(
            "Elaborazione parallela", value=True, help="Usa più core CPU per velocizzare"
        )

        max_workers = st.sidebar.slider(
            "Worker paralleli",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            disabled=not enable_parallel,
            help="Numero di processi paralleli",
        )

        # Salva configurazioni in session state
        st.session_state.config = {
            "forecast_steps": forecast_steps,
            "validation_split": validation_split,
            "max_models_to_try": max_models,
            "timeout_per_model": timeout_per_model,
            "enable_parallel": enable_parallel,
            "max_workers": max_workers if enable_parallel else 1,
        }

    def render_data_upload(self):
        """Render sezione upload dati"""
        st.header("📁 Caricamento Dati")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Metodo 1: File Singolo")
            uploaded_file = st.file_uploader(
                "Carica file CSV con dati serie temporali",
                type=["csv"],
                help="File con colonne: date, value, series_name (opzionale)",
            )

            if uploaded_file:
                self.process_single_file(uploaded_file)

        with col2:
            st.subheader("Metodo 2: Multipli File")
            uploaded_files = st.file_uploader(
                "Carica multipli CSV",
                type=["csv"],
                accept_multiple_files=True,
                help="Un file per serie temporale",
            )

            if uploaded_files:
                self.process_multiple_files(uploaded_files)

        # Mostra dati caricati
        if st.session_state.uploaded_data:
            st.subheader("📋 Dati Caricati")
            self.show_uploaded_data_summary()

    def process_single_file(self, uploaded_file):
        """Process singolo file CSV"""
        try:
            # Leggi CSV
            df = pd.read_csv(uploaded_file)

            # Rileva formato
            if "series_name" in df.columns:
                # Formato multi-serie
                series_dict = {}
                for name in df["series_name"].unique():
                    series_data = df[df["series_name"] == name]
                    if "date" in series_data.columns:
                        series_data = series_data.set_index("date")["value"]
                    else:
                        series_data = series_data["value"]
                    series_dict[name] = series_data

                st.session_state.uploaded_data.update(series_dict)
                st.success(f"✅ Caricate {len(series_dict)} serie temporali")

            else:
                # Formato singola serie
                series_name = uploaded_file.name.replace(".csv", "")

                if "date" in df.columns and "value" in df.columns:
                    series_data = df.set_index("date")["value"]
                elif len(df.columns) >= 2:
                    # Prima colonna = date, seconda = value
                    series_data = df.set_index(df.columns[0])[df.columns[1]]
                else:
                    # Solo valori
                    series_data = df.iloc[:, 0]

                st.session_state.uploaded_data[series_name] = series_data
                st.success(f"✅ Caricata serie '{series_name}' con {len(series_data)} osservazioni")

        except Exception as e:
            st.error(f"❌ Errore caricamento file: {str(e)}")

    def process_multiple_files(self, uploaded_files):
        """Process multipli file CSV"""
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                series_name = file.name.replace(".csv", "")

                if "date" in df.columns and "value" in df.columns:
                    series_data = df.set_index("date")["value"]
                elif len(df.columns) >= 2:
                    series_data = df.set_index(df.columns[0])[df.columns[1]]
                else:
                    series_data = df.iloc[:, 0]

                st.session_state.uploaded_data[series_name] = series_data

            except Exception as e:
                st.warning(f"⚠️ Errore file {file.name}: {str(e)}")

        if uploaded_files:
            st.success(f"✅ Processati {len(uploaded_files)} file")

    def show_uploaded_data_summary(self):
        """Mostra summary dati caricati"""
        summary_data = []

        for name, series in st.session_state.uploaded_data.items():
            summary_data.append(
                {
                    "Serie": name,
                    "Osservazioni": len(series),
                    "Media": f"{np.mean(series):.2f}",
                    "Min": f"{np.min(series):.2f}",
                    "Max": f"{np.max(series):.2f}",
                    "Zeri (%)": f"{(series == 0).sum() / len(series) * 100:.1f}%",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Cancella Tutti", type="secondary"):
                st.session_state.uploaded_data = {}
                st.rerun()

        with col2:
            if st.button("📊 Preview Dati", type="secondary"):
                self.show_data_preview()

        with col3:
            if st.button("🚀 Avvia Batch Forecasting", type="primary"):
                if len(st.session_state.uploaded_data) > 0:
                    self.start_batch_processing()
                else:
                    st.warning("⚠️ Carica almeno una serie temporale")

    def show_data_preview(self):
        """Mostra preview dati"""
        st.subheader("👀 Preview Dati")

        # Seleziona serie per preview
        selected_series = st.selectbox(
            "Seleziona serie da visualizzare:", options=list(st.session_state.uploaded_data.keys())
        )

        if selected_series:
            series_data = st.session_state.uploaded_data[selected_series]

            # Stats base
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Osservazioni", len(series_data))
            with col2:
                st.metric("Media", f"{np.mean(series_data):.2f}")
            with col3:
                st.metric("Std Dev", f"{np.std(series_data):.2f}")
            with col4:
                st.metric("Zeri (%)", f"{(series_data == 0).sum() / len(series_data) * 100:.1f}%")

            # Grafico time series
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=series_data.values,
                    mode="lines",
                    name=selected_series,
                    line=dict(color="#1f77b4", width=2),
                )
            )
            fig.update_layout(
                title=f"Serie Temporale: {selected_series}",
                xaxis_title="Periodo",
                yaxis_title="Valore",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Pattern detection preview
            try:
                from arima_forecaster.automl.auto_selector import SeriesPatternDetector

                detector = SeriesPatternDetector()
                pattern, confidence = detector.detect_pattern(series_data)

                st.info(f"🔍 **Pattern Rilevato**: {pattern.value} (Confidence: {confidence:.1%})")

            except Exception as e:
                st.warning(f"Pattern detection non disponibile: {str(e)}")

    def start_batch_processing(self):
        """Avvia batch processing"""
        st.session_state.processing_status = "running"
        st.session_state.batch_results = {}

        # Progress containers
        progress_container = st.container()
        with progress_container:
            st.subheader("⚡ Batch Processing in Corso...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()

        try:
            # Configura processor
            self.processor.set_config(
                enable_parallel=st.session_state.config["enable_parallel"],
                max_workers=st.session_state.config["max_workers"],
                validation_split=st.session_state.config["validation_split"],
                max_models_to_try=st.session_state.config["max_models_to_try"],
                timeout_per_model=st.session_state.config["timeout_per_model"],
                verbose=True,
            )

            # Avvia processing con callback per progress
            def progress_callback(progress: BatchProgress):
                # Update progress bar
                completion = progress.completed_tasks / progress.total_tasks
                progress_bar.progress(completion)

                # Update status
                status_text.markdown(f"""
                **Progress**: {progress.completed_tasks}/{progress.total_tasks} serie completate  
                **Successi**: {progress.successful_tasks} | **Errori**: {progress.failed_tasks}  
                **Tempo**: {progress.elapsed_time:.1f}s | **ETA**: {progress.estimated_completion:.1f}s
                """)

                # Update results table in real-time
                if progress.completed_tasks > 0:
                    self.update_results_preview(results_container, progress)

            # Esegui batch processing
            results = self.processor.fit_batch(
                st.session_state.uploaded_data,
                forecast_steps=st.session_state.config["forecast_steps"],
                progress_callback=progress_callback,
            )

            # Salva risultati
            st.session_state.batch_results = results
            st.session_state.processing_status = "completed"

            # Success message
            successful_count = sum(1 for r in results.values() if r.status == "success")
            st.success(
                f"✅ **Batch Processing Completato!**  \n{successful_count}/{len(results)} serie elaborate con successo"
            )

            # Mostra risultati
            self.render_batch_results()

        except Exception as e:
            st.session_state.processing_status = "error"
            st.error(f"❌ **Errore durante processing**: {str(e)}")

    def update_results_preview(self, container, progress: BatchProgress):
        """Update real-time results preview"""
        if not st.session_state.batch_results:
            return

        preview_data = []
        for name, result in st.session_state.batch_results.items():
            if result.status == "success":
                preview_data.append(
                    {
                        "Serie": name,
                        "Modello": result.explanation.recommended_model,
                        "Confidence": f"{result.explanation.confidence_score:.1%}",
                        "Pattern": result.explanation.pattern_detected,
                        "Tempo": f"{result.training_time:.1f}s",
                    }
                )

        if preview_data:
            preview_df = pd.DataFrame(preview_data)
            container.dataframe(preview_df, use_container_width=True)

    def render_batch_results(self):
        """Render risultati batch processing"""
        if not st.session_state.batch_results:
            return

        st.header("📊 Risultati Batch Processing")

        # Summary metrics
        total_series = len(st.session_state.batch_results)
        successful_series = sum(
            1 for r in st.session_state.batch_results.values() if r.status == "success"
        )
        failed_series = total_series - successful_series
        avg_time = np.mean(
            [
                r.training_time
                for r in st.session_state.batch_results.values()
                if r.status == "success"
            ]
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Serie Totali", total_series)
        with col2:
            st.metric(
                "Successi", successful_series, delta=f"{successful_series / total_series:.1%}"
            )
        with col3:
            st.metric("Errori", failed_series)
        with col4:
            st.metric("Tempo Medio", f"{avg_time:.1f}s")

        # Results table
        self.show_results_table()

        # Visualizations
        self.show_results_charts()

        # Export options
        self.show_export_options()

    def show_results_table(self):
        """Mostra tabella risultati dettagliata"""
        st.subheader("📋 Dettagli Risultati")

        results_data = []
        for name, result in st.session_state.batch_results.items():
            if result.status == "success":
                results_data.append(
                    {
                        "Serie": name,
                        "Modello": result.explanation.recommended_model,
                        "Confidence": result.explanation.confidence_score,
                        "Pattern": result.explanation.pattern_detected,
                        "Accuracy": result.metrics.get("mae", 0) if result.metrics else 0,
                        "Forecast Medio": f"{np.mean(result.forecast):.2f}"
                        if result.forecast is not None
                        else "N/A",
                        "Tempo (s)": result.training_time,
                        "Status": "✅ Success",
                    }
                )
            else:
                results_data.append(
                    {
                        "Serie": name,
                        "Modello": "N/A",
                        "Confidence": 0,
                        "Pattern": "N/A",
                        "Accuracy": 0,
                        "Forecast Medio": "N/A",
                        "Tempo (s)": result.training_time,
                        "Status": f"❌ {result.error}",
                    }
                )

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

        # Filtri e ordinamento
        col1, col2 = st.columns(2)
        with col1:
            model_filter = st.multiselect(
                "Filtra per modello:",
                options=results_df["Modello"].unique(),
                default=results_df["Modello"].unique(),
            )

        with col2:
            sort_by = st.selectbox(
                "Ordina per:", options=["Serie", "Confidence", "Accuracy", "Tempo (s)"], index=1
            )

        # Apply filters
        if model_filter:
            filtered_df = results_df[results_df["Modello"].isin(model_filter)]
            filtered_df = filtered_df.sort_values(sort_by, ascending=False)
            st.dataframe(filtered_df, use_container_width=True, key="filtered_results")

    def show_results_charts(self):
        """Mostra grafici risultati"""
        st.subheader("📈 Visualizzazioni")

        successful_results = {
            k: v for k, v in st.session_state.batch_results.items() if v.status == "success"
        }

        if not successful_results:
            st.warning("Nessun risultato valido per visualizzazioni")
            return

        # Chart tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Modelli Selezionati", "📊 Performance", "🔍 Forecasts"])

        with tab1:
            # Model distribution
            model_counts = {}
            for result in successful_results.values():
                model = result.explanation.recommended_model
                model_counts[model] = model_counts.get(model, 0) + 1

            fig = px.pie(
                values=list(model_counts.values()),
                names=list(model_counts.keys()),
                title="Distribuzione Modelli Selezionati",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Confidence vs Training Time
            confidence_data = []
            time_data = []
            names = []

            for name, result in successful_results.items():
                confidence_data.append(result.explanation.confidence_score)
                time_data.append(result.training_time)
                names.append(name)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=confidence_data,
                    mode="markers+text",
                    text=names,
                    textposition="top center",
                    marker=dict(
                        size=10, color=confidence_data, colorscale="Viridis", showscale=True
                    ),
                    name="Serie",
                )
            )
            fig.update_layout(
                title="Confidence vs Training Time",
                xaxis_title="Tempo Training (s)",
                yaxis_title="Confidence Score",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Forecast comparison
            selected_series = st.selectbox(
                "Seleziona serie per dettaglio forecast:", options=list(successful_results.keys())
            )

            if selected_series:
                result = successful_results[selected_series]
                original_data = st.session_state.uploaded_data[selected_series]

                fig = go.Figure()

                # Original data
                fig.add_trace(
                    go.Scatter(
                        y=original_data.values[-100:],  # Last 100 points
                        mode="lines",
                        name="Dati Storici",
                        line=dict(color="blue", width=2),
                    )
                )

                # Forecast
                if result.forecast is not None:
                    forecast_start = len(original_data)
                    forecast_x = list(range(forecast_start, forecast_start + len(result.forecast)))

                    fig.add_trace(
                        go.Scatter(
                            x=forecast_x,
                            y=result.forecast,
                            mode="lines",
                            name="Forecast",
                            line=dict(color="red", width=2, dash="dash"),
                        )
                    )

                fig.update_layout(
                    title=f"Forecast: {selected_series} - {result.explanation.recommended_model}",
                    xaxis_title="Periodo",
                    yaxis_title="Valore",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Result details
                st.info(f"""
                **Modello**: {result.explanation.recommended_model}  
                **Confidence**: {result.explanation.confidence_score:.1%}  
                **Pattern**: {result.explanation.pattern_detected}  
                **Why Chosen**: {result.explanation.why_chosen}  
                **Business Recommendation**: {result.explanation.business_recommendation}
                """)

    def show_export_options(self):
        """Mostra opzioni export risultati"""
        st.subheader("💾 Export Risultati")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Export CSV", type="secondary"):
                csv_data = self.generate_csv_export()
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"batch_forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        with col2:
            if st.button("📊 Export Excel", type="secondary"):
                excel_data = self.generate_excel_export()
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_data,
                    file_name=f"batch_forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        with col3:
            if st.button("📋 Export Report", type="secondary"):
                report_data = self.generate_report_export()
                st.download_button(
                    label="⬇️ Download Report",
                    data=report_data,
                    file_name=f"batch_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                )

    def generate_csv_export(self) -> str:
        """Genera export CSV"""
        export_data = []

        for name, result in st.session_state.batch_results.items():
            if result.status == "success":
                base_record = {
                    "serie_name": name,
                    "model": result.explanation.recommended_model,
                    "confidence": result.explanation.confidence_score,
                    "pattern": result.explanation.pattern_detected,
                    "why_chosen": result.explanation.why_chosen,
                    "business_recommendation": result.explanation.business_recommendation,
                    "training_time": result.training_time,
                }

                # Add forecast values
                if result.forecast is not None:
                    for i, value in enumerate(result.forecast):
                        record = base_record.copy()
                        record.update({"forecast_period": i + 1, "forecast_value": value})
                        export_data.append(record)
                else:
                    export_data.append(base_record)

        df = pd.DataFrame(export_data)
        return df.to_csv(index=False)

    def generate_excel_export(self) -> bytes:
        """Genera export Excel"""
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = []
            forecast_data = []

            for name, result in st.session_state.batch_results.items():
                if result.status == "success":
                    summary_data.append(
                        {
                            "Serie": name,
                            "Modello": result.explanation.recommended_model,
                            "Confidence": result.explanation.confidence_score,
                            "Pattern": result.explanation.pattern_detected,
                            "Training_Time": result.training_time,
                            "Why_Chosen": result.explanation.why_chosen,
                            "Business_Recommendation": result.explanation.business_recommendation,
                        }
                    )

                    # Forecast data
                    if result.forecast is not None:
                        for i, value in enumerate(result.forecast):
                            forecast_data.append(
                                {"Serie": name, "Period": i + 1, "Forecast_Value": value}
                            )

            # Write sheets
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(forecast_data).to_excel(writer, sheet_name="Forecasts", index=False)

        output.seek(0)
        return output.getvalue()

    def generate_report_export(self) -> str:
        """Genera report HTML"""
        successful_results = {
            k: v for k, v in st.session_state.batch_results.items() if v.status == "success"
        }

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Forecast Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #1f77b4; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .results {{ margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 Batch Forecasting Report</h1>
                <p>Generato il: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="summary">
                <h2>📊 Summary</h2>
                <p><strong>Serie Totali:</strong> {len(st.session_state.batch_results)}</p>
                <p><strong>Successi:</strong> {len(successful_results)}</p>
                <p><strong>Tasso Successo:</strong> {len(successful_results) / len(st.session_state.batch_results):.1%}</p>
            </div>
            
            <div class="results">
                <h2>📋 Risultati Dettagliati</h2>
                <table>
                    <tr>
                        <th>Serie</th>
                        <th>Modello</th>
                        <th>Confidence</th>
                        <th>Pattern</th>
                        <th>Raccomandazione Business</th>
                    </tr>
        """

        for name, result in successful_results.items():
            html_content += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{result.explanation.recommended_model}</td>
                        <td>{result.explanation.confidence_score:.1%}</td>
                        <td>{result.explanation.pattern_detected}</td>
                        <td>{result.explanation.business_recommendation}</td>
                    </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        return html_content

    def run(self):
        """Esegui dashboard"""
        self.render_header()
        self.render_sidebar()

        # Main content
        if st.session_state.processing_status == "completed" and st.session_state.batch_results:
            self.render_batch_results()
        else:
            self.render_data_upload()

        # Footer
        st.markdown("---")
        st.markdown("**ARIMA AutoML Batch Forecasting** - Powered by Claude Code")


def main():
    """Main entry point"""
    dashboard = BatchForecastDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
