"""
Utilità per Dashboard Mobile Responsive.

Gestisce layout adattivi e rilevamento dispositivi per Streamlit.
"""

import streamlit as st
from typing import Tuple, Dict, Any
import pandas as pd


class MobileResponsiveManager:
    """Gestore layout responsive per dashboard mobile."""

    def __init__(self):
        """Inizializza il manager responsive."""
        self.screen_width = self._get_screen_width()
        self.is_mobile = self._detect_mobile_device()
        self.is_tablet = self._detect_tablet_device()

    def _get_screen_width(self) -> int:
        """Rileva larghezza schermo tramite JavaScript."""
        # Simula rilevamento per ora, in produzione usare st.components
        return st.session_state.get("screen_width", 1920)

    def _detect_mobile_device(self) -> bool:
        """Rileva se il dispositivo è mobile."""
        return self.screen_width <= 768

    def _detect_tablet_device(self) -> bool:
        """Rileva se il dispositivo è tablet."""
        return 768 < self.screen_width <= 1024

    def get_layout_config(self) -> Dict[str, Any]:
        """Restituisce configurazione layout per dispositivo."""
        if self.is_mobile:
            return {
                "columns": 1,
                "chart_height": 300,
                "sidebar_state": "collapsed",
                "font_size": "small",
                "spacing": "compact",
            }
        elif self.is_tablet:
            return {
                "columns": 2,
                "chart_height": 400,
                "sidebar_state": "auto",
                "font_size": "medium",
                "spacing": "normal",
            }
        else:
            return {
                "columns": 3,
                "chart_height": 500,
                "sidebar_state": "expanded",
                "font_size": "normal",
                "spacing": "wide",
            }

    def create_responsive_columns(self, ratios: list = None) -> list:
        """Crea colonne responsive basate sul dispositivo."""
        config = self.get_layout_config()
        num_cols = config["columns"]

        if num_cols == 1:
            return [st.container()]
        elif num_cols == 2:
            if ratios:
                return st.columns(ratios[:2])
            return st.columns(2)
        else:
            if ratios:
                return st.columns(ratios[:3])
            return st.columns([2, 1, 1])

    def apply_responsive_css(self):
        """Applica CSS responsive alla dashboard."""
        config = self.get_layout_config()

        css = f"""
        <style>
        /* Mobile Responsive CSS */
        .main .block-container {{
            padding: {"0.5rem" if self.is_mobile else "1rem"};
        }}
        
        .metric-card {{
            font-size: {"0.8rem" if self.is_mobile else "1rem"};
            padding: {"0.5rem" if self.is_mobile else "1rem"};
        }}
        
        .stSelectbox > div > div {{
            font-size: {"0.85rem" if self.is_mobile else "1rem"};
        }}
        
        .stButton > button {{
            width: {"100%" if self.is_mobile else "auto"};
            font-size: {"0.85rem" if self.is_mobile else "1rem"};
        }}
        
        .chart-container {{
            height: {config["chart_height"]}px;
        }}
        
        /* Tablet specific */
        @media (min-width: 769px) and (max-width: 1024px) {{
            .stSelectbox {{
                margin-bottom: 0.5rem;
            }}
        }}
        
        /* Mobile specific */
        @media (max-width: 768px) {{
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.2rem;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                font-size: 0.8rem;
                padding: 0.3rem 0.5rem;
            }}
            
            .stMetric {{
                background: #f0f2f6;
                padding: 0.5rem;
                border-radius: 0.3rem;
                margin: 0.2rem 0;
            }}
            
            .stDataFrame {{
                font-size: 0.75rem;
            }}
        }}
        
        /* Hide non-essential elements on mobile */
        @media (max-width: 600px) {{
            .element-container:has(.stAlert) {{
                display: none;
            }}
        }}
        </style>
        """

        st.markdown(css, unsafe_allow_html=True)

    def create_mobile_navigation(self, sections: list) -> str:
        """Crea navigazione mobile ottimizzata."""
        if self.is_mobile:
            # Tab orizzontali compatte per mobile
            selected = st.selectbox("Sezione", sections, label_visibility="collapsed")
        else:
            # Tab standard per desktop
            selected = st.radio("Navigazione", sections, horizontal=True)

        return selected

    def display_metrics_responsive(self, metrics: Dict[str, Any]):
        """Visualizza metriche in formato responsive."""
        config = self.get_layout_config()

        if config["columns"] == 1:
            # Mobile: metriche impilate
            for key, value in metrics.items():
                st.metric(label=key, value=value.get("value", 0), delta=value.get("delta", None))
        else:
            # Desktop/Tablet: metriche in colonne
            cols = self.create_responsive_columns()
            metrics_items = list(metrics.items())

            for i, col in enumerate(cols):
                if i < len(metrics_items):
                    key, value = metrics_items[i]
                    with col:
                        st.metric(
                            label=key, value=value.get("value", 0), delta=value.get("delta", None)
                        )

    def create_responsive_chart(self, fig, title: str = ""):
        """Crea grafico responsive."""
        config = self.get_layout_config()

        # Aggiorna dimensioni grafico
        fig.update_layout(
            height=config["chart_height"],
            title=dict(text=title, font=dict(size=16 if self.is_mobile else 20)),
            legend=dict(
                orientation="h" if self.is_mobile else "v",
                yanchor="bottom" if self.is_mobile else "top",
                y=1.02 if self.is_mobile else 1,
                xanchor="center" if self.is_mobile else "left",
                x=0.5 if self.is_mobile else 0,
            ),
            margin=dict(
                l=20 if self.is_mobile else 40,
                r=20 if self.is_mobile else 40,
                t=60 if self.is_mobile else 80,
                b=40 if self.is_mobile else 60,
            ),
        )

        # Font più piccolo per mobile
        if self.is_mobile:
            fig.update_xaxes(tickfont=dict(size=10))
            fig.update_yaxes(tickfont=dict(size=10))

        return fig

    def create_responsive_dataframe(self, df: pd.DataFrame, max_rows: int = None):
        """Visualizza DataFrame responsive."""
        if self.is_mobile:
            # Mobile: mostra meno righe, colonne essenziali
            display_df = df.head(max_rows or 10)

            # Riduci colonne se troppe
            if len(display_df.columns) > 4:
                essential_cols = display_df.columns[:4]
                display_df = display_df[essential_cols]
                st.caption(
                    f"Mostrate {len(essential_cols)}/{len(df.columns)} colonne per spazio limitato"
                )

            st.dataframe(display_df, use_container_width=True, height=200)
        else:
            # Desktop: DataFrame completo
            st.dataframe(df.head(max_rows or 20), use_container_width=True)


def init_responsive_dashboard():
    """Inizializza dashboard responsive."""
    responsive_manager = MobileResponsiveManager()
    responsive_manager.apply_responsive_css()

    # Store nel session state per uso globale
    st.session_state.responsive_manager = responsive_manager

    return responsive_manager


def get_responsive_manager():
    """Ottiene il manager responsive dalla session."""
    if "responsive_manager" not in st.session_state:
        return init_responsive_dashboard()
    return st.session_state.responsive_manager
