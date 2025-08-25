"""
Test per verificare il nuovo stile degli alert con contrasto migliorato
"""

import streamlit as st
import pandas as pd

# CSS migliorato
st.markdown("""
<style>
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #000000;
        font-weight: 500;
    }
    .alert-critica {
        background-color: #ffffff; 
        border: 3px solid #dc3545; 
        box-shadow: 0 2px 6px rgba(220, 53, 69, 0.3);
    }
    .alert-alta {
        background-color: #ffffff; 
        border: 3px solid #fd7e14; 
        box-shadow: 0 2px 6px rgba(253, 126, 20, 0.3);
    }
    .alert-media {
        background-color: #ffffff; 
        border: 3px solid #ffc107; 
        box-shadow: 0 2px 6px rgba(255, 193, 7, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Test Stile Alert - Contrasto Migliorato")

st.markdown("### Prima (Problematico)")
st.markdown("""
<div style='padding: 15px; background-color: #ffcccc; border-left: 5px solid #ff0000; margin: 10px 0; border-radius: 5px;'>
    <strong>[CRITICA] Scorte Basse</strong><br>
    Carrozzina Sportiva Titanio: 8 unità rimanenti (minimo: 10)<br>
    <em>Azione suggerita: Ordinare almeno 20 unità</em>
</div>
""", unsafe_allow_html=True)

st.markdown("### Dopo (Corretto)")
st.markdown("""
<div class='alert-box alert-critica'>
    <strong style='color: #dc3545; font-size: 16px;'>[CRITICA] Scorte Basse</strong><br>
    <span style='color: #000000; font-size: 14px;'>Carrozzina Sportiva Titanio: 8 unità rimanenti (minimo: 10)</span><br>
    <em style='color: #333333; font-size: 13px;'>Azione suggerita: Ordinare almeno 20 unità</em>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='alert-box alert-alta'>
    <strong style='color: #fd7e14; font-size: 16px;'>[ALTA] Scorte Basse</strong><br>
    <span style='color: #000000; font-size: 14px;'>Materasso Memory Foam: 15 unità rimanenti (minimo: 20)</span><br>
    <em style='color: #333333; font-size: 13px;'>Azione suggerita: Ordinare almeno 40 unità</em>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='alert-box alert-media'>
    <strong style='color: #ffc107; font-size: 16px;'>[MEDIA] Scorte Basse</strong><br>
    <span style='color: #000000; font-size: 14px;'>Sollevatore Pazienti Elettrico: 6 unità rimanenti (minimo: 8)</span><br>
    <em style='color: #333333; font-size: 13px;'>Azione suggerita: Ordinare almeno 16 unità</em>
</div>
""", unsafe_allow_html=True)

st.success("✅ **Alert corretti**: Sfondo bianco, testo nero, contrasto ottimale!")