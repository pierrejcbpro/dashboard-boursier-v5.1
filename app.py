# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from lib import fetch_all_markets

st.set_page_config(page_title="Dash Boursier v5.1", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.badge {padding:4px 8px; border-radius:12px; font-weight:600; font-size:0.85rem;}
.buy {background:#e8f5e9; color:#0b8f3a;}
.sell {background:#ffebee; color:#b71c1c;}
.hold {background:#fff3e0; color:#e65100;}
.watch {background:#e3f2fd; color:#0d47a1;}
.avoid {background:#fce4ec; color:#ad1457;}
</style>
""", unsafe_allow_html=True)

st.title("💹 Dash Boursier v5.1 — Accueil")
st.caption("Version visuelle & IA — compatible Streamlit Cloud")

try:
    data = fetch_all_markets([("CAC 40",""),("DAX 40",""),("NASDAQ 100",""),("S&P 500",""),("Dow Jones","")], days_hist=30)
    if not data.empty and "pct_1d" in data:
        top = data.sort_values("pct_1d", ascending=False).head(5)[["Ticker","name","Close","pct_1d","Indice"]]
        low = data.sort_values("pct_1d", ascending=True).head(5)[["Ticker","name","Close","pct_1d","Indice"]]
        c1,c2=st.columns(2)
        with c1: st.subheader("Top 5 (Jour)"); st.dataframe(top, use_container_width=True, hide_index=True)
        with c2: st.subheader("Bottom 5 (Jour)"); st.dataframe(low, use_container_width=True, hide_index=True)
    else:
        st.info("Les données Yahoo Finance sont momentanément indisponibles.")
except Exception as e:
    st.warning(f"Problème de chargement initial: {e}")
