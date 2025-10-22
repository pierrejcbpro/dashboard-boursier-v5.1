# -*- coding: utf-8 -*-
import streamlit as st, pandas as pd, altair as alt, yfinance as yf
from lib import compute_metrics, news_summary, ai_flash_note, decision_label_from_row

st.title("🔎 Recherche 90 jours — IA & Actus")

ticker = st.text_input("Ticker / ISIN / WKN (ex: AIR.PA, AAPL, TTE.PA)").strip().upper()
if not ticker: st.stop()

try:
    h = yf.download(ticker, period="110d", interval="1d", auto_adjust=False, progress=False)
except Exception as e:
    st.error(f"Impossible de récupérer l'historique: {e}"); st.stop()

if h.empty:
    st.warning("Aucune donnée pour ce ticker."); st.stop()

d = h.reset_index()
base = alt.Chart(d).mark_line(color="#1e88e5").encode(
    x=alt.X("Date:T", title=""), y=alt.Y("Close:Q", title="Cours"),
    tooltip=["Date:T","Close:Q"]
).properties(title=f"{ticker} — 90 jours", height=320)
st.altair_chart(base, use_container_width=True)

m = compute_metrics(h.assign(Ticker=ticker))
if m.empty:
    st.info("Indicateurs indisponibles."); st.stop()

row = m.tail(1).iloc[0]
txt, _, items = news_summary(ticker, ticker)
note = ai_flash_note(ticker, ticker, row)
dec = decision_label_from_row(row, held=False)

st.subheader("Analyse IA")
st.write(f"**Décision** : {dec}")
st.write(note)

st.subheader("📰 Actus récentes")
if items:
    for t,u in items[:5]:
        st.markdown(f"- [{t}]({u})")
else:
    st.caption("Aucune actualité saillante.")
