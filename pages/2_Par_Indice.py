# -*- coding: utf-8 -*-
import streamlit as st, pandas as pd, altair as alt
from lib import members, fetch_prices, compute_metrics, news_summary, ai_flash_note, decision_label_from_row

st.title("📊 Analyse par Indice — IA & Pastilles")

idx = st.selectbox("Indice", ["CAC 40","DAX 40","NASDAQ 100","S&P 500","Dow Jones"], index=0)
periode = st.radio("Période", ["Jour","7 jours","30 jours"], index=0, horizontal=True)
value_col = {"Jour":"pct_1d","7 jours":"pct_7d","30 jours":"pct_30d"}[periode]

mem = members(idx)
if mem.empty: st.warning("Constituants introuvables."); st.stop()

px = fetch_prices(mem["ticker"].tolist(), days=120)
met = compute_metrics(px).merge(mem, left_on="Ticker", right_on="ticker", how="left")
if met.empty: st.warning("Prix indisponibles."); st.stop()

top5 = met.sort_values(value_col, ascending=False).head(5)
low5 = met.sort_values(value_col, ascending=True).head(5)

st.subheader("Top/Low (tableau)")
st.dataframe(top5[["name","Ticker","Close",value_col]], use_container_width=True, hide_index=True)
st.dataframe(low5[["name","Ticker","Close",value_col]], use_container_width=True, hide_index=True)

st.subheader("Sélection IA (3 + 3)")
subset = pd.concat([top5.head(3), low5.head(3)], ignore_index=True)
rows=[]
for _,r in subset.iterrows():
    name=r.get("name", r.get("Ticker"))
    tick=r.get("Ticker","")
    txt,_,items=news_summary(str(name), tick)
    note=ai_flash_note(str(name), tick, r)
    dec=decision_label_from_row(r, held=False)
    news_links=" • ".join([f"[{t}]({u})" for t,u in items[:2]]) if items else "—"
    rows.append({"Nom":name,"Ticker":tick,"Décision":dec,"Note IA":note,"Actu":news_links})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
