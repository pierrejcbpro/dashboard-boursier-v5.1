# -*- coding: utf-8 -*-
import streamlit as st, pandas as pd, numpy as np, altair as alt
from lib import fetch_all_markets, news_summary, ai_flash_note, decision_label_from_row

st.title("🌍 Marché Global — IA & Pastilles")

periode = st.radio("Période", ["Jour","7 jours","30 jours"], index=0, horizontal=True)
days_hist = {"Jour":60,"7 jours":90,"30 jours":150}[periode]
value_col = {"Jour":"pct_1d","7 jours":"pct_7d","30 jours":"pct_30d"}[periode]

MARKETS_AND_WL=[("CAC 40",""),("DAX 40",""),("NASDAQ 100",""),("S&P 500",""),("Dow Jones","")]
data = fetch_all_markets(MARKETS_AND_WL, days_hist=days_hist)
if data.empty:
    st.warning("Aucune donnée disponible."); st.stop()

if value_col not in data.columns:
    st.warning("Pas de variations calculables."); st.stop()

valid=data.dropna(subset=[value_col]).copy()
top=valid.sort_values(value_col, ascending=False).head(5)
low=valid.sort_values(value_col, ascending=True).head(5)

def bar(df, title):
    d=df.copy()
    d["Name"]=d.get("name", d.get("Ticker","")).astype(str)
    d["pct"]=d[value_col]*100
    d["color"]=np.where(d["pct"]>=0,"Hausses","Baisses")
    ch=alt.Chart(d).mark_bar().encode(
        x=alt.X("Name:N", sort="-y", title="Société"),
        y=alt.Y("pct:Q", title="Variation (%)"),
        color=alt.Color("color:N", scale=alt.Scale(domain=["Hausses","Baisses"], range=["#0b8f3a","#d5353a"]), legend=None),
        tooltip=["Name","Ticker",alt.Tooltip("pct",format=".2f")]
    ).properties(title=title, height=300)
    st.altair_chart(ch, use_container_width=True)

c1,c2=st.columns(2)
with c1: bar(top, "Top 5 hausses")
with c2: bar(low, "Top 5 baisses")

st.subheader("🔍 Analyses IA (Top/Low)")
def table_ai(df, held=False):
    rows=[]
    for _,r in df.iterrows():
        name=r.get("name", r.get("Ticker"))
        tick=r.get("Ticker","")
        txt,_,items=news_summary(str(name), tick)
        note=ai_flash_note(str(name), tick, r)
        dec=decision_label_from_row(r, held=held)
        news_links=" • ".join([f"[{t}]({u})" for t,u in items[:3]]) if items else "—"
        rows.append({"Nom":name,"Ticker":tick,"Var%":round((r.get(value_col,0) or 0)*100,2),"Décision":dec,"Note IA":note,"Actu":news_links})
    return pd.DataFrame(rows)

st.dataframe(table_ai(top), use_container_width=True, hide_index=True)
st.dataframe(table_ai(low), use_container_width=True, hide_index=True)
