# -*- coding: utf-8 -*-
import streamlit as st, pandas as pd, numpy as np, json, os, altair as alt
from lib import fetch_prices, compute_metrics, decision_label_from_row, ai_flash_note

st.title("💼 Mon Portefeuille — IA & Pastilles")

PATH="data/portfolio.json"
if os.path.exists(PATH):
    port = pd.read_json(PATH)
else:
    port = pd.DataFrame(columns=["Name","Ticker","Account","Quantity","PRU"])

st.subheader("Éditeur")
edited = st.data_editor(port, num_rows="dynamic", use_container_width=True, key="port_editor")
c1,c2=st.columns(2)
with c1:
    if st.button("💾 Sauvegarder"):
        edited.to_json(PATH, orient="records", indent=2, force_ascii=False)
        st.success("Sauvegardé.")
with c2:
    if st.button("🗑 Réinitialiser"):
        if os.path.exists(PATH): os.remove(PATH)
        st.session_state.pop("port_editor", None)
        st.experimental_rerun()

if edited.empty:
    st.info("Ajoutez des lignes (Ticker requis)."); st.stop()

tickers = edited["Ticker"].dropna().unique().tolist()
data = fetch_prices(tickers, days=90)
met = compute_metrics(data)
merged = edited.merge(met, on="Ticker", how="left")

rows=[]
for _,r in merged.iterrows():
    px=float(r.get("Close", np.nan)); q=float(r.get("Quantity",0) or 0); pru=float(r.get("PRU", np.nan) or np.nan)
    val = (px*q) if np.isfinite(px) else 0.0
    perf = ((px-pru)/pru*100) if (np.isfinite(px) and np.isfinite(pru) and pru>0) else np.nan
    dec = decision_label_from_row(r, held=True)
    note = ai_flash_note(str(r.get("Name", r.get("Ticker",""))), r.get("Ticker",""), r)
    rows.append({"Compte":r.get("Account",""),"Nom":r.get("Name", r.get("Ticker","")), "Ticker":r.get("Ticker",""),
                 "Cours":round(px,2) if np.isfinite(px) else None, "PRU":pru, "Qté":q, "Valeur":round(val,2), "Perf%":round(perf,2) if np.isfinite(perf) else None, "Décision":dec, "Note IA":note})
out=pd.DataFrame(rows)

st.subheader("Vue portefeuille")
st.dataframe(out, use_container_width=True, hide_index=True)

st.subheader("Répartition par valeur (€)")
if not out.empty:
    ch=alt.Chart(out).mark_bar().encode(x=alt.X("Nom:N", sort="-y"), y="Valeur:Q", color="Décision:N")
    st.altair_chart(ch, use_container_width=True)
