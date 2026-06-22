"""Streamlit dashboard — a thin client over the FastAPI service.

It does no analytics itself; it calls the API and renders the results. This
keeps the architecture clean (the API is the single source of truth) and the
frontend demonstrates client-side backend integration.

    # terminal 1: uvicorn consumer_intel.api.app:app --reload
    # terminal 2: streamlit run app/dashboard.py
    # (or: docker compose up)

Set API_URL to point at the service (default http://localhost:8000).
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Consumer Intelligence", layout="wide")
st.title("Consumer Intelligence")


def api_get(path: str, **params) -> object | None:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=15)
    except requests.RequestException as exc:
        st.error(f"Cannot reach API at {API_URL}: {exc}")
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


health = api_get("/health")
if health:
    st.caption(f"API: {health['status']} · {health['customers']:,} customers")

tab_seg, tab_cust, tab_nbo = st.tabs(["Segments", "Customer 360", "Next Best Offer"])

with tab_seg:
    st.subheader("Segment value rollup")
    data = api_get("/segments")
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        if {"segment", "total_revenue"}.issubset(df.columns):
            st.bar_chart(df.set_index("segment")["total_revenue"])

    st.subheader("Top customers by predicted CLV")
    top = api_get("/customers/top-clv", limit=20)
    if top:
        st.dataframe(pd.DataFrame(top), use_container_width=True)

with tab_cust:
    st.subheader("Customer 360")
    cid = st.text_input("Customer ID", value="12347")
    if cid:
        profile = api_get(f"/customers/{cid}")
        if profile is None:
            st.warning(f"Customer {cid} not found.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Segment", profile.get("segment") or "—")
            c2.metric("Predicted CLV", f"£{(profile.get('predicted_clv') or 0):,.0f}")
            c3.metric("P(alive)", f"{(profile.get('prob_alive') or 0):.0%}")
            prop = profile.get("propensity")
            c4.metric("90d propensity", f"{prop:.0%}" if prop is not None else "—")

            insight = api_get(f"/customers/{cid}/insight")
            if insight:
                st.markdown(f"**{insight['headline']}**")
                st.markdown("**Observations**")
                for o in insight["observations"]:
                    st.markdown(f"- {o}")
                st.markdown("**Recommended actions**")
                for a in insight["recommended_actions"]:
                    st.markdown(f"- {a}")
                st.caption(
                    f"Risk level: {insight['risk_level']} · narrative grounded in stored metrics"
                )

with tab_nbo:
    st.subheader("Next Best Offer (by product)")
    code = st.text_input("Stock code", value="20725")
    if code:
        recs = api_get(f"/products/{code}/next-best-offer", limit=10)
        if not recs:
            st.info("No rules for this product at the current thresholds.")
        else:
            st.dataframe(pd.DataFrame(recs), use_container_width=True)
