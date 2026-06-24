"""消費者智慧儀表板 — 呼叫 FastAPI 服務的薄前端(繁體中文)。

前端本身不做任何運算,只呼叫 API 取數並呈現;API 是唯一的真相來源。

    # 終端 1:uvicorn consumer_intel.api.app:app --reload
    # 終端 2:streamlit run app/dashboard.py
    # (或:docker compose up)

以 API_URL 指向服務(預設 http://localhost:8000)。
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from consumer_intel.labels import CLUSTER_ZH, segment_zh

API_URL = os.environ.get("API_URL", "http://localhost:8000")

NAVY, TEAL, AMBER, CORAL, SLATE = "#0f2a4a", "#2a9d8f", "#e9a23b", "#e76f51", "#6c7a89"
SEQ = ["#0f2a4a", "#1d4e74", "#2a9d8f", "#7cc6b8", "#e9a23b", "#e76f51", "#b8627d"]
BLUES = ["#7cc6b8", "#0f2a4a"]
RISK_COLOR = {"low": TEAL, "medium": AMBER, "high": CORAL}
RISK_ZH = {"low": "低", "medium": "中", "high": "高"}

st.set_page_config(page_title="消費者智慧儀表板", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1200px;}
      h1, h2, h3 {color: #0f2a4a; letter-spacing: .3px;}
      [data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e6ebf2; border-radius: 14px;
        padding: 14px 16px;
      }
      [data-testid="stMetricLabel"] {color: #6c7a89;}
      .insight-card {
        background: #f7f9fc; border: 1px solid #e6ebf2; border-left: 5px solid #2a9d8f;
        border-radius: 12px; padding: 16px 20px; margin: 6px 0 14px 0;
      }
      .risk-pill {
        display:inline-block; padding:2px 12px; border-radius:999px;
        color:#fff; font-size:.8rem; font-weight:600;
      }
      .chart-title {font-weight:700; color:#0f2a4a; margin:2px 0 2px 2px;}
      .caption-dim {color:#6c7a89; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, **params) -> object | None:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=20)
    except requests.RequestException as exc:
        st.error(f"無法連線到 API({API_URL}):{exc}")
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def money(x: float | None) -> str:
    return f"£{(x or 0):,.0f}"


def title(text: str) -> None:
    st.markdown(f"<div class='chart-title'>{text}</div>", unsafe_allow_html=True)


def bare(fig, h=360):
    fig.update_layout(
        height=h,
        margin=dict(l=0, r=10, t=8, b=0),
        plot_bgcolor="white",
        coloraxis_showscale=False,
    )
    return fig


# ---- 標題列 -------------------------------------------------------------
st.title("📊 消費者智慧儀表板")
health = api_get("/health")
if health:
    st.markdown(
        f"<span class='caption-dim'>API 狀態:{health['status']} ｜ "
        f"客戶數:{health['customers']:,}</span>",
        unsafe_allow_html=True,
    )

tab_seg, tab_cust, tab_prod = st.tabs(["客群總覽", "客戶分析", "產品分析"])

# ======================================================================
# 客群總覽
# ======================================================================
with tab_seg:
    data = api_get("/segments")
    if data:
        df = pd.DataFrame(data)
        df["客群"] = df["segment"].map(segment_zh)

        total_cust = int(df["customers"].sum())
        total_rev = float(df["total_revenue"].fillna(0).sum())
        avg_clv = float(
            (df["avg_predicted_clv"].fillna(0) * df["customers"]).sum() / max(total_cust, 1)
        )
        avg_alive = float(
            (df["avg_prob_alive"].fillna(0) * df["customers"]).sum() / max(total_cust, 1)
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總客戶數", f"{total_cust:,}")
        c2.metric("總營收", money(total_rev))
        c3.metric("平均預估 CLV", money(avg_clv))
        c4.metric("平均存活機率", f"{avg_alive:.0%}")

        st.markdown("#### 客群價值")
        col_l, col_r = st.columns([3, 2])
        with col_l:
            title("各客群營收")
            d = df.sort_values("total_revenue", ascending=True)
            fig = px.bar(
                d,
                x="total_revenue",
                y="客群",
                orientation="h",
                labels={"total_revenue": "營收(£)", "客群": ""},
                color="total_revenue",
                color_continuous_scale=BLUES,
            )
            st.plotly_chart(bare(fig, 380), use_container_width=True)
        with col_r:
            title("營收占比")
            fig = px.pie(
                df,
                values="total_revenue",
                names="客群",
                hole=0.5,
                color_discrete_sequence=SEQ,
            )
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=8, b=0),
                legend=dict(font=dict(size=10), orientation="v"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 客群分布:客戶數 × 平均預估 CLV(泡泡大小 = 營收)")
        fig = px.scatter(
            df,
            x="customers",
            y="avg_predicted_clv",
            size="total_revenue",
            color="客群",
            text="客群",
            size_max=60,
            color_discrete_sequence=SEQ,
            labels={"customers": "客戶數", "avg_predicted_clv": "平均預估 CLV(£)"},
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(showlegend=False)
        st.plotly_chart(bare(fig, 430), use_container_width=True)

        st.markdown("#### 預估 CLV 最高的客戶")
        top = api_get("/customers/top-clv", limit=15)
        if top:
            tdf = pd.DataFrame(top)
            tdf["客群"] = tdf["segment"].map(segment_zh)
            fig = px.bar(
                tdf.sort_values("predicted_clv"),
                x="predicted_clv",
                y="customer_id",
                orientation="h",
                color="客群",
                color_discrete_sequence=SEQ,
                labels={"predicted_clv": "預估 CLV(£)", "customer_id": "客戶編號"},
            )
            fig.update_layout(height=430, margin=dict(l=0, r=0, t=8, b=0), plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("客群明細表"):
            show = df[
                ["客群", "customers", "total_revenue", "avg_predicted_clv", "avg_prob_alive"]
            ].rename(
                columns={
                    "customers": "客戶數",
                    "total_revenue": "營收",
                    "avg_predicted_clv": "平均預估CLV",
                    "avg_prob_alive": "平均存活機率",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)

# ======================================================================
# 客戶分析
# ======================================================================
with tab_cust:
    st.markdown("#### 查詢單一客戶(輸入客戶編號)")
    cid = st.text_input("客戶編號", value="12347")
    if cid:
        profile = api_get(f"/customers/{cid}")
        if profile is None:
            st.warning(f"查無客戶 {cid}。")
        else:
            seg = segment_zh(profile.get("segment"))
            clu = CLUSTER_ZH.get(profile.get("cluster_name") or "", profile.get("cluster_name"))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("客群", seg)
            c2.metric("預估 CLV", money(profile.get("predicted_clv")))
            c3.metric("存活機率", f"{(profile.get('prob_alive') or 0):.0%}")
            prop = profile.get("propensity")
            c4.metric("90 天回購傾向", f"{prop:.0%}" if prop is not None else "—")

            col_l, col_r = st.columns([2, 3])
            with col_l:
                title("RFM 概況")
                st.metric("最近購買(天)", f"{profile.get('recency') or 0:,}")
                st.metric("購買次數", f"{profile.get('frequency') or 0:,}")
                st.metric("累積消費", money(profile.get("monetary")))
                st.markdown(f"<span class='caption-dim'>分群:{clu}</span>", unsafe_allow_html=True)
            with col_r:
                title("存活機率 / 回購傾向")
                fig = go.Figure()
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=(profile.get("prob_alive") or 0) * 100,
                        title={"text": "存活機率 (%)"},
                        domain={"row": 0, "column": 0},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": TEAL}},
                    )
                )
                if prop is not None:
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number",
                            value=prop * 100,
                            title={"text": "回購傾向 (%)"},
                            domain={"row": 0, "column": 1},
                            gauge={"axis": {"range": [0, 100]}, "bar": {"color": AMBER}},
                        )
                    )
                    fig.update_layout(grid={"rows": 1, "columns": 2})
                fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            title("AI 洞察")
            insight = api_get(f"/customers/{cid}/insight")
            if insight:
                rl = insight["risk_level"]
                pill = (
                    f"<span class='risk-pill' style='background:{RISK_COLOR.get(rl, SLATE)}'>"
                    f"流失風險:{RISK_ZH.get(rl, rl)}</span>"
                )
                obs = "".join(f"<li>{o}</li>" for o in insight["observations"])
                act = "".join(f"<li>{a}</li>" for a in insight["recommended_actions"])
                st.markdown(
                    f"<div class='insight-card'>"
                    f"<div style='font-size:1.05rem;font-weight:700;margin-bottom:6px'>"
                    f"{insight['headline']}</div>{pill}"
                    f"<div style='margin-top:10px'><b>觀察</b><ul>{obs}</ul></div>"
                    f"<div><b>建議行動</b><ul>{act}</ul></div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<span class='caption-dim'>數字皆由後端計算,AI 僅負責敘述(grounded)。</span>",
                    unsafe_allow_html=True,
                )

    st.markdown("#### 客戶瀏覽表(消費金額前 100 名,可從中找客戶編號)")
    clist = api_get("/customers", limit=100)
    if clist:
        cdf = pd.DataFrame(clist)
        cdf["客群"] = cdf["segment"].map(segment_zh)
        show = cdf[
            [
                "customer_id",
                "客群",
                "recency",
                "frequency",
                "monetary",
                "predicted_clv",
                "propensity",
            ]
        ].rename(
            columns={
                "customer_id": "客戶編號",
                "recency": "最近購買(天)",
                "frequency": "購買次數",
                "monetary": "累積消費",
                "predicted_clv": "預估CLV",
                "propensity": "回購傾向",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True, height=320)

# ======================================================================
# 產品分析
# ======================================================================
with tab_prod:
    prods = api_get("/products", limit=500)
    if prods:
        pdf = pd.DataFrame(prods)
        c1, c2, c3 = st.columns(3)
        c1.metric("商品數(前 500 大)", f"{len(pdf):,}")
        c2.metric("前 500 大商品總營收", money(pdf["revenue"].sum()))
        c3.metric("最高營收商品", money(pdf["revenue"].max()))

        st.markdown("#### 營收最高的商品 Top 15")
        top15 = pdf.head(15).copy()
        top15["label"] = top15["stock_code"] + " " + top15["description"].str.slice(0, 22)
        fig = px.bar(
            top15.sort_values("revenue"),
            x="revenue",
            y="label",
            orientation="h",
            color="revenue",
            color_continuous_scale=BLUES,
            labels={"revenue": "營收(£)", "label": ""},
        )
        st.plotly_chart(bare(fig, 460), use_container_width=True)

        st.markdown("#### 營收 × 銷售數量(泡泡大小 = 購買客戶數,前 60 大)")
        b = pdf.head(60)
        fig = px.scatter(
            b,
            x="quantity",
            y="revenue",
            size="customers",
            color="revenue",
            hover_name="description",
            color_continuous_scale=BLUES,
            labels={"quantity": "銷售數量", "revenue": "營收(£)"},
        )
        st.plotly_chart(bare(fig, 420), use_container_width=True)

        st.markdown("#### 商品瀏覽表(營收前 100 名,可從中找商品代碼)")
        show = pdf.head(100)[
            ["stock_code", "description", "revenue", "quantity", "orders", "customers"]
        ].rename(
            columns={
                "stock_code": "商品代碼",
                "description": "品名",
                "revenue": "營收",
                "quantity": "銷售數量",
                "orders": "訂單數",
                "customers": "購買客戶數",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True, height=320)

    st.markdown("#### 查詢單一商品 + 下一步最佳推薦(輸入商品代碼)")
    code = st.text_input("商品代碼", value="20725")
    if code:
        detail = api_get(f"/products/{code}")
        if detail is None:
            st.warning(f"查無商品 {code}。")
        else:
            st.markdown(f"**{detail['stock_code']} — {detail.get('description') or ''}**")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("營收", money(detail.get("revenue")))
            d2.metric("銷售數量", f"{detail.get('quantity') or 0:,}")
            d3.metric("訂單數", f"{detail.get('orders') or 0:,}")
            d4.metric("購買客戶數", f"{detail.get('customers') or 0:,}")

        recs = api_get(f"/products/{code}/next-best-offer", limit=10)
        if not recs:
            st.info("在目前的門檻下,此商品沒有對應的關聯規則。")
        else:
            rdf = pd.DataFrame(recs)
            title("最常一起購買的商品(依提升度)")
            fig = px.bar(
                rdf.sort_values("lift"),
                x="lift",
                y="consequents",
                orientation="h",
                color="lift",
                color_continuous_scale=BLUES,
                labels={"lift": "提升度 (lift)", "consequents": "推薦商品"},
            )
            st.plotly_chart(bare(fig, 360), use_container_width=True)
            show = rdf[["consequents", "support", "confidence", "lift"]].rename(
                columns={
                    "consequents": "推薦商品",
                    "support": "支持度",
                    "confidence": "信心度",
                    "lift": "提升度",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.markdown(
                "<span class='caption-dim'>提升度 > 1 代表兩商品同時出現的機率,"
                "高於各自獨立時的預期。</span>",
                unsafe_allow_html=True,
            )
