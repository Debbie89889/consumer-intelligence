"""消費者智慧儀表板 — 呼叫 FastAPI 服務的薄前端(繁體中文)。

前端不做運算,只呼叫 API 取數並呈現;API 是唯一真相來源。

    # 終端 1:uvicorn consumer_intel.api.app:app --reload
    # 終端 2:streamlit run app/dashboard.py   (或 docker compose up)

以 API_URL 指向服務(預設 http://localhost:8000)。
"""

from __future__ import annotations

import html
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from consumer_intel.labels import CLUSTER_ZH, SEGMENT_ZH, segment_zh

API_URL = os.environ.get("API_URL", "http://localhost:8000")

NAVY, TEAL, AMBER, CORAL, SLATE = "#0f2a4a", "#2a9d8f", "#e9a23b", "#e76f51", "#6c7a89"
BLUES = ["#7cc6b8", "#0f2a4a"]
RISK_COLOR = {"low": TEAL, "medium": AMBER, "high": CORAL}
RISK_ZH = {"low": "低", "medium": "中", "high": "高"}
# 固定的客群配色:同一客群在所有圖表都同色(#3)
_PAL = [
    "#0f2a4a",
    "#1d4e74",
    "#2a9d8f",
    "#7cc6b8",
    "#e9a23b",
    "#e76f51",
    "#b8627d",
    "#8a6fb0",
    "#4c9f70",
    "#c98a3b",
    "#6c7a89",
]
SEGMENT_COLOR = {zh: _PAL[i % len(_PAL)] for i, zh in enumerate(SEGMENT_ZH.values())}

# 國名 → ISO-3 代碼(用 ISO-3 上色,避免 plotly 'country names' 的相容性警告)。
# 未列入的名稱(Channel Islands、Unspecified 等)會自動從地圖略過。
COUNTRY_ISO3 = {
    "United Kingdom": "GBR",
    "EIRE": "IRL",
    "Netherlands": "NLD",
    "Germany": "DEU",
    "France": "FRA",
    "Australia": "AUS",
    "Spain": "ESP",
    "Switzerland": "CHE",
    "Sweden": "SWE",
    "Denmark": "DNK",
    "Belgium": "BEL",
    "Portugal": "PRT",
    "Japan": "JPN",
    "Norway": "NOR",
    "Italy": "ITA",
    "Finland": "FIN",
    "Cyprus": "CYP",
    "Austria": "AUT",
    "Greece": "GRC",
    "Singapore": "SGP",
    "Israel": "ISR",
    "Poland": "POL",
    "United Arab Emirates": "ARE",
    "USA": "USA",
    "Iceland": "ISL",
    "Lithuania": "LTU",
    "Malta": "MLT",
    "Canada": "CAN",
    "Thailand": "THA",
    "RSA": "ZAF",
    "Lebanon": "LBN",
    "Brazil": "BRA",
    "Bahrain": "BHR",
    "Korea": "KOR",
    "Czech Republic": "CZE",
    "Saudi Arabia": "SAU",
    "Nigeria": "NGA",
}

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


# ---- 資料抓取:快取 + 冷啟動友善訊息(#1, #5) -------------------------
@st.cache_data(ttl=300, show_spinner="資料載入中…")
def _get(url: str, params_items: tuple) -> object | None:
    resp = requests.get(url, params=dict(params_items), timeout=25)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def api_get(path: str, **params) -> object | None:
    try:
        return _get(f"{API_URL}{path}", tuple(sorted(params.items())))
    except requests.RequestException:
        st.warning("⏳ 服務可能正在喚醒(免費方案閒置會休眠),請稍候數秒後重新整理頁面。")
        return None


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


def selected_rows(event) -> list[int]:
    """從 st.dataframe 的選取結果取出被點選的列索引(相容不同版本)。"""
    try:
        sel = event.selection
        return list(sel["rows"]) if isinstance(sel, dict) else list(sel.rows)
    except Exception:
        return []


# ---- 標題列 -------------------------------------------------------------
st.title("📊 消費者智慧儀表板")
health = api_get("/health")
if health:
    st.markdown(
        f"<span class='caption-dim'>API 狀態:{health['status']} ｜ "
        f"客戶數:{health['customers']:,}</span>",
        unsafe_allow_html=True,
    )

tab_seg, tab_cust, tab_prod, tab_trend = st.tabs(["客群總覽", "客戶分析", "產品分析", "趨勢與地區"])

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
                color="客群",
                color_discrete_map=SEGMENT_COLOR,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(bare(fig, 380), use_container_width=True)
        with col_r:
            title("營收占比")
            fig = px.pie(
                df,
                values="total_revenue",
                names="客群",
                hole=0.5,
                color="客群",
                color_discrete_map=SEGMENT_COLOR,
            )
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=8, b=0),
                legend=dict(font=dict(size=10)),
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
            color_discrete_map=SEGMENT_COLOR,
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
            # 客戶編號是數字字串,加前綴並強制類別軸,避免被當數字縮寫成「16k」
            tdf["客戶"] = "客戶 " + tdf["customer_id"].astype(str)
            tdf = tdf.sort_values("predicted_clv")
            fig = px.bar(
                tdf,
                x="predicted_clv",
                y="客戶",
                orientation="h",
                color="客群",
                color_discrete_map=SEGMENT_COLOR,
                text=tdf["predicted_clv"].map(lambda v: f"£{v:,.0f}"),
                custom_data=["prob_alive"],
                labels={"predicted_clv": "預估 CLV(£)", "客戶": ""},
            )
            fig.update_traces(
                textposition="outside",
                textfont_size=11,
                cliponaxis=False,
                hovertemplate="%{y}<br>預估 CLV:£%{x:,.0f}"
                "<br>存活機率:%{customdata[0]:.0%}<extra></extra>",
            )
            fig.update_yaxes(type="category")
            fig.update_xaxes(tickprefix="£", showgrid=True, gridcolor="#eef2f7")
            fig.update_layout(height=480, margin=dict(l=0, r=60, t=8, b=0), plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# 客戶分析(查詢 + 可點選的瀏覽表 #2 #4)
# ======================================================================
with tab_cust:
    st.session_state.setdefault("cid", "12347")
    st.markdown("#### 查詢客戶")
    with st.form("form_cust"):
        col_a, col_b = st.columns([3, 1])
        typed = col_a.text_input("客戶編號", value=st.session_state["cid"])
        if col_b.form_submit_button("查詢", use_container_width=True):
            st.session_state["cid"] = (typed or "").strip()

    st.markdown(
        "<span class='caption-dim'>或點選下表任一列即可查詢。</span>",
        unsafe_allow_html=True,
    )
    clist = api_get("/customers", limit=100)
    if clist:
        cdf = pd.DataFrame(clist)
        cdf["客群"] = cdf["segment"].map(segment_zh)
        disp = cdf[
            [
                "customer_id",
                "客群",
                "recency",
                "frequency",
                "monetary",
                "predicted_clv",
                "propensity",
            ]
        ]
        ev = st.dataframe(
            disp,
            key="tbl_cust",
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                "customer_id": st.column_config.TextColumn("客戶編號"),
                "recency": st.column_config.NumberColumn("最近購買(天)"),
                "frequency": st.column_config.NumberColumn("購買次數"),
                "monetary": st.column_config.NumberColumn("累積消費", format="£%d"),
                "predicted_clv": st.column_config.NumberColumn("預估CLV", format="£%d"),
                "propensity": st.column_config.NumberColumn("回購傾向", format="%.0f%%"),
            },
        )
        rows = selected_rows(ev)
        if rows and rows[0] < len(cdf):
            st.session_state["cid"] = str(cdf.iloc[rows[0]]["customer_id"])

    cid = st.session_state["cid"]
    profile = api_get(f"/customers/{cid}") if cid else None
    if cid and profile is None:
        st.warning(f"查無客戶 {cid}。")
    elif profile:
        st.markdown(f"### 客戶 {cid}")
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
            # #7:LLM/動態文字先 escape 再進 HTML,避免注入
            head = html.escape(insight["headline"])
            obs = "".join(f"<li>{html.escape(o)}</li>" for o in insight["observations"])
            act = "".join(f"<li>{html.escape(a)}</li>" for a in insight["recommended_actions"])
            st.markdown(
                f"<div class='insight-card'>"
                f"<div style='font-size:1.05rem;font-weight:700;margin-bottom:6px'>{head}</div>"
                f"{pill}"
                f"<div style='margin-top:10px'><b>觀察</b><ul>{obs}</ul></div>"
                f"<div><b>建議行動</b><ul>{act}</ul></div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<span class='caption-dim'>數字皆由後端計算,AI 僅負責敘述(grounded)。</span>",
                unsafe_allow_html=True,
            )

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
        fig = px.scatter(
            pdf.head(60),
            x="quantity",
            y="revenue",
            size="customers",
            color="revenue",
            hover_name="description",
            color_continuous_scale=BLUES,
            labels={"quantity": "銷售數量", "revenue": "營收(£)"},
        )
        st.plotly_chart(bare(fig, 420), use_container_width=True)

    st.session_state.setdefault("pcode", "20725")
    st.markdown("#### 查詢商品 + 下一步最佳推薦")
    with st.form("form_prod"):
        col_a, col_b = st.columns([3, 1])
        typed = col_a.text_input("商品代碼", value=st.session_state["pcode"])
        if col_b.form_submit_button("查詢", use_container_width=True):
            st.session_state["pcode"] = (typed or "").strip()

    st.markdown(
        "<span class='caption-dim'>或點選下表任一列即可查詢。</span>",
        unsafe_allow_html=True,
    )
    if prods:
        disp = pdf.head(100)[
            ["stock_code", "description", "revenue", "quantity", "orders", "customers"]
        ]
        ev = st.dataframe(
            disp,
            key="tbl_prod",
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                "stock_code": st.column_config.TextColumn("商品代碼"),
                "description": st.column_config.TextColumn("品名", width="large"),
                "revenue": st.column_config.NumberColumn("營收", format="£%d"),
                "quantity": st.column_config.NumberColumn("銷售數量"),
                "orders": st.column_config.NumberColumn("訂單數"),
                "customers": st.column_config.NumberColumn("購買客戶數"),
            },
        )
        rows = selected_rows(ev)
        if rows and rows[0] < len(pdf):
            st.session_state["pcode"] = str(pdf.iloc[rows[0]]["stock_code"])

    code = st.session_state["pcode"]
    detail = api_get(f"/products/{code}") if code else None
    if code and detail is None:
        st.warning(f"查無商品 {code}。")
    elif detail:
        st.markdown(f"### {detail['stock_code']} — {detail.get('description') or ''}")
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
            st.dataframe(
                rdf[["consequents", "support", "confidence", "lift"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "consequents": st.column_config.TextColumn("推薦商品", width="large"),
                    "support": st.column_config.NumberColumn("支持度", format="%.3f"),
                    "confidence": st.column_config.NumberColumn("信心度", format="%.2f"),
                    "lift": st.column_config.NumberColumn("提升度", format="%.1f"),
                },
            )

# ======================================================================
# 趨勢與地區
# ======================================================================
with tab_trend:
    monthly = api_get("/analytics/monthly")
    if monthly:
        mdf = pd.DataFrame(monthly)
        peak = mdf.loc[mdf["revenue"].idxmax()]
        c1, c2, c3 = st.columns(3)
        c1.metric("涵蓋月份", f"{len(mdf)} 個月")
        c2.metric("月營收高峰", money(peak["revenue"]))
        c3.metric("高峰月份", str(peak["month"]))

        st.markdown("#### 月營收趨勢")
        fig = px.area(
            mdf,
            x="month",
            y="revenue",
            markers=True,
            labels={"month": "月份", "revenue": "營收(£)"},
            color_discrete_sequence=[TEAL],
        )
        fig.update_traces(line=dict(width=2), fillcolor="rgba(42,157,143,.15)")
        st.plotly_chart(bare(fig, 360), use_container_width=True)

        st.markdown("#### 月訂單數與下單客戶數")
        long = mdf.melt(
            id_vars="month",
            value_vars=["orders", "customers"],
            var_name="指標",
            value_name="數量",
        )
        long["指標"] = long["指標"].map({"orders": "訂單數", "customers": "下單客戶數"})
        fig = px.line(
            long,
            x="month",
            y="數量",
            color="指標",
            markers=True,
            color_discrete_sequence=[NAVY, AMBER],
            labels={"month": "月份"},
        )
        st.plotly_chart(bare(fig, 340), use_container_width=True)

    st.markdown("#### 各國分布")
    mc1, mc2 = st.columns([2, 1])
    metric_label = mc1.selectbox("上色指標", ["營收", "訂單數", "下單客戶數"], index=0)
    drop_uk = mc2.checkbox("排除英國", value=False)
    metric = {"營收": "revenue", "訂單數": "orders", "下單客戶數": "customers"}[metric_label]
    countries = api_get("/analytics/countries", limit=50)
    if countries:
        cdf = pd.DataFrame(countries)
        if drop_uk:
            cdf = cdf[cdf["country"] != "United Kingdom"]
        mapdf = cdf.copy()
        mapdf["iso3"] = mapdf["country"].map(COUNTRY_ISO3)
        mapdf = mapdf.dropna(subset=["iso3"])
        fig = px.choropleth(
            mapdf,
            locations="iso3",
            locationmode="ISO-3",
            color=metric,
            hover_name="country",
            color_continuous_scale="Plasma",
            labels={"revenue": "營收(£)", "orders": "訂單數", "customers": "客戶數"},
        )
        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#cfd8e3",
            landcolor="#f2f5f9",
            projection_type="natural earth",
        )
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=8, b=0))
        st.plotly_chart(fig, use_container_width=True)

        show = cdf.head(15)[["country", "revenue", "orders", "customers"]]
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "country": st.column_config.TextColumn("國家"),
                "revenue": st.column_config.NumberColumn("營收", format="£%d"),
                "orders": st.column_config.NumberColumn("訂單數"),
                "customers": st.column_config.NumberColumn("購買客戶數"),
            },
        )
        st.markdown(
            "<span class='caption-dim'>此資料集以英國為主(故地圖上英國最亮);"
            "可勾選排除英國,或切換上色指標。部分非國家名稱(如 Channel Islands、"
            "Unspecified)無法定位,已從地圖略過。</span>",
            unsafe_allow_html=True,
        )
