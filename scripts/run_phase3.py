"""Phase 3 pipeline: market basket analysis -> association rules -> Next Best
Offer demos.

Run from the project root (needs the Phase 0 clean parquet):

    python scripts/run_phase3.py

Outputs:
    data/processed/association_rules.parquet
    reports/phase3_basket_nbo.md
    reports/basket_rules_scatter.html
"""

from __future__ import annotations

import plotly.express as px

from consumer_intel import config
from consumer_intel.basket import nbo, rules
from consumer_intel.data.load import load_clean

MIN_ITEM_SUPPORT = 0.01
MIN_SUPPORT = 0.01
MIN_LIFT = 1.0
RULES_PARQUET = config.PROCESSED_DIR / "association_rules.parquet"

# A few popular seed products to demo Next Best Offer in the report.
DEMO_PRODUCTS = ["22423", "85123A", "47566", "20725"]


def _rule_label(codes: frozenset, desc: dict[str, str]) -> str:
    return " + ".join(desc.get(c, c) for c in sorted(codes))


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean transactions ...")
    tx = load_clean()
    desc = rules.product_descriptions(tx)

    print(f"Mining rules (min_item_support={MIN_ITEM_SUPPORT}, min_support={MIN_SUPPORT}) ...")
    rule_df = rules.mine_rules(
        tx,
        min_item_support=MIN_ITEM_SUPPORT,
        min_support=MIN_SUPPORT,
        metric="lift",
        min_threshold=MIN_LIFT,
    )
    print(f"  {len(rule_df):,} rules")

    # --- save rules (parquet-friendly: frozensets -> readable strings) -----
    save = rule_df.copy()
    save["antecedents_codes"] = save["antecedents"].apply(lambda s: sorted(s))
    save["consequents_codes"] = save["consequents"].apply(lambda s: sorted(s))
    save["antecedents"] = save["antecedents"].apply(lambda s: _rule_label(s, desc))
    save["consequents"] = save["consequents"].apply(lambda s: _rule_label(s, desc))
    keep_cols = [
        "antecedents",
        "consequents",
        "antecedents_codes",
        "consequents_codes",
        "support",
        "confidence",
        "lift",
    ]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save[keep_cols].to_parquet(RULES_PARQUET, index=False)
    print(f"Wrote rules -> {RULES_PARQUET}")

    # --- scatter: support vs confidence, sized/coloured by lift ------------
    fig = px.scatter(
        rule_df,
        x="support",
        y="confidence",
        color="lift",
        size="lift",
        title="Association rules (support vs confidence, coloured by lift)",
        hover_data={"lift": ":.1f"},
    )
    fig.write_html(config.REPORTS_DIR / "basket_rules_scatter.html")

    # --- report ------------------------------------------------------------
    lines: list[str] = ["# Phase 3 — Market Basket Analysis & Next Best Offer\n"]
    lines.append(
        f"Mined **{len(rule_df):,}** association rules from "
        f"{tx['InvoiceNo'].nunique():,} baskets over "
        f"{len(rules.frequent_products(tx, MIN_ITEM_SUPPORT)):,} frequent products "
        f"(products in ≥{MIN_ITEM_SUPPORT:.0%} of baskets; FP-Growth, "
        f"min_support={MIN_SUPPORT}).\n"
    )

    lines.append("## Top 15 rules by lift\n")
    top = rule_df.head(15).copy()
    top["antecedents"] = top["antecedents"].apply(lambda s: _rule_label(s, desc))
    top["consequents"] = top["consequents"].apply(lambda s: _rule_label(s, desc))
    top["support"] = top["support"].round(4)
    top["confidence"] = top["confidence"].round(3)
    top["lift"] = top["lift"].round(1)
    lines.append(
        top[["antecedents", "consequents", "support", "confidence", "lift"]].to_markdown(
            index=False
        )
    )

    lines.append("\n## Next Best Offer examples\n")
    lines.append(
        "Given a seed product, the engine returns the highest-lift products to "
        "recommend next (excluding the seed).\n"
    )
    for code in DEMO_PRODUCTS:
        recs = nbo.add_descriptions(nbo.recommend_for_product(rule_df, code, top_n=5), desc)
        seed = desc.get(code, code)
        lines.append(f"\n**Seed: {code} — {seed}**\n")
        if recs.empty:
            lines.append("_No rules fired for this product at the current thresholds._\n")
            continue
        view = recs[["StockCode", "Description", "lift", "confidence"]].copy()
        view["lift"] = view["lift"].round(1)
        view["confidence"] = view["confidence"].round(3)
        lines.append(view.to_markdown(index=False))

    lines.append("\n## Charts\n- `reports/basket_rules_scatter.html`\n")

    (config.REPORTS_DIR / "phase3_basket_nbo.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report -> {config.REPORTS_DIR / 'phase3_basket_nbo.md'}")


if __name__ == "__main__":
    main()
