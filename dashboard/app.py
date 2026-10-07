"""A business-facing client. All model calculations are owned by the API."""

import json
import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Retail Pricing Workbench", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top:4rem; max-width:1400px}
h1 {letter-spacing:-.045em; font-weight:750 !important}
[data-testid="stMetric"] {background:#f1f5f4; border:1px solid #dbe5e2; border-radius:12px; padding:18px}
[data-testid="stMetricLabel"] {color:#526661}
[data-testid="stSidebar"] {background:#edf3f0}
</style>""",
    unsafe_allow_html=True,
)
COLORS = ["#16796d", "#e6a33b", "#597cb5"]


def api(path, payload=None):
    try:
        with httpx.Client(timeout=45) as client:
            response = (
                client.get(API + path) if payload is None else client.post(API + path, json=payload)
            )
        if response.is_error:
            detail = response.json().get("detail", response.text)
            st.error(f"Request could not be completed: {detail}")
            return None
        return response.json()
    except httpx.RequestError:
        st.error(
            "The local API is unavailable. Start the API service using the README instructions."
        )
        return None


@st.cache_data(ttl=60)
def get(path):
    return api(path)


def chart(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", color="#253b35"),
        margin=dict(l=10, r=10, t=35, b=10),
        legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch")


def support_label(s):
    parts = [
        name
        for name, key in [
            ("Circular", "feature"),
            ("Display", "display"),
            ("Price tag only", "tpr_only"),
        ]
        if s[key]
    ]
    return " + ".join(parts) if parts else "No promotional support"


catalog = get("/products")
if not catalog:
    st.stop()
products = catalog["products"]
with st.sidebar:
    st.markdown("### ◈ Retail workbench")
    st.caption("PRICE & PROMOTION ANALYSIS")
    categories = sorted({p["category"] for p in products})
    category = st.selectbox("Category", categories)
    selection = [p for p in products if p["category"] == category]
    product = st.selectbox("Product", selection, format_func=lambda p: p["description"])
    product_id = product["product_id"]
    all_analysis = get(f"/products/{product_id}/analysis")
    if not all_analysis:
        st.stop()
    store = st.selectbox(
        "Store",
        all_analysis["stores"],
        format_func=lambda s: f"{s['store_name']} · {s['store_id']}",
    )
    store_id = store["store_id"]
    page = st.radio(
        "Workspace",
        ["Sales overview", "Elasticity & promotions", "Scenario simulator", "Model validation"],
    )
    st.divider()
    st.caption(
        f"Historical dataset · {catalog['quality']['first_week']} to {catalog['quality']['last_week']}"
    )
    st.caption("dunnhumby · Breakfast at the Frat")
    st.caption("Amounts are in source dollars. No currency conversion.")

analysis = get(f"/products/{product_id}/analysis?store_id={store_id}")
if not analysis:
    st.stop()
context = analysis["context"]
st.caption(f"{category}  /  {store['store_name']}  /  {product_id}")
st.title(page)
st.write(product["description"].title())

if page == "Sales overview":
    across = st.toggle("Show all stores for this product", value=False)
    view = all_analysis if across else analysis
    totals = view["totals"]
    cols = st.columns(4)
    cols[0].metric("Observed revenue", f"${totals['revenue']:,.0f}")
    cols[1].metric("Units sold", f"{totals['units']:,.0f}")
    cols[2].metric("Promoted observations", f"{totals['promotion_share']:.1%}")
    cols[3].metric("Recorded store-weeks", f"{totals['rows']:,}")
    history = pd.DataFrame(view["history"])
    chart(
        px.line(history, x="week", y="units", title="Weekly demand", color_discrete_sequence=COLORS)
    )
    left, right = st.columns(2)
    with left:
        chart(
            px.line(
                history,
                x="week",
                y=["price", "base_price"],
                title="Shelf price · unweighted store mean" if across else "Shelf price",
                color_discrete_sequence=COLORS[1:],
            )
        )
    with right:
        chart(
            px.line(
                history,
                x="week",
                y=["feature", "display", "tpr_only"],
                title="Promotional support · share of observations",
                color_discrete_sequence=COLORS,
            )
        )
    with st.expander("Data quality and coverage"):
        q = catalog["quality"]
        st.write(
            f"{q['clean_rows']:,} valid observations; {q['rejected_rows']:,} excluded across the dataset."
        )
        st.write(
            f"{q['missing_pair_weeks']:,} absent product/store weeks are not treated as zero sales. Some reflect product availability rather than missing reporting."
        )
        st.json(q)

elif page == "Elasticity & promotions":
    d = analysis["diagnostics"]
    low, high = d["elasticity_ci"]
    cols = st.columns(3)
    cols[0].metric("Estimated elasticity", f"{d['elasticity']:.2f}")
    cols[1].metric("95% confidence interval", f"{low:.2f} to {high:.2f}")
    cols[2].metric("Training observations", f"{d['training_rows']:,}")
    st.write(
        f"A 1% price increase is associated with approximately {d['elasticity']:.2f}% change in demand, holding modeled conditions fixed. This product estimate pools stores; each store has its own baseline demand."
    )
    if context["warnings"]:
        st.warning("Automatic recommendation unavailable: " + " ".join(context["warnings"]))
    else:
        st.success(
            "This product and store pass the evidence checks for bounded scenario recommendations."
        )
    coeffs = pd.DataFrame(
        [
            {"Support": key.replace("_", " ").title(), "Log coefficient": value}
            for key, value in d["support_coefficients"].items()
        ]
    )
    chart(
        px.bar(
            coeffs,
            x="Support",
            y="Log coefficient",
            title="Promotional support associations, controlling for price",
            color_discrete_sequence=COLORS,
        )
    )
    st.caption(
        "Use the simulator for percentage lift at the same price. Tag-only promotion is tied to a discount and is not an independently randomized treatment."
    )
    with st.expander("How the model works", expanded=True):
        st.latex(
            r"\log(Q) = \alpha_{store} + \beta\log(P) + \gamma^T Promotion + Seasonality + Trend + \epsilon"
        )
        st.write(
            "The model fits positive sales only. Exponentiated predictions use a training-derived smearing correction. Standard errors are clustered by store. Unobserved stockouts, common weekly shocks, and retailer pricing decisions can still bias estimates."
        )
        st.json(d)

elif page == "Scenario simulator":
    st.info(
        f"Simulating week ending {context['next_week']}. Baseline: ${context['baseline_price']:.2f}, {support_label(context['baseline_support'])}."
    )
    st.caption(
        f"Observed store price support: USD {context['price_support'][0]:.2f}–{context['price_support'][1]:.2f}. The unchanged baseline is also allowed."
    )
    with st.form("scenario"):
        left, right = st.columns(2)
        with left:
            mode = st.radio(
                "Price input", ["Shelf price", "Discount from reference"], horizontal=True
            )
            # Both inputs remain visible; only the selected mode is sent to the API.
            price = st.number_input(
                "Proposed shelf price ($)",
                min_value=0.01,
                value=context["baseline_price"],
                step=0.01,
            )
            reference = st.number_input(
                "Reference price ($)", min_value=0.01, value=context["reference_price"], step=0.01
            )
            discount = st.slider("Discount from reference (%)", 0, 50, 10)
            combinations = context["support_combinations"]
            base_idx = combinations.index(context["baseline_support"])
            support = st.selectbox(
                "Promotional support", combinations, index=base_idx, format_func=support_label
            )
        with right:
            include_cost = st.checkbox("Include assumed costs and contribution profit")
            unit_cost = st.number_input("Assumed unit cost ($)", min_value=0.0, value=0.0, step=0.1)
            promo_cost = st.number_input(
                "Assumed weekly promotion cost ($)", min_value=0.0, value=0.0, step=1.0
            )
            base_cost = st.number_input(
                "Assumed baseline promotion cost ($)", min_value=0.0, value=0.0, step=1.0
            )
            st.caption(
                "Costs are not supplied by the dataset. Zero means an explicit zero-cost assumption. Revenue is the optimization objective."
            )
        submitted = st.form_submit_button("Compare scenario", type="primary")
    selection_key = (product_id, store_id)
    if submitted:
        payload = {
            "product_id": product_id,
            "store_id": store_id,
            "support": support,
            "reference_price": reference,
            "unit_cost": unit_cost if include_cost else None,
            "promotion_cost": promo_cost if include_cost else 0,
            "baseline_promotion_cost": base_cost if include_cost else 0,
        }
        payload["price" if mode == "Shelf price" else "discount_pct"] = (
            price if mode == "Shelf price" else discount
        )
        result = api("/scenarios", payload)
        st.session_state["comparison"] = (selection_key, result)
    saved = st.session_state.get("comparison")
    if saved and saved[0] == selection_key and saved[1]:
        result = saved[1]
        cols = st.columns(3)
        cols[0].metric(
            "Simulated units",
            f"{result['proposed']['units']:.1f}",
            f"{result['changes']['units_pct']:+.1f}%",
        )
        cols[1].metric(
            "Simulated revenue",
            f"${result['proposed']['revenue']:.2f}",
            f"{result['changes']['revenue_pct']:+.1f}%",
        )
        lift = result["support_lift_at_same_price_pct"]
        cols[2].metric(
            "Support lift at same price", f"{lift:+.1f}%" if lift is not None else "Unavailable"
        )
        if result["warnings"]:
            st.warning("Exploratory only: " + " ".join(result["warnings"]))
        table = pd.DataFrame(
            [
                {"Scenario": label, **result[key]}
                for label, key in [("Baseline", "baseline"), ("Proposed", "proposed")]
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        curve = pd.DataFrame(result["curve"])
        if not curve.empty:
            chart(
                px.line(
                    curve,
                    x="price",
                    y="revenue",
                    markers=True,
                    title="Simulated revenue at the selected support settings",
                    color_discrete_sequence=COLORS,
                )
            )
            chart(
                px.line(
                    curve,
                    x="price",
                    y="units",
                    title="Simulated demand curve",
                    color_discrete_sequence=COLORS[1:],
                )
            )
        export = table.copy()
        export["model_version"] = result["model_version"]
        export["simulation_week"] = result["week"]
        export["assumptions"] = json.dumps(result["assumptions"])
        export["warnings"] = json.dumps(result["warnings"])
        export["disclaimer"] = result["disclaimer"]
        st.download_button(
            "Download comparison CSV",
            export.to_csv(index=False),
            "pricing_scenario.csv",
            "text/csv",
        )
        st.caption(f"Saved scenario: {result['scenario_id']}")
        st.caption(result["disclaimer"])
    st.divider()
    st.subheader("Find the best feasible candidate")
    with st.form("optimizer"):
        cols = st.columns(3)
        cap = cols[0].slider("Maximum discount (%)", 0, 50, 20)
        cost_enabled = cols[1].checkbox("Apply cost and margin assumptions")
        opt_cost = cols[1].number_input(
            "Unit cost for optimization ($)", min_value=0.0, value=0.0, step=0.1
        )
        margin = cols[2].slider("Minimum contribution margin (%)", 0, 90, 0)
        opt_promo = cols[2].number_input(
            "Weekly cost per promoted candidate ($)", min_value=0.0, value=0.0, step=1.0
        )
        opt_base = cols[2].number_input(
            "Baseline weekly cost ($)", min_value=0.0, value=0.0, step=1.0
        )
        run = st.form_submit_button("Evaluate candidate prices")
    if run:
        result = api(
            "/recommendations",
            {
                "product_id": product_id,
                "store_id": store_id,
                "max_discount_pct": cap,
                "unit_cost": opt_cost if cost_enabled else None,
                "min_margin_pct": margin if cost_enabled else None,
                "promotion_cost": opt_promo if cost_enabled else 0,
                "baseline_promotion_cost": opt_base if cost_enabled else 0,
            },
        )
        if result:
            if result["status"] == "recommended":
                best = result["recommendation"]
                st.success(
                    f"Best tested candidate: ${best['price']:.2f} · {support_label(best['support'])}. Simulated revenue change: {result['simulated_revenue_change_pct']:+.1f}%."
                )
            elif result["status"] == "infeasible":
                st.warning(
                    "No candidates satisfy these constraints. Adjust the budget assumptions, margin, or discount limit."
                )
            else:
                st.warning(
                    "Insufficient evidence for an automatic recommendation. "
                    + " ".join(result["warnings"])
                )
            st.caption(result["note"])
            st.caption(
                "Promotion costs default to zero unless supplied. Support options are limited to observed combinations."
            )
            st.dataframe(pd.DataFrame(result["candidates"]), hide_index=True, width="stretch")
            st.download_button(
                "Download candidate results",
                json.dumps(result, indent=2),
                "recommendation.json",
                "application/json",
            )

else:
    summary = get("/models/metrics")
    if not summary:
        st.stop()
    st.write(
        "Models use the first 104 weeks. The next 26 weeks gate recommendations. The final 26 weeks report held-out performance."
    )
    scope = st.radio(
        "Evaluation scope", ["All modeled products", "Selected product"], horizontal=True
    )
    rows = []
    for split in ["validation", "test"]:
        scores = (
            summary["metrics"][split]["overall"]
            if scope == "All modeled products"
            else summary["metrics"][split]["by_product"].get(product_id, {})
        )
        for model, score in scores.items():
            rows.append(
                {
                    "Split": split,
                    "Model": model,
                    "MAE (units)": score["mae"],
                    "WAPE (%)": 100 * score["wape"] if score["wape"] is not None else None,
                    "Bias (units)": score["bias"],
                    "Rows": score["rows"],
                }
            )
    scores = pd.DataFrame(rows)
    chart(
        px.bar(
            scores,
            x="Model",
            y="WAPE (%)",
            color="Split",
            barmode="group",
            title="Forecast error · lower is better",
            color_discrete_sequence=COLORS,
        )
    )
    st.dataframe(scores, hide_index=True, width="stretch")
    prediction_history = pd.DataFrame(analysis.get("prediction_history", []))
    if not prediction_history.empty:
        chart(
            px.line(
                prediction_history,
                x="week",
                y=["units", "baseline", "log_log", "xgboost"],
                title="Held-out actual and predicted units · selected product and store",
                color_discrete_sequence=["#253b35", "#e6a33b", "#16796d", "#597cb5"],
            )
        )
    st.caption(
        "Positive bias means overprediction. All models are compared on the same eligible rows. The rolling baseline uses only earlier observed sales, updated one week at a time."
    )
    st.subheader("What this evaluation does—and does not—show")
    st.write(
        "Forecast accuracy at observed prices does not validate the effect of an unobserved intervention. XGBoost is a predictive challenger and never drives the price simulator. No model is refitted on the test set."
    )
    st.write(
        "Missing weeks are not zero sales. Prices may reflect unobserved demand changes, and inventory is unavailable. Confidence intervals exclude these sources of uncertainty. Cross-product substitution and promotion carryover are outside this version."
    )
    st.caption(f"Model run: {summary['version']} · Training ended {summary['training_end']}")
