"""Streamlit dashboard for procurement officers (make dashboard).

Login -> request form -> ranked option cards -> Pareto chart (cost vs delay, coloured by risk)
-> per-supplier explanation bars -> fired rules -> decision buttons (accept / override / return).
Uses the same service layer as the REST API, so every action lands in the audit log.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from procure.config import load_rules_config, load_settings
from procure.data import geo
from procure.db.models import AppUser
from procure.db.session import make_engine, make_sessionmaker, seed_reference, seed_users
from procure.domain import ProcurementRequest
from procure.explain.narrate import FEATURE_LABELS, METHOD_NAMES
from procure.pipeline import OPTION_TITLES, Engine
from procure import service

st.set_page_config(page_title="Procurement Recommendations - North-East Nigeria", layout="wide")


@st.cache_resource
def resources() -> tuple[Engine, object]:
    """Load models/data once and prepare the database."""
    eng = Engine.from_disk()
    factory = make_sessionmaker(make_engine())
    with factory() as s:
        seed_users(s)
        seed_reference(s, eng.store.transactions, eng.store.registry())
        s.commit()
    return eng, factory


engine, factory = resources()
reg = engine.store.registry().set_index("supplier_id")


def name_of(sid: str) -> str:
    """Business name for display (supplier table is access-controlled; never logged)."""
    return f"{reg.loc[sid, 'display_name']} ({sid})" if sid in reg.index else sid


# ---------------- login ----------------
if "user_id" not in st.session_state:
    st.title("Procurement recommendation system")
    st.caption("Explainable AI support for procurement officers - the officer always makes the final decision.")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            with factory() as s:
                user = service.authenticate(s, u, p)
                s.commit()
            if user:
                st.session_state.update(user_id=user.id, role=user.role)
                st.rerun()
            st.error("Wrong username or password.")
    st.stop()


def current_user(s) -> AppUser:
    return s.get(AppUser, st.session_state["user_id"])


perms = service.permissions(st.session_state["role"])
st.sidebar.write(f"Signed in as **{st.session_state['role'].replace('_', ' ')}**")
if st.sidebar.button("Log out"):
    st.session_state.clear()
    st.rerun()
page = st.sidebar.radio("Page", ["New request"] + (["Audit log"] if "view_audit" in perms else []))

# ---------------- audit page ----------------
if page == "Audit log":
    st.header("Audit log")
    with factory() as s:
        rows = service.list_audit(s, current_user(s), limit=500)
        s.commit()
    df = pd.DataFrame(rows)
    if not df.empty:
        kind = st.multiselect("Event types", sorted(df.event_type.unique()), default=sorted(df.event_type.unique()))
        st.dataframe(df[df.event_type.isin(kind)], use_container_width=True, hide_index=True)
    st.stop()

# ---------------- request form ----------------
st.title("New procurement request")
tx = engine.store.transactions
items = tx.groupby("item")["item_category"].first().sort_index()
methods = list(load_rules_config()["params"]["method_thresholds"])
with st.form("request"):
    c1, c2, c3 = st.columns(3)
    item = c1.selectbox("Item", items.index)
    qty = c1.number_input("Quantity (units)", min_value=1.0, value=200.0, step=10.0)
    state = c2.selectbox("Delivery state", load_settings()["region"]["states"], index=2)
    lga = c2.selectbox("Delivery LGA", geo.lga_table().query("state == @state").lga.tolist())
    method = c3.selectbox("Procurement method", methods, format_func=lambda m: METHOD_NAMES.get(m, m))
    budget = c3.number_input("Budget (naira, 2024 prices)", min_value=1.0, value=20_000_000.0, step=500_000.0)
    when = c1.date_input("Request date", value=dt.date(2024, 7, 15))
    k = c2.slider("Maximum suppliers to split the order between", 1, 5, load_settings()["optimise"]["k_max"])
    go_btn = st.form_submit_button("Get recommendations", disabled="recommend" not in perms)
if go_btn:
    req = ProcurementRequest(item=item, item_category=items[item], quantity=qty, delivery_state=state,
                             delivery_lga=lga, method=method, budget=budget, date=when, k_max=k)
    with st.spinner("Checking rules, predicting and optimising..."):
        with factory() as s:
            res = service.create_recommendation(s, current_user(s), req, engine)
            s.commit()
    st.session_state["result"] = res.to_dict()

res = st.session_state.get("result")
if not res:
    st.stop()

st.subheader(f"Recommendation {res['recommendation_id']}")
st.caption(f"{res['n_eligible']} of {res['n_candidates']} suppliers passed the compliance rules | "
           f"computed in {res['runtime_seconds']:.1f} s | model {res['model_version']} | costs in 2024 naira")
if res["message"]:
    st.warning(res["message"])

# ---------------- fired rules ----------------
with st.expander(f"Compliance rules: {len(res['excluded'])} excluded, {len(res['flags'])} warnings",
                 expanded=bool(res["flags"])):
    for e in res["excluded"]:
        st.error(f"Excluded {name_of(e['supplier_id'])}: " + " ".join(e["reasons"]))
    for f in res["flags"]:
        st.warning(f"[{f['rule_id']}] {f['reason']}")

if not res["options"]:
    st.stop()

# ---------------- option cards ----------------
cols = st.columns(len(res["options"]))
for col, o in zip(cols, res["options"]):
    with col:
        title = OPTION_TITLES.get(o["label"], o["label"])
        st.markdown(f"### #{o['rank']} {title}")
        if o["also_best_for"]:
            st.caption("Also: " + ", ".join(OPTION_TITLES.get(b, b) for b in o["also_best_for"]))
        st.metric("Total cost", f"N{o['total_cost']:,.0f}", "within budget" if o["within_budget"] else "over budget",
                  delta_color="normal" if o["within_budget"] else "inverse")
        st.metric("Slowest delivery (days late)", f"{o['worst_delay_days']:.0f}")
        st.metric("Chance any supplier fails", f"{o['failure_prob']:.0%}")
        st.metric("Average quality (1-5)", f"{o['quality']:.1f}")
        for a in o["allocations"]:
            st.markdown(f"**{a['share']:.0%}** - {name_of(a['supplier_id'])}: {a['quantity']:,.0f} units "
                        f"at N{a['unit_cost']:,.0f}")
            for fl in a["flags"]:
                st.caption(f"Warning: {fl}")

# ---------------- Pareto chart ----------------
st.subheader("All good trade-offs found (Pareto front)")
pf = pd.DataFrame(res["pareto_front"])
fig = px.scatter(pf, x="cost", y="worst_delay", color="failure_prob", color_continuous_scale="RdYlGn_r",
                 labels={"cost": "Total cost (N)", "worst_delay": "Slowest delivery (days late)",
                         "failure_prob": "Failure risk"}, opacity=0.7)
fig.update_traces(marker=dict(size=9))
opt = pd.DataFrame([{"cost": o["total_cost"], "worst_delay": o["worst_delay_days"],
                     "label": f"#{o['rank']} {OPTION_TITLES.get(o['label'], o['label'])}"} for o in res["options"]])
fig.add_trace(go.Scatter(x=opt.cost, y=opt.worst_delay, mode="markers+text", text=opt.label, textposition="top center",
                         marker=dict(size=14, symbol="star", color="black"), name="Recommended options"))
budget_value = res["request"]["budget"]
if budget_value <= 1.5 * pf["cost"].max():
    fig.add_vline(x=budget_value, line_dash="dash", annotation_text="Budget")
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Each dot is an allocation no other allocation beats on every goal at once. Budget: N{budget_value:,.0f}"
           + ("" if budget_value <= 1.5 * pf["cost"].max() else " (well above every option shown)."))

# ---------------- explanations ----------------
st.subheader("Why? Explanations per supplier")
choice = st.selectbox("Option", [o["rank"] for o in res["options"]],
                      format_func=lambda r: f"#{r} {OPTION_TITLES.get(res['options'][r - 1]['label'])}")
o = res["options"][choice - 1]
st.info(o["summary"])
for a in o["allocations"]:
    st.markdown(f"#### {name_of(a['supplier_id'])} - {a['share']:.0%} of the order")
    tabs = st.tabs(["Cost", "Delay", "Failure risk"])
    for tab, m, unit in zip(tabs, ["cost", "delay", "failure"], ["% on price", "days", "percentage points"]):
        with tab:
            st.write(a["reasons"][m][0])
            top = pd.DataFrame(a["shap_top"][m])
            top["label"] = top["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
            top["direction"] = top["effect"].map(lambda e: "makes it worse" if e > 0 else "makes it better")
            bar = px.bar(top, x="effect", y="label", orientation="h", color="direction",
                         color_discrete_map={"makes it worse": "#c0392b", "makes it better": "#27ae60"},
                         labels={"effect": f"Effect ({unit})", "label": "", "direction": ""})
            bar.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(bar, use_container_width=True)
            for sentence in a["reasons"][m][1:]:
                st.caption("- " + sentence)

# ---------------- decision ----------------
st.subheader("Your decision")
if "decide" not in perms:
    st.caption("Your role can view recommendations but not record decisions.")
    st.stop()
c1, c2, c3 = st.columns(3)
rank = c1.selectbox("Option to accept", [o["rank"] for o in res["options"]])
reason = c2.text_area("Reason (required to override or return)")
override_to = c3.multiselect("Override: supplier(s) chosen instead",
                             sorted(set(reg.index) - {e["supplier_id"] for e in res["excluded"]}), format_func=name_of)
b1, b2, b3 = st.columns(3)
action = None
if b1.button("Accept option", type="primary"):
    action = "accept"
if b2.button("Override with reason"):
    action = "override"
if b3.button("Return for revision"):
    action = "return"
if action:
    try:
        with factory() as s:
            out = service.record_decision(s, current_user(s), res["recommendation_id"], action, rank, reason, override_to)
            s.commit()
        st.success(f"Decision recorded: {out['status']}. It has been written to the audit log.")
    except service.ServiceError as e:
        st.error(str(e))
