"""Phase 5 tests: end-to-end recommendation, explanations, narration, LIME agreement."""

from __future__ import annotations

import re

import numpy as np

from procure.config import load_settings
from procure.explain.lime_check import lime_agreement
from procure.explain.narrate import narrate_model
from procure.pipeline import recommend

JARGON = re.compile(r"shap|log-odds|log_|_rate|feature|margin", re.I)


def test_end_to_end_under_latency_budget(sample_request, tiny_engine) -> None:
    res = recommend(sample_request, tiny_engine)
    assert res.runtime_seconds < load_settings()["api"]["latency_budget_seconds"]
    cfg = load_settings()["explain"]
    assert cfg["min_options"] <= len(res.options) <= cfg["max_options"]
    assert [o.rank for o in res.options] == list(range(1, len(res.options) + 1))
    for o in res.options:
        assert np.isclose(sum(a.share for a in o.allocations), 1.0)
        assert len(o.allocations) <= load_settings()["optimise"]["k_max"]
        assert o.summary and ("within budget" in o.summary or "OVER BUDGET" in o.summary)
        for a in o.allocations:
            for m in ("cost", "delay", "failure"):
                assert len(a.reasons[m]) == 1 + cfg["top_k"]
                assert len(a.shap_top[m]) == cfg["top_k"]
                assert not JARGON.search(" ".join(a.reasons[m])), a.reasons[m]
    assert res.pareto_front and {"cost", "worst_delay", "quality", "failure_prob"} <= set(res.pareto_front[0])
    assert res.to_dict()["request"]["date"] == str(sample_request.date)


def test_no_eligible_suppliers_gives_message(sample_request, tiny_engine) -> None:
    from dataclasses import replace

    req = replace(sample_request, candidate_supplier_ids=["SUP-doesnotexist"])
    res = recommend(req, tiny_engine)
    assert res.options == [] and "No supplier is eligible" in res.message


def test_narration_example_matches_spec_style() -> None:
    text, reasons = narrate_model(
        "delay", prediction=19.0, baseline=13.0,
        shap_row={"rainy_season": 4.0, "conflict_events_90d": 2.5, "distance_km": -0.5, "n_bidders": 0.1},
        values={"rainy_season": 1, "conflict_events_90d": 31, "distance_km": 40.0, "n_bidders": 4},
        ctx={"delivery_lga": "Gwoza", "supplier_lga": "Bama"}, margin=0.0, failure_weight=1.0)
    assert text.startswith("Delay is predicted at 19 days, 6 days longer than a typical order, mainly because of "
                           "rainy-season delivery (May to October) to Gwoza")
    assert "works the other way" in text and len(reasons) == 3
    assert reasons[0]["sentence"].endswith("adds about 4.0 days.")


def test_cost_and_failure_narration_units() -> None:
    t, r = narrate_model("cost", 110.0, 100.0, {"past_price_index": np.log(1.1), "item": 2.0},
                         {"past_price_index": np.log(1.1), "item": "Rice"}, {"delivery_lga": "Yola North"}, 0.0, 1.0)
    assert "10% above the typical N100" in t and all(x["feature"] != "item" for x in r)
    t, r = narrate_model("failure", 0.3, 0.2, {"past_failure_rate": 0.4}, {"past_failure_rate": 0.35},
                         {"delivery_lga": "Bama"}, 0.0, 2.0)
    assert "30%" in t and "percentage points" in r[0]["sentence"]


def test_lime_agreement_runs(tiny_bundle, small_frames) -> None:
    from procure.explain.shap_explain import ShapExplainer
    from procure.models.predict import Predictor

    p = Predictor(tiny_bundle)
    a = lime_agreement(p, ShapExplainer(tiny_bundle), p.transform(small_frames["train"]),
                       p.transform(small_frames["test"]), n=3, num_samples=200)
    assert set(a["model"]) == {"cost", "delay", "failure"} and a["agreement"].between(0, 1).all()
