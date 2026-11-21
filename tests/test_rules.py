"""Phase 2 tests: one test per rule, zero-eligible edge case, expression safety."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from procure.config import load_rules_config
from procure.domain import ProcurementRequest
from procure.rules.engine import RuleError, apply_rules, safe_eval

PARAMS = load_rules_config()["params"]


def _req(method: str = "selective_tendering") -> ProcurementRequest:
    return ProcurementRequest(item="Cement 50kg bag", item_category="construction", quantity=100,
                              delivery_state="Yobe", delivery_lga="Damaturu", method=method,
                              budget=5e6, date=date(2024, 3, 1), request_id="REQ-T")


def _sup(sid: str, **kw: object) -> dict:
    base = {"supplier_id": sid, "registered": True, "debarred": False, "tax_clearance": True, "cac": True,
            "past_failure_rate": 0.1, "bid_value": 1e6}
    return {**base, **kw}


def _run(*sups: dict, method: str = "selective_tendering"):
    return apply_rules(_req(method), pd.DataFrame(list(sups)))


def _ids(res) -> set[str]:
    return set(res.eligible["supplier_id"])


def test_r1_unregistered_or_debarred_excluded() -> None:
    res = _run(_sup("A", registered=False), _sup("B", debarred=True), _sup("C"), _sup("D"))
    assert _ids(res) == {"C", "D"}
    assert all(e.rule_ids == ["R1"] for e in res.excluded)


def test_r2_value_above_method_threshold_excluded() -> None:
    limit = PARAMS["method_thresholds"]["request_for_quotation"]
    res = _run(_sup("A", bid_value=limit + 1), _sup("B", bid_value=limit - 1), _sup("C"), method="request_for_quotation")
    assert _ids(res) == {"B", "C"}
    assert res.excluded[0].rule_ids == ["R2"] and "above" in res.excluded[0].reasons[0]


def test_r2_unknown_method_is_an_error() -> None:
    with pytest.raises(RuleError, match="no_such_method"):
        _run(_sup("A"), method="no_such_method")


def test_r3_missing_tax_or_cac_excluded() -> None:
    res = _run(_sup("A", tax_clearance=False), _sup("B", cac=False), _sup("C"), _sup("D"))
    assert _ids(res) == {"C", "D"} and {e.rule_ids[0] for e in res.excluded} == {"R3"}


def test_r4_single_eligible_supplier_flagged() -> None:
    res = _run(_sup("A"), _sup("B", registered=False))
    assert _ids(res) == {"A"}
    assert any(f.rule_id == "R4" and f.supplier_id is None for f in res.flags)
    assert not any(f.rule_id == "R4" for f in _run(_sup("A"), _sup("B")).flags)


def test_r5_high_failure_rate_flagged_not_excluded() -> None:
    res = _run(_sup("A", past_failure_rate=PARAMS["max_failure_rate"] + 0.01), _sup("B"))
    assert _ids(res) == {"A", "B"}
    assert [f.supplier_id for f in res.flags if f.rule_id == "R5"] == ["A"]


def test_zero_eligible_suppliers() -> None:
    res = _run(_sup("A", registered=False), _sup("B", cac=False))
    assert res.eligible.empty and len(res.excluded) == 2
    assert [f.rule_id for f in res.flags] == ["R6"]
    assert not any(f.rule_id == "R4" for f in res.flags)


def test_multiple_reasons_kept_and_every_fired_rule_logged() -> None:
    res = _run(_sup("A", registered=False, cac=False, past_failure_rate=0.9), _sup("B"), _sup("C"))
    ex = res.excluded[0]
    assert ex.rule_ids == ["R1", "R3"] and len(ex.reasons) == 2
    assert {f["rule_id"] for f in res.fired} == {"R1", "R3", "R5"}
    assert not res.flags  # R5 flag on an excluded supplier is not shown to the officer


def test_safe_eval_blocks_code_execution() -> None:
    for bad in ["__import__('os')", "supplier.__class__", "open('x')", "[x for x in ()]"]:
        with pytest.raises(RuleError):
            safe_eval(bad, {"supplier": {"a": 1}})
    assert safe_eval("params.t['x'] > 1 and not flag", {"params": {"t": {"x": 2}}, "flag": False}) is True


def test_rule_params_are_numeric() -> None:
    assert all(isinstance(v, (int, float)) for v in PARAMS["method_thresholds"].values())
    assert isinstance(PARAMS["max_failure_rate"], float)
