"""Rule layer: compliance / eligibility filter driven by config/rules.yaml.

Conditions are evaluated with a small, safe expression evaluator (Python AST whitelist): no
function calls, imports or dunder access are possible, so rules can be edited by
administrators without touching code. Every fired rule is logged with its reason.
"""

from __future__ import annotations

import ast
import logging
import operator
import re
from dataclasses import dataclass
from typing import Any, NamedTuple

import pandas as pd

from procure.config import load_rules_config
from procure.domain import ExcludedSupplier, ProcurementRequest, RuleFlag

log = logging.getLogger("procure.rules")


class RuleError(ValueError):
    """Raised when a rule is malformed or refers to unknown data."""


class Namespace:
    """Attribute-style read access over a dict (supplier.registered, params.max_failure_rate)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, key: str) -> Any:
        """Return a value; wrap nested dicts so dotted access keeps working."""
        if key not in self._data:
            raise RuleError(f"unknown field '{key}'")
        v = self._data[key]
        return Namespace(v) if isinstance(v, dict) else v

    def __getitem__(self, key: Any) -> Any:
        return self.get(key)


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


def safe_eval(expr: str, variables: dict[str, Any]) -> Any:
    """Evaluate a whitelisted expression (bool ops, comparisons, arithmetic, attribute/key access)."""

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            vals = (ev(v) for v in node.values)
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not ev(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise RuleError(f"unknown variable '{node.id}'")
            v = variables[node.id]
            return Namespace(v) if isinstance(v, dict) else v
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise RuleError("private attributes are not allowed")
            obj = ev(node.value)
            if not isinstance(obj, Namespace):
                raise RuleError(f"cannot read '{node.attr}'")
            return obj.get(node.attr)
        if isinstance(node, ast.Subscript):
            obj = ev(node.value)
            return obj[ev(node.slice)]
        if isinstance(node, ast.Constant):
            return node.value
        raise RuleError(f"expression element not allowed: {type(node).__name__}")

    try:
        return ev(ast.parse(expr, mode="eval"))
    except SyntaxError as e:
        raise RuleError(f"invalid rule expression '{expr}': {e}") from e


_PLACEHOLDER = re.compile(r"\{([^{}:]+)(?::([^{}]*))?\}")


def render_reason(template: str, variables: dict[str, Any]) -> str:
    """Fill {expression[:format]} placeholders using safe_eval."""

    def sub(m: re.Match) -> str:
        val = safe_eval(m.group(1).strip(), variables)
        return format(val, m.group(2)) if m.group(2) else str(val)

    return _PLACEHOLDER.sub(sub, template)


@dataclass
class Rule:
    """One rule from rules.yaml."""

    id: str
    description: str
    scope: str
    condition: str
    action: str
    reason: str

    def __post_init__(self) -> None:
        if self.action not in {"exclude", "flag"}:
            raise RuleError(f"{self.id}: action must be exclude or flag")
        if self.scope not in {"supplier", "set"}:
            raise RuleError(f"{self.id}: scope must be supplier or set")
        ast.parse(self.condition, mode="eval")


class RuleOutcome(NamedTuple):
    """(eligible, excluded_with_reasons, flags) plus the full fired-rule log."""

    eligible: pd.DataFrame
    excluded: list[ExcludedSupplier]
    flags: list[RuleFlag]
    fired: list[dict[str, Any]]


def load_rules(cfg: dict[str, Any] | None = None) -> tuple[list[Rule], dict[str, Any]]:
    """Parse rules.yaml into Rule objects and the params block."""
    cfg = cfg or load_rules_config()
    return [Rule(**r) for r in cfg["rules"]], cfg.get("params", {})


def _clean(v: Any) -> Any:
    """Convert numpy scalars to python for safe evaluation and formatting."""
    return v.item() if hasattr(v, "item") else v


def apply_rules(request: ProcurementRequest, suppliers: pd.DataFrame,
                cfg: dict[str, Any] | None = None) -> RuleOutcome:
    """Apply every rule. Supplier-scope rules run per candidate; set-scope rules run once on the
    eligible set. Excluded suppliers keep all reasons that fired (not just the first)."""
    rules, params = load_rules(cfg)
    req = request.to_dict()
    excluded: dict[str, ExcludedSupplier] = {}
    flags: list[RuleFlag] = []
    fired: list[dict[str, Any]] = []
    for _, row in suppliers.iterrows():
        sup = {k: _clean(v) for k, v in row.items()}
        env = {"supplier": sup, "request": req, "params": params}
        for rule in (r for r in rules if r.scope == "supplier"):
            if not safe_eval(rule.condition, env):
                continue
            reason = render_reason(rule.reason, env)
            fired.append({"rule_id": rule.id, "action": rule.action, "supplier_id": sup["supplier_id"], "reason": reason})
            log.info("rule %s fired (%s) for %s: %s", rule.id, rule.action, sup["supplier_id"], reason)
            if rule.action == "exclude":
                ex = excluded.setdefault(sup["supplier_id"], ExcludedSupplier(sup["supplier_id"], [], []))
                ex.rule_ids.append(rule.id)
                ex.reasons.append(reason)
            else:
                flags.append(RuleFlag(rule.id, reason, sup["supplier_id"]))
    eligible = suppliers[~suppliers["supplier_id"].isin(excluded)].reset_index(drop=True)
    # flags on excluded suppliers are irrelevant to the officer's choice
    flags = [f for f in flags if f.supplier_id not in excluded]
    env = {"n_eligible": len(eligible), "request": req, "params": params}
    for rule in (r for r in rules if r.scope == "set"):
        if safe_eval(rule.condition, env):
            reason = render_reason(rule.reason, env)
            fired.append({"rule_id": rule.id, "action": rule.action, "supplier_id": None, "reason": reason})
            log.info("rule %s fired (%s): %s", rule.id, rule.action, reason)
            if rule.action == "flag":
                flags.append(RuleFlag(rule.id, reason))
            else:
                excluded.update({s: ExcludedSupplier(s, [rule.id], [reason]) for s in eligible["supplier_id"]})
                eligible = eligible.iloc[0:0]
    return RuleOutcome(eligible, list(excluded.values()), flags, fired)
