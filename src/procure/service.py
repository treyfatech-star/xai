"""Application service shared by the Flask API and the Streamlit dashboard.

Every recommendation, explanation, fired rule and officer decision is written to audit_log.
Audit payloads contain IDs, numbers and reasons only - never personal names (NDPA 2023).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash

from procure.config import load_settings
from procure.db.models import AppUser, AuditLog, ModelOutput
from procure.domain import ProcurementRequest, RecommendationResult
from procure.pipeline import Engine, recommend

DECISIONS: set[str] = {"accept", "override", "return"}


class ServiceError(ValueError):
    """Bad input or a forbidden action; message is safe to show to the user."""


def permissions(role: str) -> set[str]:
    """Actions a role may perform (config api.permissions)."""
    return set(load_settings()["api"]["permissions"].get(role, []))


def audit(session: Session, event_type: str, payload: dict[str, Any], user: AppUser | None = None,
          recommendation_id: str | None = None) -> None:
    """Append one audit_log row."""
    session.add(AuditLog(event_type=event_type, recommendation_id=recommendation_id,
                         user_id=user.id if user else None, role=user.role if user else None, payload=payload))


def authenticate(session: Session, username: str, password: str) -> AppUser | None:
    """Return the active user if the password matches; logs the attempt (no password stored)."""
    user = session.scalar(select(AppUser).where(AppUser.username == username, AppUser.active.is_(True)))
    ok = user is not None and check_password_hash(user.password_hash, password)
    audit(session, "login", {"success": ok}, user if ok else None)
    return user if ok else None


def create_recommendation(session: Session, user: AppUser, request: ProcurementRequest,
                          engine: Engine | None = None) -> RecommendationResult:
    """Run the pipeline, persist model_output and write the audit trail."""
    if "recommend" not in permissions(user.role):
        raise ServiceError("Your role may not create recommendations.")
    res = recommend(request, engine)
    data = res.to_dict()
    session.add(ModelOutput(id=res.recommendation_id, request_id=request.request_id, created_by=user.id,
                            model_version=res.model_version, runtime_seconds=res.runtime_seconds,
                            request_json=data["request"], result_json=data))
    rid = res.recommendation_id
    audit(session, "recommendation", {
        "request_id": request.request_id, "n_candidates": res.n_candidates, "n_eligible": res.n_eligible,
        "options": [{"rank": o.rank, "label": o.label, "total_cost": o.total_cost, "worst_delay_days": o.worst_delay_days,
                     "failure_prob": o.failure_prob, "suppliers": {a.supplier_id: a.share for a in o.allocations}}
                    for o in res.options],
        "model_version": res.model_version, "runtime_seconds": res.runtime_seconds, "message": res.message}, user, rid)
    for ex in res.excluded:
        for r_id, why in zip(ex.rule_ids, ex.reasons):
            audit(session, "rule_fired", {"rule_id": r_id, "action": "exclude", "supplier_id": ex.supplier_id,
                                          "reason": why}, user, rid)
    for f in res.flags:
        audit(session, "rule_fired", {"rule_id": f.rule_id, "action": "flag", "supplier_id": f.supplier_id,
                                      "reason": f.reason}, user, rid)
    for o in res.options:
        audit(session, "explanation", {"rank": o.rank, "label": o.label, "summary": o.summary,
                                       "reasons": {a.supplier_id: a.reasons for a in o.allocations}}, user, rid)
    return res


def get_recommendation(session: Session, user: AppUser, rec_id: str) -> dict[str, Any]:
    """Stored recommendation (and its status); the view itself is audited."""
    if "view_recommendation" not in permissions(user.role):
        raise ServiceError("Your role may not view recommendations.")
    mo = session.get(ModelOutput, rec_id)
    if mo is None:
        raise ServiceError(f"Recommendation {rec_id} not found.")
    audit(session, "view", {}, user, rec_id)
    return {**mo.result_json, "status": mo.status}


def record_decision(session: Session, user: AppUser, rec_id: str, action: str, option_rank: int | None = None,
                    reason: str | None = None, chosen_supplier_ids: list[str] | None = None) -> dict[str, Any]:
    """Officer decision: accept an option, override (reason required) or return for revision (reason required)."""
    if "decide" not in permissions(user.role):
        raise ServiceError("Your role may not record decisions.")
    if action not in DECISIONS:
        raise ServiceError(f"action must be one of {sorted(DECISIONS)}")
    mo = session.get(ModelOutput, rec_id)
    if mo is None:
        raise ServiceError(f"Recommendation {rec_id} not found.")
    if mo.status != "pending":
        raise ServiceError(f"A decision ({mo.status}) was already recorded for {rec_id}.")
    ranks = [o["rank"] for o in mo.result_json["options"]]
    if action == "accept" and option_rank not in ranks:
        raise ServiceError(f"option_rank must be one of {ranks} to accept.")
    if action in {"override", "return"} and not (reason and reason.strip()):
        raise ServiceError(f"A reason is required to {action}.")
    if action == "override" and chosen_supplier_ids:
        excluded = {e["supplier_id"] for e in mo.result_json["excluded"]}
        bad = sorted(set(chosen_supplier_ids) & excluded)
        if bad:
            raise ServiceError(f"Cannot override to excluded supplier(s) {bad}; they failed a compliance rule.")
    mo.status = {"accept": "accepted", "override": "overridden", "return": "returned"}[action]
    payload = {"action": action, "option_rank": option_rank, "reason": (reason or "").strip(),
               "chosen_supplier_ids": chosen_supplier_ids or []}
    audit(session, "decision", payload, user, rec_id)
    return {"recommendation_id": rec_id, "status": mo.status, **payload}


def list_audit(session: Session, user: AppUser, recommendation_id: str | None = None,
               event_type: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Audit entries, newest first (administrators only)."""
    if "view_audit" not in permissions(user.role):
        raise ServiceError("Only administrators may read the audit log.")
    q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
    if recommendation_id:
        q = q.where(AuditLog.recommendation_id == recommendation_id)
    if event_type:
        q = q.where(AuditLog.event_type == event_type)
    return [{"id": a.id, "timestamp": a.timestamp.isoformat(), "event_type": a.event_type,
             "recommendation_id": a.recommendation_id, "user_id": a.user_id, "role": a.role, "payload": a.payload}
            for a in session.scalars(q)]
