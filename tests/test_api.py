"""Phase 5 API tests: login, role-based access, recommend, view, decisions, audit trail."""

from __future__ import annotations

import pytest

from procure.api.app import create_app

PW = "test-pass"


@pytest.fixture(scope="module")
def client(tmp_path_factory, tiny_engine):
    db = tmp_path_factory.mktemp("db") / "t.db"
    app = create_app(db_url=f"sqlite:///{db}", engine=tiny_engine, seed_password=PW)
    app.config["TESTING"] = True
    return app.test_client()


def _token(client, user: str) -> dict:
    r = client.post("/login", json={"username": user, "password": PW})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.get_json()['token']}"}


def _body(req) -> dict:
    return {"item": req.item, "item_category": req.item_category, "quantity": req.quantity,
            "delivery_state": req.delivery_state, "delivery_lga": req.delivery_lga, "method": req.method,
            "budget": req.budget, "date": str(req.date)}


@pytest.fixture(scope="module")
def rec(client, sample_request) -> dict:
    r = client.post("/recommend", json=_body(sample_request), headers=_token(client, "officer1"))
    assert r.status_code == 201, r.get_json()
    return r.get_json()


def test_login_rejects_bad_password(client) -> None:
    assert client.post("/login", json={"username": "officer1", "password": "nope"}).status_code == 401
    assert client.get("/health").get_json() == {"status": "ok"}


def test_auth_required_and_roles(client, sample_request) -> None:
    assert client.post("/recommend", json=_body(sample_request)).status_code == 401
    assert client.get("/audit", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert client.get("/audit", headers=_token(client, "officer1")).status_code == 403
    assert client.post("/decision", json={}, headers=_token(client, "logistics1")).status_code == 403


def test_validation_errors_are_plain(client) -> None:
    r = client.post("/recommend", json={"item": "x", "item_category": "food", "quantity": -1, "delivery_state": "Lagos",
                                        "delivery_lga": "Ikeja", "method": "magic", "budget": 10},
                    headers=_token(client, "officer1"))
    assert r.status_code == 400
    details = " ".join(r.get_json()["details"])
    assert "positive" in details and "delivery_state" in details and "method" in details


def test_recommend_returns_explained_options(rec) -> None:
    assert rec["recommendation_id"].startswith("REC-") and len(rec["options"]) >= 3
    assert rec["options"][0]["allocations"][0]["reasons"]["delay"][0].startswith("Delay is predicted")


def test_view_and_decisions(client, rec) -> None:
    rid = rec["recommendation_id"]
    got = client.get(f"/recommendation/{rid}", headers=_token(client, "logistics1")).get_json()
    assert got["status"] == "pending"
    off = _token(client, "officer1")
    r = client.post("/decision", json={"recommendation_id": rid, "action": "override"}, headers=off)
    assert r.status_code == 400 and "reason" in r.get_json()["error"]
    r = client.post("/decision", json={"recommendation_id": rid, "action": "accept", "option_rank": 99}, headers=off)
    assert r.status_code == 400
    r = client.post("/decision", json={"recommendation_id": rid, "action": "accept", "option_rank": 1}, headers=off)
    assert r.get_json()["status"] == "accepted"
    assert client.post("/decision", json={"recommendation_id": rid, "action": "accept", "option_rank": 1},
                       headers=off).status_code == 400  # only one decision per recommendation


def test_return_and_override_rules(client, sample_request) -> None:
    off = _token(client, "officer1")
    rid = client.post("/recommend", json=_body(sample_request), headers=off).get_json()["recommendation_id"]
    r = client.post("/decision", json={"recommendation_id": rid, "action": "return", "reason": "Need a 2nd quote"},
                    headers=off)
    assert r.get_json()["status"] == "returned"


def test_audit_trail_complete_and_anonymous(client, rec) -> None:
    rid = rec["recommendation_id"]
    entries = client.get(f"/audit?recommendation_id={rid}", headers=_token(client, "admin1")).get_json()
    kinds = {e["event_type"] for e in entries}
    assert {"recommendation", "explanation", "decision", "view"} <= kinds
    n_rules = len(rec["flags"]) + sum(len(e["rule_ids"]) for e in rec["excluded"])
    assert sum(e["event_type"] == "rule_fired" for e in entries) == n_rules
    text = str(entries)
    assert "officer1" not in text and "display_name" not in text  # no usernames / business names in the log
