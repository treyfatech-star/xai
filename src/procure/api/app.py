"""Flask REST API.

POST /login                    {username, password} -> {token, role}
POST /recommend                request fields -> RecommendationResult JSON
GET  /recommendation/<id>      stored recommendation + status
POST /decision                 {recommendation_id, action: accept|override|return, option_rank?, reason?}
GET  /audit                    ?recommendation_id=&event_type=&limit=   (administrator)
GET  /health
"""

from __future__ import annotations

from typing import Any

from flask import Flask, g, jsonify, request

from procure.api.auth import issue_token, require, secret_key
from procure.api.schemas import ValidationError, parse_recommend
from procure.config import load_settings
from procure.db.session import make_engine, make_sessionmaker, seed_users
from procure.pipeline import Engine
from procure import service


def create_app(db_url: str | None = None, engine: Engine | None = None, seed_password: str | None = None) -> Flask:
    """App factory. `engine` (data + models) is loaded lazily from disk when not given."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = secret_key()
    db = make_engine(db_url)
    factory = make_sessionmaker(db)
    with factory() as s:
        seed_users(s, seed_password)
        s.commit()
    state: dict[str, Any] = {"engine": engine}

    def get_engine() -> Engine:
        if state["engine"] is None:
            state["engine"] = Engine.from_disk()
        return state["engine"]

    @app.before_request
    def open_session() -> None:
        g.db = factory()

    @app.teardown_request
    def close_session(exc: BaseException | None) -> None:
        db_s = g.pop("db", None)
        if db_s is not None:
            if exc is None:
                db_s.commit()
            else:
                db_s.rollback()
            db_s.close()

    @app.errorhandler(service.ServiceError)
    def service_error(e: service.ServiceError) -> Any:
        return jsonify(error=str(e)), 400

    @app.errorhandler(ValidationError)
    def validation_error(e: ValidationError) -> Any:
        return jsonify(error="Invalid request", details=e.errors), 400

    @app.get("/health")
    def health() -> Any:
        return jsonify(status="ok")

    @app.post("/login")
    def login() -> Any:
        body = request.get_json(silent=True) or {}
        user = service.authenticate(g.db, str(body.get("username", "")), str(body.get("password", "")))
        if user is None:
            return jsonify(error="Wrong username or password."), 401
        return jsonify(token=issue_token(user), role=user.role)

    @app.post("/recommend")
    @require("recommend")
    def recommend() -> Any:
        req = parse_recommend(request.get_json(silent=True))
        res = service.create_recommendation(g.db, g.user, req, get_engine())
        return jsonify(res.to_dict()), 201

    @app.get("/recommendation/<rec_id>")
    @require("view_recommendation")
    def get_recommendation(rec_id: str) -> Any:
        return jsonify(service.get_recommendation(g.db, g.user, rec_id))

    @app.post("/decision")
    @require("decide")
    def decision() -> Any:
        b = request.get_json(silent=True) or {}
        out = service.record_decision(g.db, g.user, str(b.get("recommendation_id", "")), str(b.get("action", "")),
                                      b.get("option_rank"), b.get("reason"), b.get("chosen_supplier_ids"))
        return jsonify(out)

    @app.get("/audit")
    @require("view_audit")
    def audit() -> Any:
        a = request.args
        return jsonify(service.list_audit(g.db, g.user, a.get("recommendation_id"), a.get("event_type"),
                                          int(a.get("limit", 200))))

    return app


def main() -> None:
    """make api: seed reference tables if empty, then serve."""
    from procure.db.session import seed_reference

    cfg = load_settings()["api"]
    eng = Engine.from_disk()
    factory = make_sessionmaker(make_engine())
    with factory() as s:
        seed_reference(s, eng.store.transactions, eng.store.registry())
        s.commit()
    create_app(engine=eng).run(host=cfg["host"], port=cfg["port"])


if __name__ == "__main__":
    main()
