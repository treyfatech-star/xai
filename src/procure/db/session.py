"""Engine/session helpers and seeding of reference data and demo role accounts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.security import generate_password_hash

from procure.config import ROOT, load_settings
from procure.data import geo
from procure.db.models import LGA, AppUser, Base, LogisticsRecord, ProcurementTransaction, State, Supplier


def db_url() -> str:
    """Database URL from env PROCURE_DB_URL or config; relative sqlite paths resolve to the repo root."""
    url = os.environ.get("PROCURE_DB_URL", load_settings()["paths"]["db_url"])
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and url != "sqlite:///:memory:":
        p = ROOT / url[len("sqlite:///"):]
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{p}"
    return url


def make_engine(url: str | None = None) -> Engine:
    """Create the engine and all tables."""
    eng = create_engine(url or db_url(), future=True)
    Base.metadata.create_all(eng)
    return eng


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    """Session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    s = factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def seed_users(session: Session, password: str | None = None) -> None:
    """Create the demo role accounts from config if they do not exist."""
    api = load_settings()["api"]
    pw = password or os.environ.get(api["demo_password_env"], "change-me")
    for u in api["demo_users"]:
        if session.scalar(select(AppUser).where(AppUser.username == u["username"])) is None:
            session.add(AppUser(username=u["username"], password_hash=generate_password_hash(pw), role=u["role"]))


def seed_reference(session: Session, transactions: pd.DataFrame, registry: pd.DataFrame) -> None:
    """Load states, LGAs, suppliers, transactions and logistics records (idempotent)."""
    if session.scalar(select(func.count()).select_from(State)):
        return
    for st, grp in geo.lga_table().groupby("state"):
        s = State(name=st)
        s.lgas = [LGA(name=r.lga, lat=r.lat, lon=r.lon) for r in grp.itertuples()]
        session.add(s)
    session.add_all([Supplier(id=r.supplier_id, display_name=r.display_name, state=r.supplier_state, lga=r.supplier_lga,
                              registered=bool(r.registered), debarred=bool(r.debarred),
                              tax_clearance=bool(r.tax_clearance), cac=bool(r.cac)) for r in registry.itertuples()])
    tx = transactions.drop_duplicates("ocid")
    session.add_all([ProcurementTransaction(
        id=r.ocid, date=r.date.date(), buyer_id=r.buyer_id, supplier_id=r.supplier_id, item=r.item,
        item_category=r.item_category, quantity=float(r.quantity), value_ngn=float(r.value),
        unit_cost_real=float(r.unit_cost_real), method=r.method, n_bidders=float(r.n_bidders),
        delivery_state=r.delivery_state, delivery_lga=r.delivery_lga) for r in tx.itertuples()])
    session.add_all([LogisticsRecord(
        transaction_id=r.ocid, delay_days=float(r.delay_days), quality_score=float(r.quality_score),
        short_delivered=bool(r.short_delivered), rejected=bool(r.rejected), failed=bool(r.failed),
        distance_km=float(r.distance_km), is_synthetic=bool(r.is_synthetic)) for r in tx.itertuples()])
