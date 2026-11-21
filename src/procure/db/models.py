"""SQLAlchemy tables: state, lga, supplier, procurement_transaction, logistics_record,
model_output, audit_log (+ app_user for role-based login).

No personal names are stored in audit_log or model_output (NDPA 2023): users appear only as
user_id + role, suppliers only as supplier_id.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base."""


class State(Base):
    __tablename__ = "state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(40), unique=True)
    lgas: Mapped[list["LGA"]] = relationship(back_populates="state")


class LGA(Base):
    __tablename__ = "lga"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("state.id"))
    name: Mapped[str] = mapped_column(String(60))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    state: Mapped[State] = relationship(back_populates="lgas")


class Supplier(Base):
    __tablename__ = "supplier"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)          # SUP-xxxxxxxx
    display_name: Mapped[str] = mapped_column(String(200))                 # business name, access-controlled
    state: Mapped[str | None] = mapped_column(String(40))
    lga: Mapped[str | None] = mapped_column(String(60))
    registered: Mapped[bool] = mapped_column(Boolean, default=False)
    debarred: Mapped[bool] = mapped_column(Boolean, default=False)
    tax_clearance: Mapped[bool] = mapped_column(Boolean, default=False)
    cac: Mapped[bool] = mapped_column(Boolean, default=False)


class ProcurementTransaction(Base):
    __tablename__ = "procurement_transaction"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)          # ocid
    date: Mapped[datetime] = mapped_column(Date)
    buyer_id: Mapped[str] = mapped_column(String(20))
    supplier_id: Mapped[str] = mapped_column(ForeignKey("supplier.id"))
    item: Mapped[str] = mapped_column(String(120))
    item_category: Mapped[str] = mapped_column(String(40))
    quantity: Mapped[float] = mapped_column(Float)
    value_ngn: Mapped[float] = mapped_column(Float)
    unit_cost_real: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(40))
    n_bidders: Mapped[float] = mapped_column(Float)
    delivery_state: Mapped[str] = mapped_column(String(40))
    delivery_lga: Mapped[str] = mapped_column(String(60))


class LogisticsRecord(Base):
    __tablename__ = "logistics_record"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("procurement_transaction.id"))
    delay_days: Mapped[float] = mapped_column(Float)
    quality_score: Mapped[float] = mapped_column(Float)
    short_delivered: Mapped[bool] = mapped_column(Boolean)
    rejected: Mapped[bool] = mapped_column(Boolean)
    failed: Mapped[bool] = mapped_column(Boolean)
    distance_km: Mapped[float] = mapped_column(Float)
    is_synthetic: Mapped[bool] = mapped_column(Boolean)


class ModelOutput(Base):
    __tablename__ = "model_output"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)          # REC-...
    request_id: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[int] = mapped_column(Integer)                       # app_user.id
    model_version: Mapped[str] = mapped_column(String(80))
    runtime_seconds: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="pending")    # pending|accepted|overridden|returned
    request_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    event_type: Mapped[str] = mapped_column(String(30))   # login|recommendation|explanation|rule_fired|decision|view
    recommendation_id: Mapped[str | None] = mapped_column(String(40), index=True)
    user_id: Mapped[int | None] = mapped_column(Integer)
    role: Mapped[str | None] = mapped_column(String(30))
    payload: Mapped[dict] = mapped_column(JSON)


class AppUser(Base):
    __tablename__ = "app_user"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(60), unique=True)        # role account, not a personal name
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(30))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
