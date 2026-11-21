"""Token authentication and role-based access for the Flask API."""

from __future__ import annotations

import os
import secrets
from functools import wraps
from typing import Any, Callable

from flask import current_app, g, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from procure.config import load_settings
from procure.db.models import AppUser
from procure.service import permissions


def secret_key() -> str:
    """Signing key from the environment; a random per-process key otherwise (dev only)."""
    return os.environ.get(load_settings()["api"]["secret_key_env"]) or secrets.token_hex(32)


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="procure-auth")


def issue_token(user: AppUser) -> str:
    """Signed, time-limited token carrying user id and role."""
    return _serializer().dumps({"uid": user.id, "role": user.role})


def require(permission: str) -> Callable:
    """Decorator: valid Bearer token whose role grants `permission`."""

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return jsonify(error="Please log in (missing Bearer token)."), 401
            try:
                data = _serializer().loads(header[7:], max_age=load_settings()["api"]["token_max_age_seconds"])
            except SignatureExpired:
                return jsonify(error="Your session has expired; please log in again."), 401
            except BadSignature:
                return jsonify(error="Invalid token."), 401
            user = g.db.get(AppUser, data["uid"])
            if user is None or not user.active:
                return jsonify(error="Account disabled."), 401
            if permission not in permissions(user.role):
                return jsonify(error=f"Your role ({user.role}) may not {permission.replace('_', ' ')}."), 403
            g.user = user
            return fn(*args, **kwargs)

        return wrapper

    return deco
