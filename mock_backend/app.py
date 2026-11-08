"""Mock marketplace backend API exposing order and listing status.

Run with:  python -m mock_backend   (listens on :8001)
If MOCK_BACKEND_API_KEY is set, requests must send it in the X-API-Key header.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException

DATA_DIR = Path(os.environ.get("MOCK_BACKEND_DATA", Path(__file__).resolve().parents[1] / "data" / "mock_backend"))


def _load(name: str) -> dict[str, dict]:
    records = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
    return {r["id"]: r for r in records}


def create_app(api_key: str | None = None) -> FastAPI:
    api_key = api_key if api_key is not None else os.environ.get("MOCK_BACKEND_API_KEY", "")
    orders, listings = _load("orders.json"), _load("listings.json")
    app = FastAPI(title="Souqna mock backend", version="0.1.0")

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/orders/{order_id}", dependencies=[Depends(check_key)])
    def get_order(order_id: str) -> dict:
        order = orders.get(order_id.upper())
        if not order:
            raise HTTPException(status_code=404, detail="order not found")
        return order

    @app.get("/listings/{listing_id}", dependencies=[Depends(check_key)])
    def get_listing(listing_id: str) -> dict:
        listing = listings.get(listing_id.upper())
        if not listing:
            raise HTTPException(status_code=404, detail="listing not found")
        return listing

    return app


app = create_app()
