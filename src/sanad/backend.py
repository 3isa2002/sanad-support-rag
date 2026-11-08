"""HTTP client for the marketplace backend API (order / listing status)."""

from __future__ import annotations

import httpx


class BackendError(RuntimeError):
    """The backend could not be reached or returned an unexpected error."""


class NotFound(LookupError):
    pass


class BackendClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 5.0,
        client: httpx.Client | None = None,
    ):
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout, headers=headers)
        if client is not None and api_key:
            self._client.headers.update(headers)

    def _get(self, path: str) -> dict:
        try:
            resp = self._client.get(path)
        except httpx.HTTPError as exc:
            raise BackendError(f"backend unreachable: {exc}") from exc
        if resp.status_code == 404:
            raise NotFound(path)
        if resp.status_code >= 400:
            raise BackendError(f"backend returned HTTP {resp.status_code} for {path}")
        try:
            return resp.json()
        except ValueError as exc:
            raise BackendError("backend returned invalid JSON") from exc

    def get_order(self, order_id: str) -> dict:
        return self._get(f"/orders/{order_id}")

    def get_listing(self, listing_id: str) -> dict:
        return self._get(f"/listings/{listing_id}")

    def ping(self) -> bool:
        try:
            return self._client.get("/health", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False
