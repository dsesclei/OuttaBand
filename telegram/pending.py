from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass

from shared_types import PendingKind, PendingPayload


@dataclass
class Pending:
    token: str
    kind: PendingKind
    payload: PendingPayload
    created_at: float


class PendingStore:
    def __init__(self, cap: int = 100, ttl_s: int = 3600) -> None:
        # Maximum number of pending entries and time-to-live in seconds
        self._cap = max(1, int(cap))
        self._ttl = max(1, int(ttl_s))
        # OrderedDict for evicting oldest entries in O(1)
        self._by_token: OrderedDict[str, Pending] = OrderedDict()

    @staticmethod
    def new_token() -> str:
        return secrets.token_hex(12)

    def put(self, kind: PendingKind, payload: PendingPayload) -> str:
        # Insert a new pending entry, evicting old ones as needed
        self._evict()
        token = self.new_token()
        self._by_token[token] = Pending(token, kind, payload, time.time())
        return token

    def pop(self, kind: PendingKind, token: str) -> Pending | None:
        # Remove and return the pending entry if it matches the given kind
        self._evict()
        p = self._by_token.pop(token, None)
        if p is None or p.kind != kind:
            return None
        return p

    def _evict(self) -> None:
        # Remove expired entries (based on TTL) and keep total count under the cap
        now = time.time()

        # Remove items from the front while they are expired
        while self._by_token:
            first = next(iter(self._by_token.values()))
            if now - first.created_at <= self._ttl:
                break
            self._by_token.popitem(last=False)

        # Enforce capacity by popping oldest items
        while len(self._by_token) > self._cap:
            self._by_token.popitem(last=False)
