from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import Lock
from typing import Any


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    value: str


class DeterministicCache:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = Lock()

    def get(self, key: CacheKey) -> Any | None:
        with self._lock:
            return self._data.get(key.value)

    def set(self, key: CacheKey, value: Any) -> None:
        with self._lock:
            self._data[key.value] = value
