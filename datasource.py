"""Concurrent batch price fetching with a fine-grained, atomic disk cache."""

from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "prices"
MAX_WORKERS = 4
FETCH_TIMEOUT = 3.0
MAX_ATTEMPTS = 2  # initial attempt + exactly one retry on timeout

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-^=]{0,11}$")


class FetchError(Exception):
    """A symbol could not be fetched."""


class InvalidSymbolError(FetchError):
    """The symbol itself is malformed; fail immediately without retry."""


@dataclass
class SymbolResult:
    symbol: str
    data: pd.DataFrame | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def validate_symbol(symbol: str) -> str:
    if not _SYMBOL_RE.match(symbol):
        raise InvalidSymbolError(f"invalid symbol: {symbol!r}")
    return symbol


def _key_lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _cache_path(symbol: str, start: str, end: str) -> Path:
    key = f"{symbol}_{start}_{end}"
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", key)
    return CACHE_DIR / f"{safe}.csv"


def _read_cache(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _write_cache_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    df.to_csv(tmp)
    os.replace(tmp, path)


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_cached(symbol: str, start: str, end: str) -> pd.DataFrame:
    path = _cache_path(symbol, start, end)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    # Per-key lock: concurrent requests for the same key queue up instead of
    # stampeding the upstream source (no cache penetration).
    with _key_lock(path.name):
        cached = _read_cache(path)
        if cached is not None:
            return cached
        df = _download(symbol, start, end)
        if df is None or df.empty:
            raise FetchError(f"no data for {symbol}")
        _write_cache_atomic(path, df)
        return df


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a single symbol, using the disk cache. Raises FetchError."""
    validate_symbol(symbol)
    return _fetch_cached(symbol, start, end)


def fetch_many(
    symbols: list[str],
    start: str,
    end: str,
    max_workers: int = MAX_WORKERS,
    timeout: float = FETCH_TIMEOUT,
) -> dict[str, SymbolResult]:
    """Batch fetch entry point.

    Runs symbols concurrently on a fixed-size thread pool. Each symbol gets
    ``timeout`` seconds per attempt, with exactly one retry after a timeout.
    Malformed symbols fail immediately without consuming a worker. A failure
    for one symbol never affects the others.
    """
    results: dict[str, SymbolResult] = {}
    pending: list[str] = []
    for symbol in symbols:
        try:
            validate_symbol(symbol)
        except InvalidSymbolError as exc:
            results[symbol] = SymbolResult(symbol=symbol, error=str(exc))
        else:
            pending.append(symbol)

    def attempt(sym: str) -> pd.DataFrame:
        return _fetch_cached(sym, start, end)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {sym: pool.submit(attempt, sym) for sym in pending}
        for sym, future in futures.items():
            try:
                results[sym] = SymbolResult(symbol=sym, data=future.result(timeout=timeout))
                continue
            except FutureTimeoutError:
                pass  # fall through to the single retry below
            except Exception as exc:
                results[sym] = SymbolResult(symbol=sym, error=str(exc))
                continue
            try:
                retry = pool.submit(attempt, sym)
                results[sym] = SymbolResult(symbol=sym, data=retry.result(timeout=timeout))
            except FutureTimeoutError:
                results[sym] = SymbolResult(symbol=sym, error=f"timeout after {MAX_ATTEMPTS} attempts")
            except Exception as exc:
                results[sym] = SymbolResult(symbol=sym, error=str(exc))
    return results
