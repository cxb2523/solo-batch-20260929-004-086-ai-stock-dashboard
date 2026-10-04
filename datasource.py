"""Batch concurrent stock data fetching with atomic on-disk caching."""

from __future__ import annotations

import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field

import pandas as pd

MAX_WORKERS = 4
FETCH_TIMEOUT = 3.0
MAX_RETRIES = 1
MAX_SYMBOLS = 6

CACHE_DIR = os.environ.get("STOCK_CACHE_DIR", os.path.join(tempfile.gettempdir(), "stock_cache"))

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-^]{0,11}$")

_key_locks: dict[str, threading.Lock] = {}
_key_locks_guard = threading.Lock()


def is_valid_symbol(symbol: str) -> bool:
    return bool(_SYMBOL_RE.match(symbol or ""))


def _lock_for(key: str) -> threading.Lock:
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def _cache_path(symbol: str, start: str, end: str, cache_dir: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9.\-^]", "_", symbol)
    return os.path.join(cache_dir, f"{safe}_{start}_{end}.csv")


def _atomic_write(df: pd.DataFrame, path: str) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            df.to_csv(fh)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"no data returned for {symbol}")
    df = data[["Close"]].dropna().copy()
    if df.empty:
        raise ValueError(f"no usable close prices for {symbol}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def fetch_one(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch a single symbol, cached on disk by (symbol, start, end).

    Concurrent callers for the same key share one lock so the source is
    hit at most once; readers never see a partially written file because
    writes go through a temp file plus atomic replace.
    """
    cache_dir = cache_dir or CACHE_DIR
    path = _cache_path(symbol, start, end, cache_dir)
    if not use_cache:
        return _download(symbol, start, end)
    with _lock_for(path):
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            if not df.empty:
                return df
        df = _download(symbol, start, end)
        os.makedirs(cache_dir, exist_ok=True)
        _atomic_write(df, path)
        return df


@dataclass
class FetchResult:
    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def fetch_many(
    symbols: list[str],
    start: str,
    end: str,
    max_workers: int = MAX_WORKERS,
    timeout: float = FETCH_TIMEOUT,
    retries: int = MAX_RETRIES,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> FetchResult:
    """Fetch many symbols concurrently.

    Invalid symbols fail immediately without any network call. Each valid
    symbol gets up to ``timeout`` seconds per attempt and is retried at
    most ``retries`` times after a timeout. Failures never raise; they are
    reported in ``FetchResult.errors``.
    """
    result = FetchResult()
    todo: list[str] = []
    for symbol in symbols:
        if is_valid_symbol(symbol):
            todo.append(symbol)
        else:
            result.errors[symbol] = "invalid symbol"

    if not todo:
        return result

    executor = ThreadPoolExecutor(max_workers=max_workers)

    def attempt(symbol: str) -> pd.DataFrame:
        last_exc: Exception | None = None
        for _ in range(retries + 1):
            future = executor.submit(fetch_one, symbol, start, end, cache_dir, use_cache)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                last_exc = TimeoutError(f"fetch timed out after {timeout}s")
                continue
        raise last_exc  # type: ignore[misc]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(attempt, symbol): symbol for symbol in todo}
        for future, symbol in futures.items():
            try:
                result.data[symbol] = future.result()
            except Exception as exc:  # noqa: BLE001 - per-symbol isolation
                result.errors[symbol] = str(exc) or exc.__class__.__name__

    executor.shutdown(wait=False, cancel_futures=True)
    return result


def align_and_normalize(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """Intersect trading days, then normalize each series to 100 at its first day.

    Returns the normalized frame (one column per symbol) plus a list of
    human-readable notes for symbols that were dropped because their first
    value was 0 or the whole column was NaN.
    """
    notes: list[str] = []
    closes: dict[str, pd.Series] = {}
    for symbol, df in data.items():
        series = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        closes[symbol] = series

    common_index: pd.Index | None = None
    for series in closes.values():
        idx = series.index
        common_index = idx if common_index is None else common_index.intersection(idx)

    normalized = pd.DataFrame()
    if common_index is None or len(common_index) == 0:
        if closes:
            notes.append("no overlapping trading days across symbols; cannot align.")
        return normalized, notes

    common_index = common_index.sort_values()
    for symbol, series in closes.items():
        aligned = series.reindex(common_index)
        if aligned.isna().all():
            notes.append(f"{symbol}: entire column is NaN, dropped.")
            continue
        first = aligned.dropna().iloc[0]
        if first == 0:
            notes.append(f"{symbol}: first value is 0, cannot normalize, dropped.")
            continue
        normalized[symbol] = aligned / first * 100.0

    return normalized, notes