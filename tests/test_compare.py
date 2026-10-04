"""Tests for the /compare batch view and datasource.fetch_many."""

from __future__ import annotations

import os
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import datasource

client = TestClient(app_module.app)


def make_df(dates: list[str], closes: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(idx, name="Date"))


DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


@pytest.fixture()
def fake_download(monkeypatch, tmp_path):
    """Route datasource._download to synthetic data and isolate the cache."""
    monkeypatch.setattr(datasource, "CACHE_DIR", str(tmp_path))
    calls = []

    def fake(symbol: str, start: str, end: str) -> pd.DataFrame:
        calls.append(symbol)
        if symbol == "AAA":
            return make_df(DATES, [10.0, 11.0, 12.0, 13.0])
        if symbol == "BBB":
            return make_df(DATES, [20.0, 22.0, 18.0, 26.0])
        if symbol == "CCC":
            return make_df(["2024-02-01", "2024-02-02"], [5.0, 6.0])
        raise ValueError(f"boom: {symbol}")

    monkeypatch.setattr(datasource, "_download", fake)
    return calls


def test_compare_normal(fake_download):
    resp = client.get("/compare?symbols=AAA,BBB")
    assert resp.status_code == 200
    html = resp.text
    assert "AAA" in html and "BBB" in html
    assert "成功" in html
    # normalized base is 100 on the first common day
    assert "100.00" in html
    # BBB ends at 26/20*100 = 130
    assert "130.00" in html
    assert "compare-chart" in html


def test_compare_partial_failure(fake_download):
    resp = client.get("/compare?symbols=AAA,FAIL1")
    assert resp.status_code == 200
    html = resp.text
    assert "FAIL1" in html
    assert "失败" in html
    assert "boom: FAIL1" in html
    # the healthy symbol still renders
    assert "成功" in html
    assert "compare-chart" in html


def test_compare_invalid_symbol_fails_fast(fake_download):
    resp = client.get("/compare?symbols=AAA,!!BAD")
    assert resp.status_code == 200
    assert "invalid symbol" in resp.text
    # invalid symbol never reaches the downloader
    assert "!!BAD" not in fake_download


def test_compare_no_overlapping_dates(fake_download):
    resp = client.get("/compare?symbols=AAA,CCC")
    assert resp.status_code == 200
    assert "no overlapping trading days" in resp.text


def test_compare_too_many_symbols_truncated(fake_download):
    resp = client.get("/compare?symbols=AAA,BBB,D1,D2,D3,D4,D5,D6")
    assert resp.status_code == 200
    assert "max 6 symbols allowed" in resp.text


def test_fetch_many_timeout_retries_once(monkeypatch, tmp_path):
    attempts = []

    def slow_fetch(symbol, start, end, cache_dir=None, use_cache=True):
        attempts.append(symbol)
        time.sleep(0.5)
        return make_df(DATES, [1.0, 2.0, 3.0, 4.0])

    monkeypatch.setattr(datasource, "fetch_one", slow_fetch)
    result = datasource.fetch_many(["AAA"], "2024-01-01", "2024-02-01", timeout=0.1)
    assert "AAA" in result.errors
    assert "timed out" in result.errors["AAA"]
    assert len(attempts) == 2  # exactly one retry after the timeout


def test_fetch_many_workers_capped_at_four():
    assert datasource.MAX_WORKERS == 4
    assert datasource.FETCH_TIMEOUT == 3.0
    assert datasource.MAX_RETRIES == 1


def test_cache_key_and_atomic_write(fake_download, tmp_path):
    df1 = datasource.fetch_one("AAA", "2024-01-01", "2024-02-01")
    df2 = datasource.fetch_one("AAA", "2024-01-01", "2024-02-01")
    assert df1.equals(df2)
    # second call served from cache: downloader hit only once
    assert fake_download.count("AAA") == 1
    # cache key is split by symbol + date range
    expected = os.path.join(str(tmp_path), "AAA_2024-01-01_2024-02-01.csv")
    assert os.path.exists(expected)
    # no temp files left behind, no half-written reads
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    # a different date range is a different cache entry
    datasource.fetch_one("AAA", "2024-01-01", "2024-03-01")
    assert fake_download.count("AAA") == 2


def test_cache_concurrent_single_flight(fake_download, tmp_path):
    barrier = threading.Barrier(8)

    def worker(out):
        barrier.wait()
        out.append(datasource.fetch_one("BBB", "2024-01-01", "2024-02-01"))

    out: list[pd.DataFrame] = []
    threads = [threading.Thread(target=worker, args=(out,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 8
    # concurrent callers did not stampede the source
    assert fake_download.count("BBB") == 1


def test_align_and_normalize_drops_bad_series():
    good = make_df(DATES, [10.0, 11.0, 12.0, 13.0])
    zero_first = make_df(DATES, [0.0, 1.0, 2.0, 3.0])
    all_nan = make_df(DATES, [float("nan")] * 4)
    normalized, notes = datasource.align_and_normalize(
        {"GOOD": good, "ZERO": zero_first, "NAN": all_nan}
    )
    assert list(normalized.columns) == ["GOOD"]
    assert normalized["GOOD"].iloc[0] == pytest.approx(100.0)
    assert any("ZERO" in n for n in notes)
    assert any("NAN" in n for n in notes)


def test_single_symbol_page_still_works(fake_download):
    resp = client.get("/?symbol=AAA")
    assert resp.status_code == 200
    assert "AAA" in resp.text
    assert "single-chart" in resp.text