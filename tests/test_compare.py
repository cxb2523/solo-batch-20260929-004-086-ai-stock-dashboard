"""Tests for the /compare endpoint and the alignment/normalization logic."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import datasource
from app import app
from compare import build_comparison


def make_df(dates, closes) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """TestClient with the network seam and cache dir isolated."""
    monkeypatch.setattr(datasource, "CACHE_DIR", tmp_path / "cache")
    return TestClient(app)


def patch_download(monkeypatch, mapping: dict[str, pd.DataFrame | Exception]):
    def fake_download(symbol, start, end):
        value = mapping[symbol]
        if isinstance(value, Exception):
            raise value
        return value.copy()

    monkeypatch.setattr(datasource, "_download", fake_download)


def test_compare_normal(client, monkeypatch):
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    patch_download(
        monkeypatch,
        {
            "AAA": make_df(dates, [10.0, 11.0, 12.0, 13.0]),
            "BBB": make_df(dates, [50.0, 50.0, 55.0, 60.0]),
        },
    )
    resp = client.get("/compare?symbols=AAA,BBB")
    assert resp.status_code == 200
    html = resp.text
    assert "AAA" in html and "BBB" in html
    assert html.count("成功") >= 2
    assert "失败" not in html
    # Normalized: each series starts at 100 on its own first common day.
    assert "100.00" in html
    assert "130.00" in html  # AAA: 13/10*100
    assert "120.00" in html  # BBB: 60/50*100


def test_compare_partial_failure(client, monkeypatch):
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    patch_download(
        monkeypatch,
        {
            "AAA": make_df(dates, [10.0, 11.0, 12.0]),
            "BAD": datasource.FetchError("no data for BAD"),
        },
    )
    resp = client.get("/compare?symbols=AAA,BAD")
    assert resp.status_code == 200
    html = resp.text
    assert "AAA" in html and "成功" in html
    assert "BAD" in html and "失败" in html
    assert "no data for BAD" in html


def test_compare_no_overlapping_dates(client, monkeypatch):
    patch_download(
        monkeypatch,
        {
            "AAA": make_df(pd.date_range("2026-01-05", periods=3, freq="B"), [1.0, 2.0, 3.0]),
            "BBB": make_df(pd.date_range("2026-02-02", periods=3, freq="B"), [4.0, 5.0, 6.0]),
        },
    )
    resp = client.get("/compare?symbols=AAA,BBB")
    assert resp.status_code == 200
    assert "交易日没有交集" in resp.text


def test_compare_too_many_symbols_truncated(client, monkeypatch):
    dates = pd.date_range("2026-01-05", periods=2, freq="B")
    patch_download(
        monkeypatch,
        {f"S{i}": make_df(dates, [1.0, 2.0]) for i in range(7)},
    )
    resp = client.get("/compare?symbols=" + ",".join(f"S{i}" for i in range(7)))
    assert resp.status_code == 200
    assert "最多同时对比 6 只" in resp.text
    assert "S5" in resp.text and "S6" not in resp.text


def test_invalid_symbol_fails_immediately(client, monkeypatch):
    dates = pd.date_range("2026-01-05", periods=2, freq="B")
    patch_download(monkeypatch, {"AAA": make_df(dates, [1.0, 2.0])})
    resp = client.get("/compare?symbols=AAA,%2E%2E%2Fetc")
    assert resp.status_code == 200
    assert "invalid symbol" in resp.text


def test_alignment_excludes_zero_and_nan():
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    results = {
        "ZERO": datasource.SymbolResult("ZERO", data=make_df(dates, [0.0, 1.0, 2.0])),
        "NAN": datasource.SymbolResult("NAN", data=make_df(dates, [float("nan")] * 3)),
        "GOOD": datasource.SymbolResult("GOOD", data=make_df(dates, [5.0, 10.0, 15.0])),
    }
    comp = build_comparison(results)
    assert set(comp.series) == {"GOOD"}
    assert {sym for sym, _ in comp.excluded} == {"ZERO", "NAN"}
    assert comp.series["GOOD"].iloc[0] == pytest.approx(100.0)
    assert comp.series["GOOD"].iloc[-1] == pytest.approx(300.0)
    assert comp.notes  # exclusion notes surfaced for the page header


def test_alignment_uses_date_intersection():
    dates_a = pd.date_range("2026-01-05", periods=4, freq="B")
    dates_b = dates_a[1:]
    results = {
        "AAA": datasource.SymbolResult("AAA", data=make_df(dates_a, [10.0, 20.0, 30.0, 40.0])),
        "BBB": datasource.SymbolResult("BBB", data=make_df(dates_b, [5.0, 10.0, 20.0])),
    }
    comp = build_comparison(results)
    assert len(comp.common_dates) == 3
    # AAA normalized on its first common day (20.0), not its own first row.
    assert comp.series["AAA"].iloc[0] == pytest.approx(100.0)
    assert comp.series["AAA"].iloc[-1] == pytest.approx(200.0)
    assert comp.series["BBB"].iloc[-1] == pytest.approx(400.0)
