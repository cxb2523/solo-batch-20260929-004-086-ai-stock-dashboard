"""Align fetched series on common trading days and normalize to 100."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from datasource import SymbolResult


@dataclass
class Comparison:
    series: dict[str, pd.Series] = field(default_factory=dict)
    common_dates: pd.DatetimeIndex | None = None
    excluded: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_comparison(results: dict[str, SymbolResult]) -> Comparison:
    comp = Comparison()

    closes: dict[str, pd.Series] = {}
    for symbol, result in results.items():
        if not result.ok:
            comp.failed.append((symbol, result.error or "fetch failed"))
            continue
        close = result.data["Close"].dropna()
        if close.empty:
            comp.excluded.append((symbol, "收盘价整列为 NaN，已剔除"))
            continue
        closes[symbol] = close

    if not closes:
        comp.notes.append("没有任一标的取数成功，无法对比。")
        return comp

    common = None
    for close in closes.values():
        idx = close.index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    comp.common_dates = common

    if len(common) == 0:
        comp.notes.append("各标的交易日没有交集，无法对齐对比。")
        return comp

    for symbol, close in closes.items():
        aligned = close.loc[common]
        base = aligned.iloc[0]
        if base == 0 or pd.isna(base):
            comp.excluded.append((symbol, "交集首日收盘价为 0 或缺失，已剔除"))
            continue
        comp.series[symbol] = aligned / base * 100.0

    for symbol, reason in comp.excluded:
        comp.notes.append(f"{symbol}: {reason}")
    if not comp.series:
        comp.notes.append("剔除异常标的后无可对比数据。")
    return comp
