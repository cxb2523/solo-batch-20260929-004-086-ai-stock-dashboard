"""FastAPI dashboard: single-symbol page plus the /compare batch view."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import datasource
import plotting

app = FastAPI(title="AI Stock Dashboard")
templates = Jinja2Templates(directory="templates")


def _default_range() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=365)
    return start.isoformat(), end.isoformat()


def parse_symbols(raw: str | None) -> tuple[list[str], str | None]:
    """Split a comma-separated symbols param; dedupe, cap at MAX_SYMBOLS."""
    if not raw:
        return [], None
    seen: list[str] = []
    for part in raw.split(","):
        symbol = part.strip()
        if symbol and symbol not in seen:
            seen.append(symbol)
    note = None
    if len(seen) > datasource.MAX_SYMBOLS:
        note = f"max {datasource.MAX_SYMBOLS} symbols allowed; extra codes were truncated."
        seen = seen[: datasource.MAX_SYMBOLS]
    return seen, note


@app.get("/", response_class=HTMLResponse)
def index(request: Request, symbol: str = Query(default="AAPL")) -> HTMLResponse:
    start, end = _default_range()
    chart = None
    error = None
    try:
        df = datasource.fetch_one(symbol, start, end)
        chart = plotting.single_symbol_line_chart(df, symbol)
    except Exception as exc:  # noqa: BLE001 - page must still render
        error = f"{symbol}: {exc}"
    return templates.TemplateResponse(
        request,
        "index.html",
        {"symbol": symbol, "chart": chart, "error": error},
    )


@app.get("/compare", response_class=HTMLResponse)
def compare(
    request: Request,
    symbols: str = Query(default=""),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
) -> HTMLResponse:
    default_start, default_end = _default_range()
    start = start or default_start
    end = end or default_end

    symbol_list, truncate_note = parse_symbols(symbols)
    notes: list[str] = []
    if truncate_note:
        notes.append(truncate_note)
    if not symbol_list:
        notes.append("provide comma-separated symbols, e.g. /compare?symbols=AAA,BBB")

    result = datasource.fetch_many(symbol_list, start, end) if symbol_list else datasource.FetchResult()
    normalized, align_notes = datasource.align_and_normalize(result.data)
    notes.extend(align_notes)

    rows = []
    for symbol in symbol_list:
        if symbol in result.errors:
            rows.append(
                {
                    "symbol": symbol,
                    "ok": False,
                    "first": "-",
                    "last": "-",
                    "change": "-",
                    "message": result.errors[symbol],
                }
            )
            continue
        if symbol not in normalized.columns:
            rows.append(
                {
                    "symbol": symbol,
                    "ok": False,
                    "first": "-",
                    "last": "-",
                    "change": "-",
                    "message": "dropped (first value 0, all-NaN column, or no overlapping dates)",
                }
            )
            continue
        series = normalized[symbol].dropna()
        first_value = float(series.iloc[0])
        last_value = float(series.iloc[-1])
        rows.append(
            {
                "symbol": symbol,
                "ok": True,
                "first": f"{first_value:.2f}",
                "last": f"{last_value:.2f}",
                "change": f"{last_value - first_value:+.2f}%",
                "message": "",
            }
        )

    chart = None
    if not normalized.empty:
        chart = plotting.normalized_line_chart(normalized)

    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "symbols_param": symbols,
            "start": start,
            "end": end,
            "notes": notes,
            "rows": rows,
            "chart": chart,
        },
    )