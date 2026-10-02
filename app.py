"""FastAPI dashboard: single-stock page, JSON API, and multi-symbol compare."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import compare as compare_mod
import datasource
import plotting

BASE_DIR = Path(__file__).resolve().parent
MAX_COMPARE_SYMBOLS = 6

app = FastAPI(title="AI Stock Dashboard API")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _default_range() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=365)
    return start.isoformat(), end.isoformat()


def parse_symbols(raw: str | None) -> tuple[list[str], str | None]:
    """Split a comma-separated query param; dedupe, cap at MAX_COMPARE_SYMBOLS."""
    if not raw:
        return [], None
    seen: list[str] = []
    for token in raw.split(","):
        symbol = token.strip().upper()
        if symbol and symbol not in seen:
            seen.append(symbol)
    note = None
    if len(seen) > MAX_COMPARE_SYMBOLS:
        seen = seen[:MAX_COMPARE_SYMBOLS]
        note = f"最多同时对比 {MAX_COMPARE_SYMBOLS} 只，已截取前 {MAX_COMPARE_SYMBOLS} 只。"
    return seen, note


@app.get("/", response_class=HTMLResponse)
def single_stock_page(request: Request, symbol: str = Query(default="AAPL")):
    start, end = _default_range()
    chart = None
    error = None
    try:
        df = datasource.fetch_one(symbol, start, end)
        chart = plotting.make_line_chart(
            {symbol: df["Close"]}, title=f"{symbol} 收盘价", y_title="价格"
        )
    except datasource.FetchError as exc:
        error = str(exc)
    return templates.TemplateResponse(
        request,
        "stock.html",
        {"symbol": symbol, "chart": chart, "error": error},
    )


@app.get("/api/stock/{symbol}")
def stock_json(symbol: str):
    start, end = _default_range()
    try:
        df = datasource.fetch_one(symbol, start, end)
    except datasource.FetchError as exc:
        return JSONResponse(status_code=404, content={"symbol": symbol, "error": str(exc)})
    close = df["Close"].dropna()
    return {
        "symbol": symbol,
        "dates": [d.strftime("%Y-%m-%d") for d in close.index],
        "closes": [round(float(v), 4) for v in close.values],
    }


@app.get("/compare", response_class=HTMLResponse)
def compare_page(
    request: Request,
    symbols: str = Query(default=""),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
):
    default_start, default_end = _default_range()
    start = start or default_start
    end = end or default_end

    symbol_list, note = parse_symbols(symbols)
    notes: list[str] = [note] if note else []
    rows: list[dict] = []
    chart = None

    if not symbol_list:
        notes.append("请在 URL 中提供 symbols 参数，例如 /compare?symbols=AAPL,MSFT")
    else:
        results = datasource.fetch_many(symbol_list, start, end)
        comparison = compare_mod.build_comparison(results)
        notes.extend(comparison.notes)

        for symbol in symbol_list:
            result = results[symbol]
            if not result.ok:
                rows.append({"symbol": symbol, "status": "失败", "detail": result.error})
                continue
            series = comparison.series.get(symbol)
            if series is None:
                reason = next(
                    (r for s, r in comparison.excluded if s == symbol), "已剔除"
                )
                rows.append({"symbol": symbol, "status": "剔除", "detail": reason})
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "status": "成功",
                    "detail": "",
                    "first": f"{series.iloc[0]:.2f}",
                    "last": f"{series.iloc[-1]:.2f}",
                    "points": len(series),
                }
            )

        if comparison.series:
            chart = plotting.make_line_chart(
                comparison.series,
                title="归一化走势对比（各自首日 = 100）",
                y_title="归一化净值",
            )

    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "symbols": ",".join(symbol_list),
            "start": start,
            "end": end,
            "notes": notes,
            "rows": rows,
            "chart": chart,
        },
    )
