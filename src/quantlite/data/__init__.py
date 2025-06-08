"""Unified data connectors for QuantLite.

Provides a single :func:`fetch` entry point that routes to the
appropriate data source (Yahoo Finance, CCXT, FRED, or local files).

Example::

    from quantlite.data import fetch

    df = fetch("AAPL", period="5y")
    df = fetch("BTC/USDT", source="ccxt", exchange="binance")
    df = fetch("DGS10", source="fred", api_key="...")
    df = fetch("prices.csv", source="local")
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import crypto as _crypto  # noqa: F401
from . import fred as _fred  # noqa: F401
from . import local as _local  # noqa: F401

# Trigger registration of built-in sources
from . import yahoo as _yahoo  # noqa: F401
from .base import DataMetadata, DataSource, attach_metadata, standardise_dataframe
from .cache import cache_get, cache_put, clear_cache
from .registry import get_source, list_sources, register_source


def _infer_source(symbol: str) -> str:
    """Guess the best source for *symbol* based on simple heuristics.

    Args:
        symbol: A ticker, pair, series id, or file path.

    Returns:
        A source name string.
    """
    if "/" in symbol and not symbol.startswith((".", "/")):
        return "ccxt"
    if symbol.endswith((".csv", ".parquet", ".pq")):
        return "local"
    return "yahoo"


def _fetch_single(
    symbol: str,
    *,
    source: str | None = None,
    cache: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch data for a single symbol.

    Args:
        symbol: Ticker, pair, series id, or file path.
        source: Explicit source name. Auto-detected if ``None``.
        cache: Whether to use disk caching.
        **kwargs: Forwarded to the data source's ``fetch`` method.

    Returns:
        Standardised OHLCV DataFrame with metadata.
    """
    resolved_source = source or _infer_source(symbol)

    if cache:
        cached = cache_get(resolved_source, symbol, kwargs)
        if cached is not None:
            return cached

    src = get_source(resolved_source)
    df = src.fetch(symbol, **kwargs)

    if cache:
        cache_put(df, resolved_source, symbol, kwargs)

    return df


def fetch(
    symbols: str | list[str] | dict[str, Any],
    *,
    source: str | None = None,
    cache: bool = True,
    **kwargs: Any,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Unified data fetcher.

    Accepts a single symbol, a list of symbols, or a dictionary mapping
    symbols to source configurations.

    Args:
        symbols: What to fetch. Can be:
            - A single symbol string.
            - A list of symbols (all fetched from the same source).
            - A dict mapping symbols to source names or config dicts.
        source: Default data source name. Auto-detected per symbol if
            ``None``.
        cache: Whether to use local disk caching. Pass ``False`` to
            bypass.
        **kwargs: Extra parameters forwarded to each source's
            ``fetch`` method.

    Returns:
        A single DataFrame when *symbols* is a string, or a dict of
        ``{symbol: DataFrame}`` when *symbols* is a list or dict.

    Raises:
        KeyError: If a requested source is not registered.
        ValueError: If a symbol returns no data.
        ImportError: If a required optional dependency is missing.
    """
    # Single symbol
    if isinstance(symbols, str):
        return _fetch_single(symbols, source=source, cache=cache, **kwargs)

    # List of symbols
    if isinstance(symbols, list):
        return {s: _fetch_single(s, source=source, cache=cache, **kwargs) for s in symbols}

    # Dict of {symbol: source_or_config}
    if isinstance(symbols, dict):
        result: dict[str, pd.DataFrame] = {}
        for sym, cfg in symbols.items():
            if isinstance(cfg, str):
                result[sym] = _fetch_single(sym, source=cfg, cache=cache, **kwargs)
            elif isinstance(cfg, dict):
                merged = {**kwargs, **cfg}
                sym_source = merged.pop("source", source)
                result[sym] = _fetch_single(sym, source=sym_source, cache=cache, **merged)
            else:
                msg = f"Invalid config for {sym!r}: expected str or dict, got {type(cfg).__name__}"
                raise TypeError(msg)
        return result

    msg = f"symbols must be str, list, or dict, got {type(symbols).__name__}"
    raise TypeError(msg)


__all__ = [
    "DataMetadata",
    "DataSource",
    "attach_metadata",
    "cache_get",
    "cache_put",
    "clear_cache",
    "fetch",
    "get_source",
    "list_sources",
    "register_source",
    "standardise_dataframe",
]
