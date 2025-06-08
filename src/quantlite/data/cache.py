"""Local disk caching with configurable TTL for fetched data."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

_DEFAULT_CACHE_DIR = Path.home() / ".quantlite" / "cache"

# Default TTLs in seconds
TTL_DAILY: int = 86_400  # 24 hours
TTL_INTRADAY: int = 3_600  # 1 hour


def _cache_key(source: str, symbol: str, params: dict[str, Any]) -> str:
    """Compute a deterministic cache key from fetch parameters.

    Args:
        source: Data source name.
        symbol: Ticker or instrument identifier.
        params: Additional parameters passed to the fetch call.

    Returns:
        A hex digest string suitable for use as a filename.
    """
    blob = json.dumps({"source": source, "symbol": symbol, **params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _is_intraday(params: dict[str, Any]) -> bool:
    """Heuristic: if interval looks intraday, return True."""
    interval = str(params.get("interval", "1d"))
    return any(tag in interval for tag in ("m", "h")) and "mo" not in interval


def cache_get(
    source: str,
    symbol: str,
    params: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    ttl: int | None = None,
) -> pd.DataFrame | None:
    """Retrieve a cached DataFrame if it exists and has not expired.

    Args:
        source: Data source name.
        symbol: Ticker or instrument identifier.
        params: Fetch parameters used for key generation.
        cache_dir: Override the default cache directory.
        ttl: Override the default TTL (seconds).

    Returns:
        The cached DataFrame, or ``None`` if the cache misses.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    key = _cache_key(source, symbol, params)
    path = cache_dir / f"{key}.parquet"
    if not path.exists():
        return None

    if ttl is None:
        ttl = TTL_INTRADAY if _is_intraday(params) else TTL_DAILY

    age = time.time() - path.stat().st_mtime
    if age > ttl:
        return None

    return pd.read_parquet(path)


def cache_put(
    df: pd.DataFrame,
    source: str,
    symbol: str,
    params: dict[str, Any],
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Write a DataFrame to the cache.

    Args:
        df: DataFrame to cache.
        source: Data source name.
        symbol: Ticker or instrument identifier.
        params: Fetch parameters used for key generation.
        cache_dir: Override the default cache directory.

    Returns:
        The path to the written cache file.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(source, symbol, params)
    path = cache_dir / f"{key}.parquet"
    df.to_parquet(path)
    return path


def clear_cache(*, cache_dir: Path | None = None) -> int:
    """Remove all cached files.

    Args:
        cache_dir: Override the default cache directory.

    Returns:
        Number of files removed.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    if not cache_dir.exists():
        return 0
    count = 0
    for f in cache_dir.glob("*.parquet"):
        f.unlink()
        count += 1
    return count
