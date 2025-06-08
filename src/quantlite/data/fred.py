"""FRED (Federal Reserve Economic Data) source."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import DataMetadata, DataSource, attach_metadata
from .registry import register_source


@register_source("fred")
class FREDSource(DataSource):
    """Fetch macroeconomic series from the FRED API via :pypi:`fredapi`.

    Requires the ``fredapi`` optional dependency and a FRED API key
    set via the ``FRED_API_KEY`` environment variable::

        pip install quantlite[fred]
    """

    def fetch(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """Download a FRED series.

        Args:
            symbol: A FRED series id (e.g. ``"DGS10"``).
            **kwargs: Additional options:
                - ``api_key`` (str): FRED API key. Falls back to
                  the ``FRED_API_KEY`` environment variable.
                - ``start`` (str): Start date.
                - ``end`` (str): End date.

        Returns:
            DataFrame with a ``close`` column (the series values)
            and a UTC DatetimeIndex.

        Raises:
            ImportError: If fredapi is not installed.
            ValueError: If the series returns no data.
        """
        try:
            from fredapi import Fred  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "fredapi is required for FRED data. Install with: pip install quantlite[fred]"
            raise ImportError(msg) from exc

        import os

        api_key = kwargs.pop("api_key", None) or os.environ.get("FRED_API_KEY")
        if not api_key:
            msg = "FRED API key required. Set FRED_API_KEY env var or pass api_key="
            raise ValueError(msg)

        fred = Fred(api_key=api_key)
        series = fred.get_series(symbol, **kwargs)

        if series is None or series.empty:
            msg = f"No data returned for FRED series {symbol!r}"
            raise ValueError(msg)

        df = series.to_frame(name="close")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index().dropna(how="all")

        meta = DataMetadata(source="fred", symbol=symbol, frequency="varies")
        attach_metadata(df, meta)
        return df

    def supported_symbols(self) -> list[str] | None:
        """FRED has hundreds of thousands of series.

        Returns:
            ``None`` (unbounded).
        """
        return None
