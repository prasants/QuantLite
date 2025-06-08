"""Yahoo Finance data source (yfinance wrapper)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import DataMetadata, DataSource, attach_metadata, standardise_dataframe
from .registry import register_source


@register_source("yahoo")
class YahooSource(DataSource):
    """Fetch OHLCV data from Yahoo Finance via :pypi:`yfinance`.

    Requires the ``yfinance`` optional dependency::

        pip install quantlite[yahoo]
    """

    def fetch(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """Download historical data for *symbol* from Yahoo Finance.

        Args:
            symbol: A Yahoo Finance ticker (e.g. ``"AAPL"``).
            **kwargs: Forwarded to ``yfinance.Ticker.history``
                (``period``, ``interval``, ``start``, ``end``, etc.).

        Returns:
            Standardised OHLCV DataFrame.

        Raises:
            ImportError: If yfinance is not installed.
            ValueError: If the download returns no data.
        """
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "yfinance is required for Yahoo data. Install with: pip install quantlite[yahoo]"
            raise ImportError(msg) from exc

        period = kwargs.pop("period", "5y")
        interval = kwargs.pop("interval", "1d")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, **kwargs)

        if df.empty:
            msg = f"No data returned for symbol {symbol!r}"
            raise ValueError(msg)

        df = standardise_dataframe(df)
        meta = DataMetadata(source="yahoo", symbol=symbol, frequency=interval)
        attach_metadata(df, meta)
        return df

    def supported_symbols(self) -> list[str] | None:
        """Yahoo Finance supports a very large, dynamic set of symbols.

        Returns:
            ``None`` (unbounded).
        """
        return None
