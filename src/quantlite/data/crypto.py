"""Cryptocurrency data source (CCXT wrapper)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import DataMetadata, DataSource, attach_metadata, standardise_dataframe
from .registry import register_source


@register_source("ccxt")
class CCXTSource(DataSource):
    """Fetch OHLCV data from cryptocurrency exchanges via :pypi:`ccxt`.

    Requires the ``ccxt`` optional dependency::

        pip install quantlite[crypto]
    """

    def fetch(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """Download OHLCV candles for *symbol* from a crypto exchange.

        Args:
            symbol: A CCXT trading pair (e.g. ``"BTC/USDT"``).
            **kwargs: Additional options:
                - ``exchange`` (str): Exchange id, default ``"binance"``.
                - ``timeframe`` (str): Candle interval, default ``"1d"``.
                - ``limit`` (int): Number of candles, default ``1000``.

        Returns:
            Standardised OHLCV DataFrame.

        Raises:
            ImportError: If ccxt is not installed.
            ValueError: If the download returns no data.
        """
        try:
            import ccxt  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "ccxt is required for crypto data. Install with: pip install quantlite[crypto]"
            raise ImportError(msg) from exc

        exchange_id = kwargs.pop("exchange", "binance")
        timeframe = kwargs.pop("timeframe", "1d")
        limit = kwargs.pop("limit", 1000)

        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            msg = f"Unknown exchange {exchange_id!r}"
            raise ValueError(msg)

        exchange = exchange_cls()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        if not ohlcv:
            msg = f"No data returned for {symbol!r} on {exchange_id}"
            raise ValueError(msg)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = standardise_dataframe(df)

        meta = DataMetadata(
            source="ccxt",
            symbol=symbol,
            frequency=timeframe,
            extra={"exchange": exchange_id},
        )
        attach_metadata(df, meta)
        return df

    def supported_symbols(self) -> list[str] | None:
        """CCXT supports thousands of pairs across many exchanges.

        Returns:
            ``None`` (unbounded).
        """
        return None
