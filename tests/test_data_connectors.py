"""Tests for quantlite.data connectors."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantlite.data import (
    DataSource,
    fetch,
    list_sources,
    register_source,
)
from quantlite.data.base import DataMetadata, attach_metadata, standardise_dataframe
from quantlite.data.cache import _cache_key, cache_get, cache_put, clear_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(rows: int = 50) -> pd.DataFrame:
    """Create a toy OHLCV DataFrame."""
    idx = pd.bdate_range("2023-01-02", periods=rows, tz="UTC")
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(rows).cumsum()
    return pd.DataFrame(
        {
            "Open": close + rng.uniform(-1, 1, rows),
            "High": close + rng.uniform(0, 2, rows),
            "Low": close - rng.uniform(0, 2, rows),
            "Close": close,
            "Volume": rng.integers(1_000, 100_000, rows),
        },
        index=idx,
    )


def _mock_yfinance() -> MagicMock:
    """Create a mock yfinance module."""
    mock_yf = MagicMock()
    raw = _make_ohlcv()
    ticker = MagicMock()
    ticker.history.return_value = raw
    mock_yf.Ticker.return_value = ticker
    return mock_yf


# ---------------------------------------------------------------------------
# Base utilities
# ---------------------------------------------------------------------------


class TestStandardiseDataframe:
    def test_lowercases_columns(self) -> None:
        df = _make_ohlcv()
        result = standardise_dataframe(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_sorts_by_index(self) -> None:
        df = _make_ohlcv().iloc[::-1]
        result = standardise_dataframe(df)
        assert result.index.is_monotonic_increasing

    def test_localises_naive_index(self) -> None:
        df = _make_ohlcv()
        df.index = df.index.tz_localize(None)
        result = standardise_dataframe(df)
        assert result.index.tz is not None

    def test_drops_all_nan_rows(self) -> None:
        df = _make_ohlcv()
        df.iloc[5] = np.nan
        result = standardise_dataframe(df)
        assert len(result) == len(df) - 1


class TestMetadata:
    def test_attach_and_read(self) -> None:
        df = _make_ohlcv()
        meta = DataMetadata(source="test", symbol="X", frequency="1d")
        attach_metadata(df, meta)
        assert df.attrs["metadata"].source == "test"
        assert df.attrs["metadata"].symbol == "X"


# ---------------------------------------------------------------------------
# Yahoo (mocked)
# ---------------------------------------------------------------------------


class TestYahooSource:
    def test_fetch_returns_standardised_df(self) -> None:
        mock_yf = _mock_yfinance()
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            df = fetch("AAPL", source="yahoo", cache=False)
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_empty_raises(self) -> None:
        mock_yf = MagicMock()
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = ticker
        with patch.dict(sys.modules, {"yfinance": mock_yf}), \
             pytest.raises(ValueError, match="No data"):
            fetch("BADTICKER", source="yahoo", cache=False)


# ---------------------------------------------------------------------------
# CCXT (mocked)
# ---------------------------------------------------------------------------


class TestCCXTSource:
    def test_fetch_candles(self) -> None:
        ts = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
        candles = [
            [ts + i * 86_400_000, 100 + i, 105 + i, 95 + i, 102 + i, 1000]
            for i in range(10)
        ]
        mock_ccxt = MagicMock()
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = candles
        mock_ccxt.binance = MagicMock(return_value=exchange)

        with patch.dict(sys.modules, {"ccxt": mock_ccxt}):
            df = fetch("BTC/USDT", source="ccxt", exchange="binance", cache=False)
            assert "close" in df.columns
            assert len(df) == 10

    def test_empty_raises(self) -> None:
        mock_ccxt = MagicMock()
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = []
        mock_ccxt.binance = MagicMock(return_value=exchange)

        with patch.dict(sys.modules, {"ccxt": mock_ccxt}), \
             pytest.raises(ValueError, match="No data"):
            fetch("BAD/PAIR", source="ccxt", exchange="binance", cache=False)


# ---------------------------------------------------------------------------
# FRED (mocked)
# ---------------------------------------------------------------------------


class TestFREDSource:
    def test_fetch_series(self) -> None:
        idx = pd.date_range("2020-01-01", periods=30, freq="B")
        series = pd.Series(np.linspace(1.5, 2.0, 30), index=idx)
        mock_fredapi = MagicMock()
        fred_instance = MagicMock()
        fred_instance.get_series.return_value = series
        mock_fredapi.Fred.return_value = fred_instance

        with patch.dict(sys.modules, {"fredapi": mock_fredapi}), \
             patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
            df = fetch("DGS10", source="fred", cache=False)
            assert "close" in df.columns
            assert len(df) == 30

    def test_missing_api_key_raises(self) -> None:
        mock_fredapi = MagicMock()
        with patch.dict(sys.modules, {"fredapi": mock_fredapi}), \
             patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="API key"):
            fetch("DGS10", source="fred", cache=False)


# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------


class TestLocalSource:
    def test_csv_loading(self, tmp_path: Path) -> None:
        df = _make_ohlcv()
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path)

        result = fetch(str(csv_path), source="local", cache=False)
        assert "close" in result.columns
        assert len(result) == len(df)

    def test_parquet_loading(self, tmp_path: Path) -> None:
        df = _make_ohlcv()
        pq_path = tmp_path / "test.parquet"
        df.to_parquet(pq_path)

        result = fetch(str(pq_path), source="local", cache=False)
        assert "close" in result.columns

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            fetch("/nonexistent/file.csv", source="local", cache=False)

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "data.xlsx"
        bad.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            fetch(str(bad), source="local", cache=False)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_write_and_read(self, tmp_path: Path) -> None:
        df = _make_ohlcv()
        df = standardise_dataframe(df)
        cache_put(df, "test", "SYM", {}, cache_dir=tmp_path)
        cached = cache_get("test", "SYM", {}, cache_dir=tmp_path)
        assert cached is not None
        assert len(cached) == len(df)

    def test_ttl_expiry(self, tmp_path: Path) -> None:
        df = _make_ohlcv()
        df = standardise_dataframe(df)
        cache_put(df, "test", "SYM", {}, cache_dir=tmp_path)

        cached = cache_get("test", "SYM", {}, cache_dir=tmp_path, ttl=0)
        assert cached is None

    def test_clear(self, tmp_path: Path) -> None:
        df = _make_ohlcv()
        df = standardise_dataframe(df)
        cache_put(df, "test", "A", {}, cache_dir=tmp_path)
        cache_put(df, "test", "B", {}, cache_dir=tmp_path)
        removed = clear_cache(cache_dir=tmp_path)
        assert removed == 2

    def test_cache_key_deterministic(self) -> None:
        k1 = _cache_key("yahoo", "AAPL", {"period": "5y"})
        k2 = _cache_key("yahoo", "AAPL", {"period": "5y"})
        assert k1 == k2

    def test_cache_key_varies_with_params(self) -> None:
        k1 = _cache_key("yahoo", "AAPL", {"period": "5y"})
        k2 = _cache_key("yahoo", "AAPL", {"period": "1y"})
        assert k1 != k2


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    def test_builtin_sources(self) -> None:
        sources = list_sources()
        assert "yahoo" in sources
        assert "ccxt" in sources
        assert "fred" in sources
        assert "local" in sources

    def test_custom_registration(self) -> None:
        @register_source("test_custom")
        class CustomSource(DataSource):
            def fetch(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
                return _make_ohlcv()

            def supported_symbols(self) -> list[str] | None:
                return ["TEST"]

        assert "test_custom" in list_sources()

    def test_non_datasource_raises(self) -> None:
        with pytest.raises(TypeError, match="not a DataSource"):

            @register_source("bad")
            class NotASource:  # type: ignore[type-arg]
                pass


# ---------------------------------------------------------------------------
# Unified fetch API
# ---------------------------------------------------------------------------


class TestFetchAPI:
    def test_list_of_symbols(self) -> None:
        mock_yf = _mock_yfinance()
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = fetch(["AAPL", "MSFT"], source="yahoo", cache=False)
            assert isinstance(result, dict)
            assert "AAPL" in result
            assert "MSFT" in result

    def test_dict_config(self) -> None:
        mock_yf = _mock_yfinance()
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = fetch({"AAPL": "yahoo"}, cache=False)
            assert isinstance(result, dict)
            assert "AAPL" in result

    def test_bad_type_raises(self) -> None:
        with pytest.raises(TypeError):
            fetch(123, cache=False)  # type: ignore[arg-type]

    def test_auto_infer_ccxt(self) -> None:
        from quantlite.data import _infer_source

        assert _infer_source("BTC/USDT") == "ccxt"
        assert _infer_source("data.csv") == "local"
        assert _infer_source("AAPL") == "yahoo"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_unknown_source_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown data source"):
            fetch("AAPL", source="nonexistent", cache=False)

    def test_missing_dependency(self) -> None:
        # Remove yfinance from sys.modules to trigger ImportError
        with patch.dict(sys.modules, {"yfinance": None}), pytest.raises(ImportError):
            fetch("AAPL", source="yahoo", cache=False)
