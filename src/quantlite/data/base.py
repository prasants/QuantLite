"""Base classes and types for data connectors."""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataMetadata:
    """Metadata attached to a fetched DataFrame.

    Attributes:
        source: Name of the data source (e.g. ``"yahoo"``, ``"ccxt"``).
        symbol: The ticker or symbol fetched.
        frequency: Data frequency string (e.g. ``"1d"``, ``"1h"``).
        fetch_time: UTC timestamp of when the data was retrieved.
        extra: Any additional source-specific metadata.
    """

    source: str
    symbol: str
    frequency: str
    fetch_time: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)


def attach_metadata(df: pd.DataFrame, metadata: DataMetadata) -> pd.DataFrame:
    """Attach a :class:`DataMetadata` instance to a DataFrame.

    Args:
        df: The DataFrame to annotate.
        metadata: Metadata to attach.

    Returns:
        The same DataFrame, with a ``.metadata`` attribute set.
    """
    df.attrs["metadata"] = metadata
    return df


def standardise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, sort by index, and drop NaN rows.

    Ensures the DataFrame has lowercase OHLCV columns, a sorted
    :class:`~pandas.DatetimeIndex`, and no fully-NaN rows.

    Args:
        df: Raw DataFrame from a data source.

    Returns:
        Cleaned DataFrame.
    """
    df.columns = [c.lower().strip() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    df = df.dropna(how="all")
    return df


class DataSource(ABC):
    """Abstract base class for all data sources.

    Subclasses must implement :meth:`fetch` and :meth:`supported_symbols`.
    """

    @abstractmethod
    def fetch(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """Fetch OHLCV data for *symbol*.

        Args:
            symbol: Ticker or instrument identifier.
            **kwargs: Source-specific parameters (period, interval, etc.).

        Returns:
            A pandas DataFrame with a DatetimeIndex and lowercase
            ``open``, ``high``, ``low``, ``close``, ``volume`` columns.

        Raises:
            ValueError: If the symbol is invalid or data is empty.
            ImportError: If a required optional dependency is missing.
        """
        ...

    @abstractmethod
    def supported_symbols(self) -> list[str] | None:
        """Return a list of supported symbols, or ``None`` if unbounded.

        Returns:
            A list of symbol strings, or ``None``.
        """
        ...
