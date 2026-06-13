"""Local file data source (CSV and Parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import DataMetadata, DataSource, attach_metadata, standardise_dataframe
from .registry import register_source


@register_source("local")
class LocalSource(DataSource):
    """Load OHLCV data from local CSV or Parquet files.

    The file must contain columns mappable to ``open``, ``high``,
    ``low``, ``close``, ``volume`` (case-insensitive). A date or
    datetime column (or the index) is used as the DatetimeIndex.
    """

    def fetch(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """Read a local CSV or Parquet file.

        Args:
            symbol: Path to the file (CSV or Parquet).
            **kwargs: Additional options:
                - ``date_column`` (str): Column name to parse as dates.
                  If omitted, the first column or index is used.

        Returns:
            Standardised OHLCV DataFrame.

        Raises:
            FileNotFoundError: If *symbol* does not point to a file.
            ValueError: If the file format is unsupported or empty.
        """
        path = Path(symbol)
        if not path.exists():
            msg = f"File not found: {symbol}"
            raise FileNotFoundError(msg)

        suffix = path.suffix.lower()
        date_column = kwargs.pop("date_column", None)

        if suffix == ".csv":
            if date_column:
                df = pd.read_csv(path, parse_dates=[date_column], index_col=date_column)
            else:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
            if date_column and date_column in df.columns:
                df = df.set_index(date_column)
        else:
            msg = f"Unsupported file format {suffix!r}. Use .csv or .parquet."
            raise ValueError(msg)

        if df.empty:
            msg = f"File {symbol} is empty"
            raise ValueError(msg)

        df = standardise_dataframe(df)
        meta = DataMetadata(source="local", symbol=symbol, frequency="unknown")
        attach_metadata(df, meta)
        return df

    def supported_symbols(self) -> list[str] | None:
        """Local files are user-provided.

        Returns:
            ``None`` (any file path is valid).
        """
        return None
