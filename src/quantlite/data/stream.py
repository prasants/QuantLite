"""Streaming price feeds from crypto exchanges.

Provides real-time price streaming via WebSocket connections (using CCXT pro)
with automatic fallback to REST polling. Supports both async iterator and
callback interfaces, with built-in throttling and reconnection logic.

Example::

    import quantlite as ql

    stream = ql.stream(["BTC-USD", "ETH-USD"], exchange="binance")

    # Async iterator
    async for tick in stream:
        print(tick)

    # Callback
    stream.on_tick(my_handler)
    await stream.start()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
)

__all__ = [
    "PriceTick",
    "PriceStream",
    "create_stream",
]

logger = logging.getLogger(__name__)


@dataclass
class PriceTick:
    """A single price observation from a streaming feed.

    Attributes:
        symbol: The trading pair symbol (e.g. ``"BTC/USDT"``).
        price: Last trade price.
        volume: Trade volume (if available).
        bid: Best bid price (if available).
        ask: Best ask price (if available).
        timestamp: Unix timestamp in seconds.
        exchange: Exchange name.
        raw: Raw ticker data from the exchange.
    """

    symbol: str
    price: float
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None
    timestamp: float = field(default_factory=time.time)
    exchange: str = ""
    raw: dict[str, Any] | None = None

    def spread(self) -> float | None:
        """Return the bid-ask spread, or None if not available."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    def mid_price(self) -> float | None:
        """Return the mid price, or None if bid/ask not available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None


def _normalise_symbol(symbol: str) -> str:
    """Convert user-friendly symbols to CCXT format.

    Args:
        symbol: Symbol like ``"BTC-USD"`` or ``"BTC/USDT"``.

    Returns:
        CCXT-style symbol like ``"BTC/USDT"``.
    """
    s = symbol.upper().replace("-", "/")
    # Common substitution: USD -> USDT for crypto pairs
    if s.endswith("/USD") and not s.endswith("/USDT"):
        s = s + "T"
    return s


TickCallback = Callable[[PriceTick], Any]


class PriceStream:
    """Streaming price feed with async iterator and callback interfaces.

    Args:
        symbols: List of trading pair symbols.
        exchange: Exchange identifier (e.g. ``"binance"``).
        throttle_ms: Minimum interval between ticks per symbol, in
            milliseconds. Defaults to 100.
        max_reconnects: Maximum consecutive reconnection attempts before
            giving up. Defaults to 10.
        reconnect_delay: Initial delay in seconds between reconnection
            attempts (doubles on each retry). Defaults to 1.0.
        use_websocket: Whether to attempt WebSocket streaming via CCXT
            pro. Falls back to polling if unavailable. Defaults to True.
        poll_interval: Polling interval in seconds when using REST
            fallback. Defaults to 1.0.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        exchange: str = "binance",
        throttle_ms: int = 100,
        max_reconnects: int = 10,
        reconnect_delay: float = 1.0,
        use_websocket: bool = True,
        poll_interval: float = 1.0,
    ) -> None:
        self._symbols = [_normalise_symbol(s) for s in symbols]
        self._exchange_name = exchange.lower()
        self._throttle_s = throttle_ms / 1000.0
        self._max_reconnects = max_reconnects
        self._reconnect_delay = reconnect_delay
        self._use_websocket = use_websocket
        self._poll_interval = poll_interval

        self._callbacks: list[TickCallback] = []
        self._queue: asyncio.Queue[PriceTick] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_tick_time: dict[str, float] = {}
        self._exchange: Any = None

    def on_tick(self, callback: TickCallback) -> None:
        """Register a callback to be invoked on each new tick.

        Args:
            callback: A callable that receives a ``PriceTick``.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: TickCallback) -> None:
        """Remove a previously registered callback.

        Args:
            callback: The callback to remove.

        Raises:
            ValueError: If the callback is not registered.
        """
        self._callbacks.remove(callback)

    async def start(self) -> None:
        """Start the streaming feed in the background.

        Creates an asyncio task that fetches prices and dispatches
        them to registered callbacks and the async iterator queue.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the streaming feed and clean up resources."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._exchange is not None:
            with contextlib.suppress(Exception):
                await self._exchange.close()
            self._exchange = None

    @property
    def is_running(self) -> bool:
        """Whether the stream is currently active."""
        return self._running

    def _should_throttle(self, symbol: str) -> bool:
        """Check whether a tick for this symbol should be throttled.

        Args:
            symbol: The trading pair symbol.

        Returns:
            True if the tick should be dropped due to throttling.
        """
        now = time.monotonic()
        last = self._last_tick_time.get(symbol, 0.0)
        if now - last < self._throttle_s:
            return True
        self._last_tick_time[symbol] = now
        return False

    def _make_tick(self, symbol: str, ticker: dict[str, Any]) -> PriceTick:
        """Convert a CCXT ticker dict to a PriceTick.

        Args:
            symbol: The trading pair symbol.
            ticker: Raw ticker data from CCXT.

        Returns:
            A ``PriceTick`` instance.
        """
        ts = ticker.get("timestamp")
        ts = ts / 1000.0 if ts is not None else time.time()

        return PriceTick(
            symbol=symbol,
            price=float(ticker.get("last", 0)),
            volume=ticker.get("quoteVolume") or ticker.get("baseVolume"),
            bid=ticker.get("bid"),
            ask=ticker.get("ask"),
            timestamp=ts,
            exchange=self._exchange_name,
            raw=ticker,
        )

    async def _dispatch(self, tick: PriceTick) -> None:
        """Send a tick to all callbacks and the iterator queue.

        Args:
            tick: The price tick to dispatch.
        """
        await self._queue.put(tick)
        for cb in self._callbacks:
            try:
                result = cb(tick)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in tick callback for %s", tick.symbol)

    async def _run_loop(self) -> None:
        """Main streaming loop with reconnection logic."""
        reconnects = 0
        delay = self._reconnect_delay

        while self._running:
            try:
                if self._use_websocket:
                    try:
                        await self._run_websocket()
                    except ImportError:
                        logger.info(
                            "CCXT pro not available, falling back to polling"
                        )
                        self._use_websocket = False
                        await self._run_polling()
                else:
                    await self._run_polling()

                # If we get here normally, reset reconnect counter
                reconnects = 0
                delay = self._reconnect_delay

            except asyncio.CancelledError:
                raise
            except Exception:
                reconnects += 1
                if reconnects > self._max_reconnects:
                    logger.error(
                        "Max reconnection attempts (%d) reached, stopping",
                        self._max_reconnects,
                    )
                    self._running = False
                    break
                logger.warning(
                    "Connection lost, reconnecting in %.1fs "
                    "(attempt %d/%d)",
                    delay,
                    reconnects,
                    self._max_reconnects,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def _run_websocket(self) -> None:
        """Stream prices via CCXT pro WebSocket.

        Raises:
            ImportError: If ``ccxt.pro`` is not available.
        """
        try:
            import ccxt.pro as ccxtpro  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "ccxt[pro] is required for WebSocket streaming. "
                "Install with: pip install 'ccxt[pro]'"
            ) from None

        exchange_class = getattr(ccxtpro, self._exchange_name, None)
        if exchange_class is None:
            raise ValueError(
                f"Exchange '{self._exchange_name}' not found in ccxt.pro"
            )

        self._exchange = exchange_class()
        try:
            while self._running:
                for symbol in self._symbols:
                    ticker = await self._exchange.watch_ticker(symbol)
                    if not self._should_throttle(symbol):
                        tick = self._make_tick(symbol, ticker)
                        await self._dispatch(tick)
        finally:
            await self._exchange.close()
            self._exchange = None

    async def _run_polling(self) -> None:
        """Stream prices via REST polling fallback."""
        try:
            import ccxt  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "ccxt is required for price streaming. "
                "Install with: pip install ccxt"
            ) from None

        exchange_class = getattr(ccxt, self._exchange_name, None)
        if exchange_class is None:
            raise ValueError(
                f"Exchange '{self._exchange_name}' not found in ccxt"
            )

        self._exchange = exchange_class()
        try:
            while self._running:
                for symbol in self._symbols:
                    try:
                        ticker = self._exchange.fetch_ticker(symbol)
                        if not self._should_throttle(symbol):
                            tick = self._make_tick(symbol, ticker)
                            await self._dispatch(tick)
                    except Exception:
                        logger.warning(
                            "Failed to fetch ticker for %s", symbol
                        )
                await asyncio.sleep(self._poll_interval)
        finally:
            self._exchange = None

    # -- Async iterator interface --

    def __aiter__(self) -> AsyncIterator[PriceTick]:
        return self

    async def __anext__(self) -> PriceTick:
        if not self._running and self._queue.empty():
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            if not self._running:
                raise StopAsyncIteration from None
            raise

    async def __aenter__(self) -> PriceStream:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()


def create_stream(
    symbols: str | Sequence[str],
    exchange: str = "binance",
    throttle_ms: int = 100,
    **kwargs: Any,
) -> PriceStream:
    """Create a price stream for the given symbols.

    This is the main entry point for streaming price data.

    Args:
        symbols: One or more trading pair symbols (e.g.
            ``"BTC-USD"`` or ``["BTC-USD", "ETH-USD"]``).
        exchange: Exchange identifier. Defaults to ``"binance"``.
        throttle_ms: Minimum interval between ticks per symbol.
        **kwargs: Additional arguments passed to ``PriceStream``.

    Returns:
        A configured ``PriceStream`` instance (not yet started).

    Example::

        stream = create_stream(["BTC-USD", "ETH-USD"])
        async for tick in stream:
            print(f"{tick.symbol}: {tick.price}")
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    return PriceStream(
        symbols=symbols,
        exchange=exchange,
        throttle_ms=throttle_ms,
        **kwargs,
    )
