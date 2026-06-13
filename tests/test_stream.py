"""Tests for the streaming price feed module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from quantlite.data.stream import (
    PriceStream,
    PriceTick,
    _normalise_symbol,
    create_stream,
)


class TestNormaliseSymbol:
    def test_dash_to_slash(self):
        assert _normalise_symbol("BTC-USD") == "BTC/USDT"

    def test_already_slash(self):
        assert _normalise_symbol("BTC/USDT") == "BTC/USDT"

    def test_usd_becomes_usdt(self):
        assert _normalise_symbol("ETH-USD") == "ETH/USDT"

    def test_usdt_stays(self):
        assert _normalise_symbol("ETH-USDT") == "ETH/USDT"

    def test_lowercase(self):
        assert _normalise_symbol("btc-usd") == "BTC/USDT"


class TestPriceTick:
    def test_spread(self):
        tick = PriceTick(symbol="BTC/USDT", price=100, bid=99, ask=101)
        assert tick.spread() == 2.0

    def test_spread_none(self):
        tick = PriceTick(symbol="BTC/USDT", price=100)
        assert tick.spread() is None

    def test_mid_price(self):
        tick = PriceTick(symbol="BTC/USDT", price=100, bid=99, ask=101)
        assert tick.mid_price() == 100.0

    def test_mid_price_none(self):
        tick = PriceTick(symbol="BTC/USDT", price=100)
        assert tick.mid_price() is None


class TestCreateStream:
    def test_single_symbol(self):
        stream = create_stream("BTC-USD")
        assert isinstance(stream, PriceStream)
        assert stream._symbols == ["BTC/USDT"]

    def test_multiple_symbols(self):
        stream = create_stream(["BTC-USD", "ETH-USD"])
        assert len(stream._symbols) == 2

    def test_custom_exchange(self):
        stream = create_stream("BTC-USD", exchange="kraken")
        assert stream._exchange_name == "kraken"


class TestPriceStream:
    def test_callback_registration(self):
        stream = PriceStream(["BTC/USDT"])
        cb = MagicMock()
        stream.on_tick(cb)
        assert cb in stream._callbacks

    def test_callback_removal(self):
        stream = PriceStream(["BTC/USDT"])
        cb = MagicMock()
        stream.on_tick(cb)
        stream.remove_callback(cb)
        assert cb not in stream._callbacks

    def test_not_running_initially(self):
        stream = PriceStream(["BTC/USDT"])
        assert not stream.is_running

    @pytest.mark.asyncio
    async def test_dispatch_to_queue(self):
        stream = PriceStream(["BTC/USDT"])
        tick = PriceTick(symbol="BTC/USDT", price=50000)
        await stream._dispatch(tick)
        result = stream._queue.get_nowait()
        assert result.price == 50000

    @pytest.mark.asyncio
    async def test_dispatch_to_callback(self):
        stream = PriceStream(["BTC/USDT"])
        received = []
        stream.on_tick(lambda t: received.append(t))
        tick = PriceTick(symbol="BTC/USDT", price=50000)
        await stream._dispatch(tick)
        assert len(received) == 1
        assert received[0].price == 50000

    @pytest.mark.asyncio
    async def test_dispatch_async_callback(self):
        stream = PriceStream(["BTC/USDT"])
        received = []

        async def handler(t):
            received.append(t)

        stream.on_tick(handler)
        tick = PriceTick(symbol="BTC/USDT", price=50000)
        await stream._dispatch(tick)
        assert len(received) == 1

    def test_throttle(self):
        stream = PriceStream(["BTC/USDT"], throttle_ms=1000)
        # First call should not throttle
        assert not stream._should_throttle("BTC/USDT")
        # Immediate second call should throttle
        assert stream._should_throttle("BTC/USDT")

    def test_make_tick(self):
        stream = PriceStream(["BTC/USDT"], exchange="binance")
        ticker = {
            "last": 50000,
            "bid": 49999,
            "ask": 50001,
            "timestamp": 1700000000000,
            "quoteVolume": 1e6,
        }
        tick = stream._make_tick("BTC/USDT", ticker)
        assert tick.price == 50000
        assert tick.bid == 49999
        assert tick.ask == 50001
        assert tick.exchange == "binance"
        assert tick.timestamp == 1700000000.0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that start/stop work via context manager (mocked)."""
        stream = PriceStream(["BTC/USDT"], use_websocket=False)
        stream._run_loop = AsyncMock()

        async with stream:
            assert stream.is_running

        assert not stream.is_running

    @pytest.mark.asyncio
    async def test_aiter_interface(self):
        stream = PriceStream(["BTC/USDT"])
        stream._running = True

        tick = PriceTick(symbol="BTC/USDT", price=50000)
        await stream._queue.put(tick)

        result = await stream.__anext__()
        assert result.price == 50000
