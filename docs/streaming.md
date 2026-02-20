# Price Streaming

QuantLite provides real-time price streaming from cryptocurrency exchanges via WebSocket connections, with automatic fallback to REST polling. The streaming module supports both async iteration and callback interfaces, with built-in throttling and reconnection logic.

## Overview

The streaming system is built around two core components:

- **`PriceStream`** — the main streaming class that manages connections, dispatches ticks, and handles reconnection
- **`PriceTick`** — a dataclass representing a single price observation with symbol, price, volume, bid/ask, and timestamp

Streaming uses [CCXT](https://github.com/ccxt/ccxt) under the hood, supporting 100+ exchanges. WebSocket streaming (via CCXT Pro) is preferred, with automatic fallback to REST polling if unavailable.

## Installation

```bash
pip install "quantlite[stream]"
```

This installs `ccxt` with WebSocket support. For REST-only polling, the base `ccxt` package suffices.

## Quick Start

```python
import asyncio
import quantlite as ql

async def main():
    stream = ql.stream(["BTC-USD", "ETH-USD"], exchange="binance")

    async for tick in stream:
        print(f"{tick.symbol}: ${tick.price:,.2f} "
              f"(spread: {tick.spread() or 'N/A'})")

asyncio.run(main())
```

## API Reference

### `PriceTick`

A single price observation from a streaming feed.

| Attribute   | Type              | Description                              |
|-------------|-------------------|------------------------------------------|
| `symbol`    | `str`             | Trading pair (e.g. `"BTC/USDT"`)         |
| `price`     | `float`           | Last trade price                         |
| `volume`    | `float \| None`   | Trade volume (if available)              |
| `bid`       | `float \| None`   | Best bid price                           |
| `ask`       | `float \| None`   | Best ask price                           |
| `timestamp` | `float`           | Unix timestamp in seconds                |
| `exchange`  | `str`             | Exchange name                            |
| `raw`       | `dict \| None`    | Raw ticker data from the exchange        |

**Methods:**

- `spread() -> float | None` — returns the bid-ask spread, or `None` if bid/ask are unavailable
- `mid_price() -> float | None` — returns the mid price, or `None` if bid/ask are unavailable

### `PriceStream`

Streaming price feed with async iterator and callback interfaces.

**Constructor parameters:**

| Parameter         | Type           | Default     | Description                                      |
|-------------------|----------------|-------------|--------------------------------------------------|
| `symbols`         | `Sequence[str]`| (required)  | Trading pair symbols                             |
| `exchange`        | `str`          | `"binance"` | Exchange identifier                              |
| `throttle_ms`     | `int`          | `100`       | Min interval between ticks per symbol (ms)       |
| `max_reconnects`  | `int`          | `10`        | Max consecutive reconnection attempts            |
| `reconnect_delay` | `float`        | `1.0`       | Initial reconnection delay (seconds, doubles)    |
| `use_websocket`   | `bool`         | `True`      | Attempt WebSocket streaming via CCXT Pro         |
| `poll_interval`   | `float`        | `1.0`       | REST polling interval in seconds                 |

**Methods:**

- `on_tick(callback)` — register a callback invoked on each new tick
- `remove_callback(callback)` — remove a previously registered callback
- `start()` — start the streaming feed (async)
- `stop()` — stop the feed and clean up resources (async)
- `is_running` — property indicating whether the stream is active

**Context manager:** `PriceStream` supports `async with` for automatic start/stop.

### `create_stream` / `ql.stream`

Factory function (also available as `ql.stream`):

```python
stream = ql.stream(
    symbols=["BTC-USD", "ETH-USD"],
    exchange="binance",
    throttle_ms=200,
)
```

Accepts a single symbol string or a list. Symbols like `"BTC-USD"` are automatically normalised to CCXT format (`"BTC/USDT"`).

## Examples

### Callback Interface

Use callbacks when you want to process ticks without managing an async loop:

```python
import asyncio
import quantlite as ql

prices: dict[str, float] = {}

def on_price(tick: ql.PriceTick) -> None:
    prices[tick.symbol] = tick.price
    spread = tick.spread()
    print(f"[{tick.exchange}] {tick.symbol}: "
          f"${tick.price:,.2f} | spread: ${spread:,.4f}" if spread else "")

async def main():
    stream = ql.stream(["BTC-USD", "ETH-USD", "SOL-USD"], exchange="binance")
    stream.on_tick(on_price)

    await stream.start()

    # Run for 60 seconds, then stop
    await asyncio.sleep(60)
    await stream.stop()

asyncio.run(main())
```

### Context Manager with Async For

```python
import asyncio
import quantlite as ql

async def main():
    stream = ql.stream(["BTC-USD"], exchange="coinbase")

    async with stream:
        count = 0
        async for tick in stream:
            print(f"${tick.price:,.2f} at {tick.timestamp:.0f}")
            count += 1
            if count >= 100:
                break

asyncio.run(main())
```

### Computing Rolling Returns from a Stream

A realistic pattern: accumulate ticks and compute rolling log returns for downstream analysis.

```python
import asyncio
import math
from collections import deque

import quantlite as ql

async def rolling_returns():
    window: deque[float] = deque(maxlen=60)
    stream = ql.stream(["BTC-USD"], exchange="binance", throttle_ms=1000)

    async with stream:
        async for tick in stream:
            if window:
                log_ret = math.log(tick.price / window[-1])
                print(f"BTC log return: {log_ret:+.6f}")
            window.append(tick.price)

asyncio.run(rolling_returns())
```

### Multi-Exchange Streaming

Run streams from multiple exchanges simultaneously:

```python
import asyncio
import quantlite as ql

async def multi_exchange():
    binance = ql.stream(["BTC-USD"], exchange="binance")
    coinbase = ql.stream(["BTC-USD"], exchange="coinbase")

    async def consume(name: str, s: ql.PriceStream):
        async with s:
            async for tick in s:
                print(f"[{name}] BTC: ${tick.price:,.2f}")

    await asyncio.gather(
        consume("Binance", binance),
        consume("Coinbase", coinbase),
    )

asyncio.run(multi_exchange())
```

### REST Polling Fallback

If WebSocket support is unavailable, force REST polling:

```python
stream = ql.stream(
    ["BTC-USD"],
    exchange="kraken",
    use_websocket=False,
    poll_interval=2.0,  # Poll every 2 seconds
)
```

## Reconnection Behaviour

The stream automatically reconnects on connection failures:

1. On disconnect, the stream waits `reconnect_delay` seconds (default: 1.0)
2. Each subsequent failure doubles the delay, up to 60 seconds
3. After `max_reconnects` consecutive failures (default: 10), the stream stops
4. A successful reconnection resets the counter

You can tune this behaviour:

```python
stream = ql.stream(
    ["BTC-USD"],
    max_reconnects=20,      # Try harder
    reconnect_delay=0.5,    # Start retrying faster
)
```

## Exchange Configuration

QuantLite supports any exchange available in CCXT. Common choices:

| Exchange   | Identifier    | WebSocket | Notes                          |
|------------|---------------|-----------|--------------------------------|
| Binance    | `"binance"`   | Yes       | Highest liquidity              |
| Coinbase   | `"coinbase"`  | Yes       | US-regulated                   |
| Kraken     | `"kraken"`    | Yes       | Good for EUR pairs             |
| OKX        | `"okx"`       | Yes       | Derivatives                    |
| Bybit      | `"bybit"`     | Yes       | Perpetuals                     |

## Throttling

Throttling prevents downstream systems from being overwhelmed by high-frequency tick data. The `throttle_ms` parameter sets the minimum interval between ticks for each symbol independently.

- `throttle_ms=100` (default): at most 10 ticks per second per symbol
- `throttle_ms=1000`: at most 1 tick per second per symbol (suitable for regime detection)
- `throttle_ms=0`: no throttling (every tick is dispatched)

## Visualisation Concept

A typical monitoring dashboard built on QuantLite streaming would display:

- **Real-time price chart** with a rolling 1-hour window, updating on each tick
- **Bid-ask spread panel** showing spread evolution (useful for detecting liquidity deterioration)
- **Tick rate indicator** showing ticks per second, coloured by throttle status
- **Multi-exchange price comparison** overlaying the same symbol from different venues to spot arbitrage opportunities

These can be built with Plotly Dash or a similar framework, consuming ticks from `PriceStream` callbacks.
