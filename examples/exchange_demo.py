"""Exchange risk analysis demo for Acme Fund.

Generates charts for docs/images/:
- exchange_concentration.png
- exchange_wallet_risk.png
- exchange_slippage.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.crypto.exchange import (
    concentration_score,
    slippage_estimate,
    wallet_risk_assessment,
)

PRIMARY = "#4E79A7"
SECONDARY = "#F28E2B"
NEGATIVE = "#E15759"
POSITIVE = "#59A14F"
NEUTRAL = "#76B7B2"
BG = "white"

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, color="#cccccc")
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --- Chart 1: Exchange concentration ---
scenarios = {
    "Acme (diversified)": {"Binance": 200e6, "Coinbase": 180e6, "Kraken": 150e6, "OKX": 120e6, "Self-custody": 350e6},
    "Competitor A": {"Binance": 800e6, "Coinbase": 150e6, "Kraken": 50e6},
    "Competitor B": {"Binance": 950e6, "Other": 50e6},
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (name, balances) in zip(axes, scenarios.items()):
    result = concentration_score(balances)
    shares = result["shares"]
    labels = list(shares.keys())
    sizes = [s * 100 for s in shares.values()]
    colours = [PRIMARY, SECONDARY, POSITIVE, NEUTRAL, NEGATIVE][:len(labels)]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=colours, startangle=90,
        textprops={"fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")

    ax.set_title(f"{name}\nHHI: {result['hhi']:.3f} ({result['risk_rating']})",
                 fontsize=10, fontweight="bold")

fig.suptitle("Exchange Concentration Risk: Acme vs Competitors", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "exchange_concentration.png"), dpi=150, facecolor=BG, bbox_inches="tight")
plt.close()
print("Saved exchange_concentration.png")


# --- Chart 2: Wallet risk across exchange profiles ---
profiles = [
    ("Acme (best practice)", 3, 97, 1_000_000_000),
    ("Mid-tier exchange", 15, 85, 500_000_000),
    ("Small exchange", 30, 70, 50_000_000),
    ("Pre-FTX era", 60, 40, 8_000_000_000),
]

fig, ax = plt.subplots(figsize=(8, 4.5))
_style_ax(ax, "Hot Wallet Risk Assessment by Exchange Profile", ylabel="Risk score")

names = [p[0] for p in profiles]
risk_scores = []
ratings = []
for name, hot, cold, total in profiles:
    result = wallet_risk_assessment(hot, cold, total)
    risk_scores.append(result["risk_score"])
    ratings.append(result["risk_rating"])

colour_map = {"low": POSITIVE, "medium": SECONDARY, "high": NEGATIVE, "critical": NEGATIVE}
bar_colours = [colour_map.get(r, NEUTRAL) for r in ratings]

bars = ax.bar(names, risk_scores, color=bar_colours, width=0.55, edgecolor="none")
for bar, score, (name, hot, cold, total) in zip(bars, risk_scores, profiles):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{score:.2f}\n({hot}% hot)", ha="center", fontsize=9, fontweight="bold")

ax.set_ylim(0, 1.2)
ax.axhline(y=0.30, color=NEUTRAL, linestyle="--", alpha=0.5)
ax.text(3.4, 0.31, "Acceptable threshold", fontsize=8, color=NEUTRAL)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "exchange_wallet_risk.png"), dpi=150, facecolor=BG)
plt.close()
print("Saved exchange_wallet_risk.png")


# --- Chart 3: Slippage estimate across trade sizes ---
# Simulate a BTC/USDT order book (asks)
rng = np.random.default_rng(42)
base_price = 65_000
levels = []
for i in range(50):
    price = base_price + i * 5 + rng.normal(0, 1)
    qty = max(0.1, rng.exponential(2.0))
    levels.append((price, qty))

trade_sizes = [1, 5, 10, 25, 50, 100, 200]
slippages = []
for size in trade_sizes:
    result = slippage_estimate(levels, size)
    slippages.append(result["slippage_pct"])

fig, ax = plt.subplots(figsize=(8, 4.5))
_style_ax(ax, "BTC/USDT Slippage Estimate by Trade Size", xlabel="Trade size (BTC)", ylabel="Slippage (%)")

ax.plot(trade_sizes, slippages, color=PRIMARY, linewidth=2, marker="o", markersize=6)

# Annotate key points
for size, slip in zip(trade_sizes, slippages):
    if size in (1, 25, 100, 200):
        ax.annotate(f"{slip:.2f}%", (size, slip), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, fontweight="bold", color=PRIMARY)

# Risk zones
ax.axhspan(0, 0.1, alpha=0.05, color=POSITIVE)
ax.axhspan(0.1, 0.5, alpha=0.05, color=SECONDARY)
ax.axhspan(0.5, max(slippages) * 1.2, alpha=0.05, color=NEGATIVE)
ax.text(trade_sizes[-1] * 0.95, 0.05, "Low impact", fontsize=8, color=POSITIVE, ha="right")
ax.text(trade_sizes[-1] * 0.95, 0.3, "Moderate", fontsize=8, color=SECONDARY, ha="right")
ax.text(trade_sizes[-1] * 0.95, max(slippages) * 0.85, "High impact", fontsize=8, color=NEGATIVE, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "exchange_slippage.png"), dpi=150, facecolor=BG)
plt.close()
print("Saved exchange_slippage.png")

print("\nAll exchange charts generated successfully.")
