"""On-chain risk analysis demo for Acme Fund.

Generates charts for docs/images/:
- onchain_tvl_concentration.png
- onchain_dependency_graph.png
- onchain_contract_risk.png
"""

import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.crypto.onchain import (
    defi_dependency_graph,
    smart_contract_risk_score,
    tvl_tracker,
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


# --- Chart 1: TVL concentration across DeFi protocols ---
tvls = {
    "Aave": [12e9, 13e9, 14e9, 13.5e9, 15e9],
    "Lido": [18e9, 19e9, 20e9, 22e9, 25e9],
    "MakerDAO": [8e9, 7.5e9, 7e9, 7.2e9, 7e9],
    "Uniswap": [5e9, 5.5e9, 6e9, 5.8e9, 6.2e9],
    "Compound": [3e9, 2.8e9, 2.5e9, 2.3e9, 2e9],
    "Curve": [2e9, 2.2e9, 2.5e9, 2.3e9, 2.4e9],
}

result = tvl_tracker(tvls)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart of TVL shares
shares = result["shares"]
labels = list(shares.keys())
sizes = [s * 100 for s in shares.values()]
colours = [PRIMARY, SECONDARY, POSITIVE, NEUTRAL, NEGATIVE, "#B07AA1"]

wedges, texts, autotexts = ax1.pie(
    sizes, labels=labels, autopct="%1.1f%%",
    colors=colours, startangle=90,
    textprops={"fontsize": 9},
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight("bold")
ax1.set_title(f"DeFi TVL Distribution\nHHI: {result['hhi']:.3f} ({result['risk_rating']})",
              fontsize=11, fontweight="bold")

# Bar chart of TVL trends
_style_ax(ax2, "TVL Trend (Period Change %)", ylabel="Change (%)")
trend_names = []
trend_changes = []
for name, trend in result["trends"].items():
    trend_names.append(name)
    trend_changes.append(trend["total_change_pct"])

bar_colours = [POSITIVE if c > 0 else NEGATIVE for c in trend_changes]
bars = ax2.barh(trend_names, trend_changes, color=bar_colours, height=0.5)
for bar, change in zip(bars, trend_changes):
    x_pos = bar.get_width() + 1 if change >= 0 else bar.get_width() - 1
    ha = "left" if change >= 0 else "right"
    ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
             f"{change:+.1f}%", va="center", ha=ha, fontsize=9, fontweight="bold")
ax2.axvline(x=0, color="#333333", linewidth=0.8)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "onchain_tvl_concentration.png"), dpi=150, facecolor=BG)
plt.close()
print("Saved onchain_tvl_concentration.png")


# --- Chart 2: DeFi dependency graph ---
protocols = [
    {"name": "Yearn", "dependencies": ["Aave", "Curve", "Chainlink"], "category": "yield"},
    {"name": "Aave", "dependencies": ["Chainlink", "USDC", "USDT"], "category": "lending"},
    {"name": "Compound", "dependencies": ["Chainlink", "USDC"], "category": "lending"},
    {"name": "Curve", "dependencies": ["USDC", "USDT", "DAI"], "category": "dex"},
    {"name": "MakerDAO", "dependencies": ["Chainlink", "ETH"], "category": "lending"},
    {"name": "Chainlink", "dependencies": [], "category": "oracle"},
    {"name": "USDC", "dependencies": [], "category": "stablecoin"},
    {"name": "USDT", "dependencies": [], "category": "stablecoin"},
    {"name": "DAI", "dependencies": ["MakerDAO"], "category": "stablecoin"},
    {"name": "ETH", "dependencies": [], "category": "base"},
]

result = defi_dependency_graph(protocols)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor(BG)
fig.set_facecolor(BG)
ax.set_title("DeFi Protocol Dependency Network\n(Acme Fund Exposure Map)",
             fontsize=13, fontweight="bold", pad=12)

# Layout by risk layers
layer_y = {}
for layer_idx, protos in sorted(result["risk_layers"].items()):
    for i, proto in enumerate(protos):
        layer_y[proto] = layer_idx

# Position nodes
positions = {}
for layer_idx, protos in sorted(result["risk_layers"].items()):
    n = len(protos)
    for i, proto in enumerate(protos):
        x = (i - (n - 1) / 2) * 2.0
        y = -layer_idx * 2.0
        positions[proto] = (x, y)

# Category colours
cat_colours = {
    "oracle": SECONDARY, "stablecoin": PRIMARY, "lending": POSITIVE,
    "dex": NEUTRAL, "yield": NEGATIVE, "base": "#B07AA1",
}
proto_cats = {p["name"]: p.get("category", "other") for p in protocols}

# Draw edges
for dep_from, dep_to in result["edges"]:
    if dep_from in positions and dep_to in positions:
        x1, y1 = positions[dep_from]
        x2, y2 = positions[dep_to]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops={"arrowstyle": "->", "color": "#999999", "lw": 1.2})

# Draw nodes
for name, (x, y) in positions.items():
    cat = proto_cats.get(name, "other")
    colour = cat_colours.get(cat, NEUTRAL)
    n_dependants = len(result["reverse_adjacency"].get(name, []))
    size = 400 + n_dependants * 200

    ax.scatter(x, y, s=size, c=colour, zorder=5, edgecolors="#333333", linewidths=1)
    ax.text(x, y - 0.15, name, ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)

# Legend
legend_items = [mpatches.Patch(color=c, label=cat.title()) for cat, c in cat_colours.items()]
ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)

ax.set_xlim(-6, 6)
ax.set_ylim(-9, 1.5)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "onchain_dependency_graph.png"), dpi=150, facecolor=BG)
plt.close()
print("Saved onchain_dependency_graph.png")


# --- Chart 3: Smart contract risk scores ---
contracts = [
    ("Aave v3", 900, True, 15e9, 0.95),
    ("Uniswap v3", 1100, True, 6e9, 0.92),
    ("New yield farm", 45, False, 5e6, 0.4),
    ("Forked lending", 120, False, 50e6, 0.6),
    ("MakerDAO", 1800, True, 8e9, 0.88),
]

fig, ax = plt.subplots(figsize=(8, 5))
_style_ax(ax, "Smart Contract Risk Scores\n(Acme DeFi Allocation Assessment)", ylabel="Safety score")

names = [c[0] for c in contracts]
scores = []
ratings = []
for name, age, audited, tvl, stability in contracts:
    result = smart_contract_risk_score(age, audited, tvl, stability)
    scores.append(result["score"])
    ratings.append(result["risk_rating"])

colour_map = {"low": POSITIVE, "medium": SECONDARY, "high": NEGATIVE, "critical": NEGATIVE}
bar_colours = [colour_map.get(r, NEUTRAL) for r in ratings]

bars = ax.bar(names, scores, color=bar_colours, width=0.55, edgecolor="none")
for bar, score, rating in zip(bars, scores, ratings):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{score:.2f}\n({rating})", ha="center", fontsize=9, fontweight="bold")

ax.set_ylim(0, 1.15)
ax.axhline(y=0.50, color=NEGATIVE, linestyle="--", alpha=0.5)
ax.text(4.4, 0.51, "Minimum for Acme", fontsize=8, color=NEGATIVE)
ax.axhline(y=0.75, color=POSITIVE, linestyle="--", alpha=0.5)
ax.text(4.4, 0.76, "Preferred", fontsize=8, color=POSITIVE)

plt.xticks(fontsize=9, rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "onchain_contract_risk.png"), dpi=150, facecolor=BG)
plt.close()
print("Saved onchain_contract_risk.png")

print("\nAll on-chain charts generated successfully.")
