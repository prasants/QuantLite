"""Decision audit trail: justify every portfolio weight with full provenance.

Logs input data summary, method used, constraints applied, regime state,
resulting weights, and rationale. Exportable as JSON, markdown, or HTML.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

__all__ = [
    "AuditEntry",
    "AuditTrail",
    "compare_weights",
]


@dataclass
class AuditEntry:
    """A single audit record for a portfolio decision.

    Attributes:
        timestamp: When the decision was made.
        method: Allocation method used (e.g. "Risk Parity", "HRP").
        input_summary: Summary statistics of input data.
        constraints: Constraints applied during optimisation.
        regime_state: Current regime information (if available).
        weights: Resulting portfolio weights.
        rationale: Per-asset explanation of why each weight is what it is.
        metadata: Additional context or parameters.
    """

    timestamp: str
    method: str
    input_summary: dict[str, Any]
    constraints: dict[str, Any]
    regime_state: dict[str, Any] | None
    weights: dict[str, float]
    rationale: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditTrail:
    """Maintains a log of portfolio decisions with full justification.

    Each portfolio rebalance produces an ``AuditEntry`` that records
    inputs, method, constraints, resulting weights, and rationale.

    Args:
        name: Name for this audit trail (e.g. portfolio name).

    Example:
        >>> trail = AuditTrail("Growth Portfolio")
        >>> trail.log(
        ...     method="Risk Parity",
        ...     returns_df=returns,
        ...     weights={"Equity": 0.4, "Bonds": 0.6},
        ...     constraints={"max_weight": 0.6},
        ... )
        >>> print(trail.to_markdown())
    """

    def __init__(self, name: str = "Portfolio") -> None:
        self.name = name
        self.entries = []  # type: List[AuditEntry]

    def log(
        self,
        method: str,
        weights: dict[str, float],
        returns_df: pd.DataFrame | None = None,
        constraints: dict[str, Any] | None = None,
        regime_state: dict[str, Any] | None = None,
        rationale: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a portfolio decision.

        Args:
            method: Allocation method name.
            weights: Resulting portfolio weights.
            returns_df: Input return data (used to generate summary).
            constraints: Constraints applied.
            regime_state: Current regime information.
            rationale: Per-asset rationale. Auto-generated if not provided.
            metadata: Additional context.

        Returns:
            The created ``AuditEntry``.
        """
        # Build input summary
        input_summary = {}  # type: Dict[str, Any]
        if returns_df is not None:
            input_summary = {
                "n_assets": len(returns_df.columns),
                "n_observations": len(returns_df),
                "assets": list(returns_df.columns),
                "date_range": (
                    f"{returns_df.index[0]} to {returns_df.index[-1]}"
                    if hasattr(returns_df.index[0], "strftime")
                    else f"0 to {len(returns_df) - 1}"
                ),
                "mean_returns": {
                    col: float(returns_df[col].mean()) for col in returns_df.columns
                },
                "volatilities": {
                    col: float(returns_df[col].std()) for col in returns_df.columns
                },
            }

        # Auto-generate rationale if not provided
        if rationale is None:
            rationale = _auto_rationale(method, weights, input_summary, constraints)

        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            method=method,
            input_summary=input_summary,
            constraints=constraints or {},
            regime_state=regime_state,
            weights=weights,
            rationale=rationale,
            metadata=metadata or {},
        )

        self.entries.append(entry)
        return entry

    def to_json(self, indent: int = 2) -> str:
        """Export the full audit trail as JSON.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        data = {
            "name": self.name,
            "entries": [asdict(e) for e in self.entries],
        }
        return json.dumps(data, indent=indent, default=str)

    def to_markdown(self) -> str:
        """Export the full audit trail as markdown.

        Returns:
            Markdown-formatted string.
        """
        lines = [f"# Audit Trail: {self.name}\n"]

        for i, entry in enumerate(self.entries, 1):
            lines.append(f"## Decision {i} — {entry.timestamp}\n")
            lines.append(f"**Method:** {entry.method}\n")

            if entry.input_summary:
                lines.append("**Input Summary:**\n")
                for k, v in entry.input_summary.items():
                    if isinstance(v, dict):
                        lines.append(f"- {k}:")
                        for kk, vv in v.items():
                            lines.append(f"  - {kk}: {vv:.6f}" if isinstance(vv, float) else f"  - {kk}: {vv}")
                    else:
                        lines.append(f"- {k}: {v}")
                lines.append("")

            if entry.constraints:
                lines.append("**Constraints:**\n")
                for k, v in entry.constraints.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")

            if entry.regime_state:
                lines.append("**Regime State:**\n")
                for k, v in entry.regime_state.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")

            lines.append("**Weights:**\n")
            lines.append("| Asset | Weight |")
            lines.append("|-------|--------|")
            for asset, w in sorted(entry.weights.items(), key=lambda x: -x[1]):
                lines.append(f"| {asset} | {w:.4f} |")
            lines.append("")

            lines.append("**Rationale:**\n")
            for asset, reason in entry.rationale.items():
                lines.append(f"- **{asset}:** {reason}")
            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Export the full audit trail as HTML.

        Returns:
            HTML string.
        """
        html = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>Audit Trail: {self.name}</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; "
            "max-width: 800px; margin: 2em auto; color: #333; }",
            "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background: #4E79A7; color: white; }",
            "tr:nth-child(even) { background: #f9f9f9; }",
            ".positive { color: #59A14F; } .negative { color: #E15759; }",
            "</style></head><body>",
            f"<h1>Audit Trail: {self.name}</h1>",
        ]

        for i, entry in enumerate(self.entries, 1):
            html.append(f"<h2>Decision {i} &mdash; {entry.timestamp}</h2>")
            html.append(f"<p><strong>Method:</strong> {entry.method}</p>")

            html.append("<table><tr><th>Asset</th><th>Weight</th><th>Rationale</th></tr>")
            for asset, w in sorted(entry.weights.items(), key=lambda x: -x[1]):
                reason = entry.rationale.get(asset, "")
                html.append(f"<tr><td>{asset}</td><td>{w:.4f}</td><td>{reason}</td></tr>")
            html.append("</table>")

        html.append("</body></html>")
        return "\n".join(html)


def _auto_rationale(
    method: str,
    weights: dict[str, float],
    input_summary: dict[str, Any],
    constraints: dict[str, Any] | None,
) -> dict[str, str]:
    """Generate automatic rationale for each weight."""
    rationale = {}
    total_w = sum(weights.values())
    vols = input_summary.get("volatilities", {})
    means = input_summary.get("mean_returns", {})

    for asset, w in weights.items():
        parts = []
        pct = w / total_w * 100 if total_w > 0 else 0

        if "parity" in method.lower():
            vol = vols.get(asset)
            if vol is not None:
                parts.append(
                    f"allocated {pct:.1f}% via {method} "
                    f"(volatility {vol:.4%}; lower volatility attracts higher weight)"
                )
            else:
                parts.append(f"allocated {pct:.1f}% via {method}")
        elif "kelly" in method.lower():
            mr = means.get(asset)
            if mr is not None:
                parts.append(
                    f"allocated {pct:.1f}% via {method} "
                    f"(mean return {mr:.4%}; Kelly sizing based on edge/odds)"
                )
            else:
                parts.append(f"allocated {pct:.1f}% via {method}")
        else:
            parts.append(f"allocated {pct:.1f}% via {method}")

        if constraints:
            max_w = constraints.get("max_weight")
            if max_w and w >= max_w - 1e-6:
                parts.append(f"capped at maximum weight constraint ({max_w:.0%})")

        rationale[asset] = "; ".join(parts) + "."

    return rationale


def compare_weights(
    previous: dict[str, float],
    current: dict[str, float],
    method: str = "",
    regime_change: str | None = None,
) -> dict[str, str]:
    """Compare two weight sets and explain the changes.

    Args:
        previous: Previous portfolio weights.
        current: Current portfolio weights.
        method: Allocation method for context.
        regime_change: Description of regime change (if any).

    Returns:
        Dictionary mapping asset names to change explanations.
    """
    all_assets = sorted(set(list(previous.keys()) + list(current.keys())))
    explanations = {}

    for asset in all_assets:
        old_w = previous.get(asset, 0.0)
        new_w = current.get(asset, 0.0)
        delta = new_w - old_w

        if abs(delta) < 1e-6:
            explanations[asset] = f"unchanged at {new_w:.4f}."
        elif delta > 0:
            driver = ""
            if regime_change:
                driver = f" driven by {regime_change}"
            explanations[asset] = (
                f"increased from {old_w:.4f} to {new_w:.4f} "
                f"(+{delta:.4f}){driver}."
            )
        else:
            driver = ""
            if regime_change:
                driver = f" driven by {regime_change}"
            explanations[asset] = (
                f"decreased from {old_w:.4f} to {new_w:.4f} "
                f"({delta:.4f}){driver}."
            )

    return explanations
