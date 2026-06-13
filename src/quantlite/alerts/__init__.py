"""Alert system for QuantLite.

Provides rule-based and threshold alerts with callback/webhook
mechanisms for real-time monitoring of market conditions,
regime changes, and portfolio metrics.

Example::

    import quantlite as ql

    def notify(alert):
        print(f"ALERT: {alert}")

    manager = ql.AlertManager()
    manager.add_rule("BTC-USD", condition="regime_change", callback=notify)
    manager.add_threshold("portfolio_var", threshold=0.05, direction="above")

    # Check alerts against new data
    manager.check(metric="BTC-USD", value=1, regime=2, previous_regime=1)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Alert",
    "AlertManager",
    "AlertRule",
    "AlertStatus",
    "ThresholdDirection",
]

logger = logging.getLogger(__name__)


class AlertStatus(Enum):
    """Status of a fired alert."""

    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ThresholdDirection(Enum):
    """Direction for threshold-based alerts."""

    ABOVE = "above"
    BELOW = "below"
    CROSS = "cross"


@dataclass
class Alert:
    """A fired alert record.

    Attributes:
        rule_name: Name of the rule that triggered.
        metric: The metric or symbol being monitored.
        message: Human-readable alert message.
        value: The value that triggered the alert.
        timestamp: Unix timestamp when the alert fired.
        status: Current alert status.
        metadata: Additional context about the alert.
    """

    rule_name: str
    metric: str
    message: str
    value: float | None = None
    timestamp: float = field(default_factory=time.time)
    status: AlertStatus = AlertStatus.TRIGGERED
    metadata: dict[str, Any] = field(default_factory=dict)


AlertCallback = Callable[[Alert], Any]


@dataclass
class AlertRule:
    """A configured alert rule.

    Attributes:
        name: Unique name for this rule.
        metric: The metric or symbol to monitor.
        condition: Condition type (e.g. ``"regime_change"``,
            ``"threshold"``).
        callback: Function to call when the alert fires.
        threshold: Threshold value (for threshold alerts).
        direction: Direction for threshold comparison.
        cooldown_s: Minimum seconds between repeated firings
            of the same rule. Defaults to 60.
        enabled: Whether this rule is active.
    """

    name: str
    metric: str
    condition: str
    callback: AlertCallback | None = None
    threshold: float | None = None
    direction: ThresholdDirection = ThresholdDirection.ABOVE
    cooldown_s: float = 60.0
    enabled: bool = True
    _last_fired: float = field(default=0.0, repr=False)
    _last_value: float | None = field(default=None, repr=False)


class AlertManager:
    """Manages alert rules, checks conditions, and maintains history.

    Args:
        max_history: Maximum number of alerts to keep in the
            history log. Defaults to 1000.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._history: list[Alert] = []
        self._max_history = max_history
        self._global_callbacks: list[AlertCallback] = []

    @property
    def rules(self) -> dict[str, AlertRule]:
        """All configured alert rules, keyed by name."""
        return dict(self._rules)

    @property
    def history(self) -> list[Alert]:
        """List of all fired alerts (most recent last)."""
        return list(self._history)

    def on_alert(self, callback: AlertCallback) -> None:
        """Register a global callback for all alerts.

        Args:
            callback: Function called whenever any alert fires.
        """
        self._global_callbacks.append(callback)

    def add_rule(
        self,
        metric: str,
        condition: str = "regime_change",
        callback: AlertCallback | None = None,
        name: str | None = None,
        cooldown_s: float = 60.0,
    ) -> AlertRule:
        """Add a rule-based alert.

        Args:
            metric: The metric or symbol to monitor.
            condition: Condition type. Currently supported:
                ``"regime_change"``.
            callback: Optional callback when the alert fires.
            name: Unique name. Auto-generated if not provided.
            cooldown_s: Cooldown between repeated firings.

        Returns:
            The created ``AlertRule``.
        """
        if name is None:
            name = f"{metric}_{condition}_{len(self._rules)}"

        rule = AlertRule(
            name=name,
            metric=metric,
            condition=condition,
            callback=callback,
            cooldown_s=cooldown_s,
        )
        self._rules[name] = rule
        return rule

    def add_threshold(
        self,
        metric: str,
        threshold: float,
        direction: str = "above",
        callback: AlertCallback | None = None,
        name: str | None = None,
        cooldown_s: float = 60.0,
    ) -> AlertRule:
        """Add a threshold-based alert.

        Args:
            metric: The metric or symbol to monitor.
            threshold: The threshold value.
            direction: One of ``"above"``, ``"below"``, or
                ``"cross"``.
            callback: Optional callback when the alert fires.
            name: Unique name. Auto-generated if not provided.
            cooldown_s: Cooldown between repeated firings.

        Returns:
            The created ``AlertRule``.
        """
        if name is None:
            name = f"{metric}_threshold_{len(self._rules)}"

        dir_enum = ThresholdDirection(direction.lower())

        rule = AlertRule(
            name=name,
            metric=metric,
            condition="threshold",
            callback=callback,
            threshold=threshold,
            direction=dir_enum,
            cooldown_s=cooldown_s,
        )
        self._rules[name] = rule
        return rule

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule by name.

        Args:
            name: The rule name to remove.

        Raises:
            KeyError: If no rule with that name exists.
        """
        if name not in self._rules:
            raise KeyError(f"No alert rule named '{name}'")
        del self._rules[name]

    def enable_rule(self, name: str) -> None:
        """Enable a disabled alert rule.

        Args:
            name: The rule name to enable.
        """
        self._rules[name].enabled = True

    def disable_rule(self, name: str) -> None:
        """Disable an alert rule without removing it.

        Args:
            name: The rule name to disable.
        """
        self._rules[name].enabled = False

    def check(
        self,
        metric: str,
        value: float | None = None,
        regime: int | None = None,
        previous_regime: int | None = None,
        **metadata: Any,
    ) -> list[Alert]:
        """Check all rules for the given metric and fire matching alerts.

        Args:
            metric: The metric or symbol being reported.
            value: Current value of the metric.
            regime: Current regime index (for regime change alerts).
            previous_regime: Previous regime index.
            **metadata: Additional context passed to the alert.

        Returns:
            List of alerts that were fired.
        """
        now = time.time()
        fired: list[Alert] = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.metric != metric:
                continue
            if now - rule._last_fired < rule.cooldown_s:
                continue

            alert = self._evaluate_rule(
                rule, value, regime, previous_regime, metadata
            )
            if alert is not None:
                rule._last_fired = now
                rule._last_value = value
                self._record_alert(alert)
                fired.append(alert)

        return fired

    def check_many(
        self,
        updates: Sequence[dict[str, Any]],
    ) -> list[Alert]:
        """Check multiple metric updates at once.

        Args:
            updates: Sequence of dicts, each with keys matching
                the ``check()`` parameters.

        Returns:
            List of all alerts fired across all updates.
        """
        all_fired: list[Alert] = []
        for update in updates:
            all_fired.extend(self.check(**update))
        return all_fired

    def clear_history(self) -> None:
        """Clear the alert history log."""
        self._history.clear()

    def _evaluate_rule(
        self,
        rule: AlertRule,
        value: float | None,
        regime: int | None,
        previous_regime: int | None,
        metadata: dict[str, Any],
    ) -> Alert | None:
        """Evaluate a single rule and return an Alert if triggered.

        Args:
            rule: The rule to evaluate.
            value: Current metric value.
            regime: Current regime.
            previous_regime: Previous regime.
            metadata: Extra context.

        Returns:
            An ``Alert`` if the rule fires, otherwise None.
        """
        if rule.condition == "regime_change":
            if (
                regime is not None
                and previous_regime is not None
                and regime != previous_regime
            ):
                return Alert(
                    rule_name=rule.name,
                    metric=rule.metric,
                    message=(
                        f"Regime change on {rule.metric}: "
                        f"{previous_regime} -> {regime}"
                    ),
                    value=float(regime),
                    metadata={
                        "previous_regime": previous_regime,
                        "new_regime": regime,
                        **metadata,
                    },
                )

        elif (
            rule.condition == "threshold"
            and value is not None
            and rule.threshold is not None
        ):
                triggered = False
                if rule.direction == ThresholdDirection.ABOVE:
                    triggered = value > rule.threshold
                elif rule.direction == ThresholdDirection.BELOW:
                    triggered = value < rule.threshold
                elif rule.direction == ThresholdDirection.CROSS:
                    if rule._last_value is not None:
                        crossed_up = (
                            rule._last_value <= rule.threshold
                            and value > rule.threshold
                        )
                        crossed_down = (
                            rule._last_value >= rule.threshold
                            and value < rule.threshold
                        )
                        triggered = crossed_up or crossed_down
                    rule._last_value = value

                if triggered:
                    return Alert(
                        rule_name=rule.name,
                        metric=rule.metric,
                        message=(
                            f"{rule.metric} is {rule.direction.value} "
                            f"threshold {rule.threshold}: {value}"
                        ),
                        value=value,
                        metadata={"threshold": rule.threshold, **metadata},
                    )

        return None

    def _record_alert(self, alert: Alert) -> None:
        """Record an alert and invoke callbacks.

        Args:
            alert: The alert to record.
        """
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Fire rule-specific callback
        rule = self._rules.get(alert.rule_name)
        if rule is not None and rule.callback is not None:
            try:
                rule.callback(alert)
            except Exception:
                logger.exception(
                    "Error in alert callback for rule %s", rule.name
                )

        # Fire global callbacks
        for cb in self._global_callbacks:
            try:
                cb(alert)
            except Exception:
                logger.exception("Error in global alert callback")
