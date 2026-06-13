# Continuous Monitoring

A point-in-time score says how good a record is today. Monitoring answers the
question an allocator actually loses sleep over: *is this record
deteriorating, and will I see it before it costs me?* The firewall
([`docs/firewall.md`](firewall.md)) makes each score reproducible and
independently sourced; monitoring runs that machinery forward through time
and raises an alarm when the picture turns — early enough to act.

## Dual-window scoring

The **expanding window** scores the entire history. It is stable, hard to
game, and is the authoritative published number. But an expanding window is
*slow*: a long, strong history mathematically buries a weak recent stretch,
so by the time the headline grade moves, the damage is done.

So each observation is also scored over a **trailing window** of recent
periods, compared against the **preceding window of equal length**. Equal
length is the crux — it makes the comparison *length-fair*. A naive "recent
vs. whole history" gap is contaminated by record length: the sufficiency
component and the Deflated Sharpe both reward more data, so a short window
scores lower regardless of performance and every record looks like it is
deteriorating. Comparing two windows of the same size removes that artifact;
any gap is a genuine change in performance.

The divergence is measured on **annualised Sharpe**, not the composite score.
The composite is capped by integrity rules (a short window trips the
short-record cap and saturates at 40), which would flatten the signal exactly
where it needs to be sharp. Sharpe stays continuous and sensitive on
sub-year windows, so deterioration shows up early.

## Cadence

`Cadence` covers `DAILY`, `WEEKLY`, `MONTHLY`, and `YEARLY`; the cadence sets
the annualisation factor (252 / 52 / 12 / 1). A strategy can be monitored at
several cadences in parallel by running several monitors — daily for fast
tripwires, monthly and yearly for stable governance reporting.

## Alerts

`Monitor.observe(returns, as_of=...)` scores the new observation, appends a
snapshot, and returns the alerts triggered by the transition. Each `Alert`
carries the triggering snapshot's hash for traceability.

| Kind | Severity | Fires when |
| --- | --- | --- |
| `TRAILING_DIVERGENCE` | warning | recent-window Sharpe falls `divergence_threshold` below the preceding equal-length window — the early-warning signal |
| `NEW_CRITICAL_FLAG` | critical | a critical integrity flag (e.g. smoothing, short-volatility signature) appears that was absent in the prior snapshot |
| `GRADE_DROP` | warning | the authoritative letter grade falls |
| `SCORE_DROP` | warning | the expanding score falls more than `score_drop_threshold` versus the prior snapshot |

`TRAILING_DIVERGENCE` is the pre-emptive signal: it fires on the recent
window before the slow-moving expanding score and grade react. The other
three confirm deterioration once it reaches the headline number, and the
critical-flag transition catches manipulation that emerges over time.

## Auditable history

Every observation appends a `ScoreSnapshot` to an append-only,
hash-chained `ScoreHistory`. Each snapshot commits to the previous snapshot's
hash, so the sequence cannot be reordered, inserted into, or back-edited
without breaking the chain — the monitoring analogue of the artifact content
hash. `ScoreHistory.is_chain_valid()` verifies the whole chain end to end,
and each snapshot's expanding score remains independently verifiable via the
firewall. This is the tamper-evident track-record history an investment
committee can rely on.

```python
from quantlite.score import Monitor, Cadence, DataSource

monitor = Monitor(
    cadence=Cadence.DAILY,
    source=DataSource.EXCHANGE_CUSTODY,
    attester="exchange:quantmarket",
    account_ref="acct-7a3f",
    trailing_window=126,          # ~6 months; default is half a year
    divergence_threshold=0.75,    # annualised-Sharpe drop that warns
)

for returns, as_of in venue_periods:        # independently-sourced fills
    for alert in monitor.observe(returns, as_of=as_of):
        notify(alert)                        # webhook / dashboard / queue

assert monitor.history.is_chain_valid()
```

## Calibration is a partner decision

The thresholds — `trailing_window`, `divergence_threshold`,
`score_drop_threshold` — are deliberately parameters, not constants. Scores
move between observations for legitimate reasons, and the line between
"normal variation" and "deteriorating" depends on the venue's risk appetite
and the strategies it lists. Default values are sensible starting points;
the operator tunes them against their own roster and alert tolerance.

## Delivery is a separate layer

The library emits alerts and histories as plain, verifiable data structures.
It does not deliver them. Webhooks, dashboards, and polling APIs are adapters
in the hosted layer that wrap `Alert.to_dict()` and `ScoreHistory.to_json()`,
so every delivery channel stays available without baking transport choices
into the core.
