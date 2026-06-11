# The QuantLite Score

**An open, versioned, verifiable rating for trading track records.**

Current methodology version: **QLS-1.0**.

---

## Why a score

Raw performance numbers are trivially gamed. Test fifty strategy variants and publish the winner. Start the equity curve at the bottom of a drawdown. Smooth the marks. Sell tail risk and post a 92% win rate until the one month that erases three years.

Marketplaces, prop firms, and allocators rank traders on exactly these numbers. The QuantLite Score replaces them with a single 0&ndash;100 rating built from the statistics that are hard to game: the Deflated Sharpe Ratio, bootstrap robustness, tail risk penalties, and integrity checks that flag the classic manipulation patterns.

The methodology is open. Anyone can read this document, run the reference implementation, and reproduce any published score bit for bit. Trust comes from reproducibility, not authority.

```python
from quantlite.score import compute_score, verify_artifact

result = compute_score(returns, n_trials=20)
print(result.score, result.grade)         # e.g. 72.4 B

payload = result.artifact.to_json()        # publish or store this
assert verify_artifact(payload, returns)   # anyone can check it
```

---

## Inputs

| Input | Description | Default |
|---|---|---|
| `returns` | Simple periodic returns of the track record | required |
| `n_trials` | Number of strategy variants tried before this record was selected | 1 |
| `freq` | Periods per year | 252 |
| `n_bootstrap` | Stationary bootstrap samples for the robustness component | 1000 |
| `seed` | Random seed for all stochastic steps | 1729 |

`n_trials` matters. The skill component deflates the Sharpe ratio by the number of strategies tested; self-reporting `n_trials=1` when fifty were tried inflates the score. Verification services should require trial logging (see `quantlite.overfit.TrialTracker`) rather than trusting the declared value.

A minimum of 30 observations is required; fewer raises an error rather than producing a meaningless number.

---

## Components (QLS-1.0)

The composite score is a weighted sum of five components, each in [0, 100].

| Component | Weight | What it measures |
|---|---|---|
| Skill | 35% | Deflated Sharpe Ratio: probability the observed Sharpe is genuine given how many strategies were tried, adjusted for skewness and kurtosis. |
| Robustness | 20% | 5th percentile of the bootstrapped annualised Sharpe distribution (stationary bootstrap, block length &radic;n), mapped through a logistic squash `100 / (1 + e^(-1.2x))`. An edge that survives resampling, not a point estimate. |
| Tail | 20% | Blend of maximum drawdown (60%) and CVaR at 95% (40%). Drawdowns score zero at &minus;50%; CVaR scores zero at 0.80 in annualised units (&#124;CVaR&#124; &times; &radic;freq). |
| Consistency | 15% | Fraction of rolling quarter-length windows (stepped by a third of a window) with positive mean return. |
| Sufficiency | 10% | Observations relative to the Minimum Track Record Length for the observed Sharpe, floored at one year. A non-positive Sharpe scores zero. |

All calibration constants are frozen for the lifetime of a methodology version and exposed as constants in `quantlite.score.engine`.

### Grades

| Score | Grade |
|---|---|
| &ge; 90 | A+ |
| &ge; 80 | A |
| &ge; 65 | B |
| &ge; 50 | C |
| &ge; 35 | D |
| &lt; 35 | F |

---

## Integrity checks

Before scoring, the track record passes through `validate_track_record`. Flags do not reject the record; they cap the composite score and travel with the artifact so a capped score is always explainable.

| Flag | Severity | Trigger |
|---|---|---|
| `non_finite_values` | critical | NaN or infinite returns |
| `zero_variance` | critical | Constant returns |
| `short_record` | critical | Fewer than half a year of observations |
| `excessive_smoothing` | warning / critical | Lag-1 autocorrelation above 0.30 / 0.50 &mdash; the signature of smoothed or stale marks |
| `outlier_dependence` | warning | Top 5 days contribute more than half of all positive log returns |
| `short_volatility_signature` | warning | Win rate above 85% with skewness below &minus;1 &mdash; the pattern of selling tail risk |
| `start_date_sensitivity` | warning | Dropping the first 10% of observations halves the Sharpe &mdash; a cherry-picked start date |

Any critical flag caps the composite at **40** (at best a D). Any warning caps it at **70** (at best a B).

---

## The artifact

Every score ships as a `ScoreArtifact`: a portable JSON record containing the methodology version, library version, every scoring parameter, a SHA-256 digest of the canonicalised input returns, the component scores, the integrity flags, and a content hash.

Canonicalisation rules:

- Input returns are cast to little-endian float64 before hashing, so the digest is identical across platforms.
- Artifact JSON uses sorted keys and compact separators.
- The content hash is SHA-256 over the canonical JSON of every field except `created_at` and the hash itself, so re-issuing the same score at a different time produces the same hash.

Verification recomputes everything:

```python
from quantlite.score import verify_artifact

verify_artifact(artifact_json, returns)  # True only if authentic
```

`verify_artifact` confirms that the artifact's hash matches its fields, that the supplied returns match the recorded input digest, and that recomputing the score with the recorded parameters reproduces the score, the components, and the content hash exactly. Tamper with any field &mdash; the score, the grade, a flag &mdash; and verification fails.

Verification is bit-exact within a library version. Methodology versions are frozen: QLS-1.0 results will never change under a QLS-1.0 implementation.

---

## Determinism guarantees

Same returns, same parameters, same library version: same score, same artifact content hash. Every stochastic step (the bootstrap) is driven by the recorded seed. There is no wall-clock, locale, or platform dependence in the scored fields.

---

## Versioning policy

- The methodology version (`QLS-x.y`) is recorded in every artifact.
- Calibration constants, weights, thresholds, and canonicalisation rules are frozen within a methodology version.
- Changes to any of these require a new methodology version. Scores from different methodology versions are not comparable and must not be displayed on the same scale without the version label.

---

## What the score is not

- **Not a prediction.** It rates the statistical credibility and risk shape of a realised track record, not future returns.
- **Not unfakeable on self-reported inputs.** `n_trials` and the returns themselves must come from a trusted source (a custodian, an exchange, a verified platform feed) for the score to mean anything. The score makes honest evaluation easy; it cannot make dishonest inputs honest.
- **Not a substitute for due diligence.** It is a screen and a common language, in the way a credit score is.
