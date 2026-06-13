# The QuantLite Firewall

A rating is worth money only when the rated party cannot buy a better one.
The moment a manager can influence the inputs that produce their own score,
the score drifts from *measurement* toward *marketing* — the failure mode
that hollowed out credit ratings in 2008. The firewall is QuantLite's
structural defence: a score is **firewall-clean** only when its inputs are
attested to a source the rated manager does not control.

This document specifies the firewall and how an allocator or marketplace
operator verifies it. It is also the credibility argument: everything here
is a property a third party can check, not a promise we ask them to trust.

## The two questions a score must answer

`quantlite.score` already answers the first; the firewall answers the second.

| Question | Mechanism | Module |
| --- | --- | --- |
| *What were the returns?* | SHA-256 `input_digest` over canonicalised returns; reproducible `ScoreArtifact` content hash | `score.artifact` |
| *Where did the returns come from?* | `SourceAttestation` binding that digest to an independent origin and attester | `score.provenance` |

A score with a perfect artifact but no independent provenance is
reproducible *and* worthless: it faithfully reproduces whatever the manager
chose to submit.

## Independent vs. controlled sources

`DataSource` enumerates origins by their independence from the rated party:

- **Independent** (firewall-clean): `EXCHANGE_CUSTODY`, `FUND_ADMINISTRATOR`,
  `PRIME_BROKER`, `ALLOCATOR_SUPPLIED`. The data is produced and vouched for
  by someone other than the manager.
- **Controlled**: `MANAGER_SUBMITTED`. The rated party supplied the inputs.
  A score built on this can never be firewall-clean, regardless of how
  cleanly it reproduces.

`is_independent(source)` is the single classification rule; everything else
derives from it.

## The attestation

A `SourceAttestation` is a sealed record asserting that a specific return
series, identified by its `input_digest`, was obtained from a named source
over a named period by a named attester:

```python
from quantlite.score import (
    compute_score, DataSource, SourceAttestation, AttestedScore,
)

result = compute_score(returns, n_trials=20)

attestation = SourceAttestation.create(
    source=DataSource.EXCHANGE_CUSTODY,
    account_ref="acct-7a3f",
    period_start="2024-01-01",
    period_end="2024-12-31",
    ingested_at="2025-01-04T09:00:00Z",
    returns=returns,
    attester="exchange:quantmarket",
)

attested = AttestedScore(artifact=result.artifact, attestation=attestation)
report = attested.verify(returns)
assert report.ok and report.firewall_clean
```

The attestation carries its own SHA-256 `attestation_hash` over every field
but the hash itself, following the same QLS-1.0 canonicalisation rules as
the score artifact (sorted keys, compact separators). It cannot be silently
edited, and re-serialising the same facts yields the same hash.

## Verification: one pass, five checks

`AttestedScore.verify(returns)` returns a `FirewallReport`. `report.ok` is
true only when all five hold:

1. **`artifact_consistent`** — the score artifact's content hash matches its
   fields (untampered score).
2. **`attestation_consistent`** — the attestation's hash matches its fields
   (untampered provenance).
3. **`digests_match`** — attestation, artifact, and the supplied returns
   share one `input_digest` (the score, the provenance, and the data in hand
   are all about the *same* series).
4. **`score_reproduces`** — recomputing from the returns reproduces the
   artifact bit for bit.
5. **`firewall_clean`** — the attestation names an independent source.

A `MANAGER_SUBMITTED` score can still pass checks 1–4: it is a real,
reproducible score. It simply fails check 5, and therefore `ok`. This is
deliberate — the report distinguishes *"the number is fake"* from *"the
number is real but the provenance is not independent."* `assert_firewall()`
collapses the provenance checks into a hard gate that raises `FirewallError`.

## Why this fits a marketplace operator

When the venue is the custodian of record — a crypto exchange scoring the
strategies listed on its own quant marketplace — the firewall is the
cleanest it can possibly be. Returns are derived from venue-side fills the
manager never touches, so the record cannot be smoothed, back-painted, or
cherry-picked *at source*. The exchange is both the `attester` and the
`EXCHANGE_CUSTODY` source; managers initiate verification but never supply
the data that grades them.

Integration shape for that case:

1. The marketplace computes returns from its own fill/settlement data for a
   strategy account over a period.
2. It calls `SourceAttestation.create(source=EXCHANGE_CUSTODY, ...)` and
   `compute_score(...)`, then publishes the `AttestedScore`.
3. Anyone — the allocator, the manager, a regulator — runs
   `verify(returns)` and gets the same five-check verdict. No trust in the
   exchange's word is required; the data substantiates itself.

## What the firewall is not

- It is **not** a signature/PKI scheme. The attestation proves *integrity*
  (the facts have not changed) and *independence* (the source is not the
  manager); binding an attester's real-world identity to a key is a separate
  layer that sits on top of `attester`.
- It does **not** judge whether the source itself is honest. It guarantees
  the rated manager did not control the inputs; vetting attesters is an
  operational concern, not a cryptographic one.
- It does **not** weaken the score for controlled sources — it refuses to
  certify them. The number is still computed; it is simply not firewall-clean.
