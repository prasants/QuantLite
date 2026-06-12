# Commercial Strategy

> **Allocators now, managers later, firewall always.**

## The asset

The QuantLite library is distribution; the **QuantLite Score** is the
business. A reproducible, independently-attestable rating of a trading
track record is trust infrastructure for a transaction where both sides
have capital at risk and neither trusts the other. Standards are where
pricing power lives, and a score that the rated party cannot buy a better
version of can become a standard.

## Why not both sides at once

Selling to allocators (who want the truth) and to managers (who want a good
grade) at the same time is not two revenue streams — it is one corrupted
product:

1. **Incentive conflict destroys the asset.** If managers are the paying
   customers, allocators rationally discount the score as issuer-funded
   marketing. This is exactly how the issuer-pays credit-rating model
   failed. The score is only worth money while it cannot be bought.
2. **They are two different companies.** Allocator-side is enterprise: few
   logos, high ACV, integration-heavy, DD-driven sales. Manager-side is
   self-serve: many small accounts, marketing-led, badge-and-onboarding.
   Different product, motion, and org. Early stage cannot split focus.
3. **Sequence is the unlock.** Manager willingness-to-pay is small until a
   score is *required*. It becomes large only after allocators have made the
   score a gate. Do it in the right order and the manager money is big and
   safe; do it first and the well is poisoned before the water is worth
   drinking.

So: **allocators first, managers second, and a firewall — enforced in code,
not policy — that makes the score independent of who pays, always.**

## Beachhead

**Marketplace operators where the venue is the custodian of record** — a
crypto exchange building a quant marketplace is the ideal first user. The
exchange owns the ground-truth fills, so the firewall is the cleanest
possible: managers on the marketplace cannot fabricate the record at source
(see [`docs/firewall.md`](docs/firewall.md)). One platform integration scores
its entire roster of strategies at once, and the platform bears the
reputational and fraud risk that makes it want the score.

Adjacent allocator segments, in priority order:

1. **Managed-account / manager marketplaces** (beachhead) — roster scale,
   real fraud exposure, integration budget.
2. **Fund-of-funds / OCIO / family offices** — high willingness to pay, slow
   bespoke DD; best as design partners, not first revenue.
3. **Seed / emerging-manager allocators** — highest fraud exposure, smallest
   cheques.

## Sequence

1. **Firewall in code first.** The score must be *provably* independent of
   who pays before anything is sold. Shipped: `score.provenance` —
   `SourceAttestation`, `AttestedScore`, `assert_firewall`, and a five-check
   `FirewallReport`. This is both the credibility document and the moat:
   the scoring formula is copyable; the independently-sourced attestation
   chain is not.
2. **Design partners, not customers.** Land 3–5 marketplace/allocator design
   partners (the crypto exchange is partner zero). They provide real data to
   validate the score against outcomes and a written "if this works, here is
   what we would require and pay."
3. **Let partners define the must-have.** Expected to be *continuous
   monitoring with an auditable score history* — alert when a record starts
   showing the smoothing / short-volatility signature over time — not the
   one-shot score. The static score is the demo; monitoring is the product.
4. **Price off the partners.** Enterprise license by roster size or AUM
   scored, mid-five to low-six figures, annual. ~15–25 logos for the first
   few million in ARR — a knowable, winnable list.
5. **Managers only after a partner requires the score.** The trigger to build
   the manager tier is the first allocator who says "show me your QLS."
   Manager monetisation is always for *verification and distribution* — the
   badge, the listing, the attestation — never for the grade, which is
   computed identically whether the manager pays or not.

## The firewall invariant (non-negotiable)

Paying more can never produce a better number. The score is a pure function
of independently-sourced inputs; the only thing money buys is verification,
attestation, and distribution. If a feature would let a manager improve
their grade by paying, it does not ship. This invariant is what keeps the
entire business worth money.
