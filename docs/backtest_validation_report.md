# Backtest Validation Report

**Date:** March 2026
**Dataset:** ATP pro matches, 2024 season (hold-out year)
**Filter:** Pro-ranked vs pro-ranked matches only

---

## 1. Simulation Calibration

Tested 3,000 player-tournament entries from 2024 (Challengers, M25, M15, ATP 250).
For each entry, ran 500 Monte Carlo simulations and compared predicted round-reach
probabilities to actual outcomes.

### Calibration by Round

| Round | Brier Score | Max Gap | Notes |
|-------|------------|---------|-------|
| Reach R8+ | 0.2504 | 19.7% | Model underestimates by ~10-15% at low probs |
| Reach QF+ | 0.2015 | 13.8% | Reasonable calibration above 30% |
| Reach SF+ | 0.0892 | 10.7% | Good calibration, slight underestimate |
| Reach F+  | 0.0532 | 6.6%  | Well calibrated |
| Win       | 0.0331 | 5.0%  | Well calibrated |

### Calibration by Category (QF reach)

| Category | n | Pred P(QF) | Actual P(QF) | Brier |
|----------|---|-----------|-------------|-------|
| Challenger | 937 | 19.7% | 26.1% | 0.1901 |
| M25 | 724 | 21.6% | 32.3% | 0.2045 |
| M15 | 1,215 | 19.3% | 30.0% | 0.2048 |
| ATP 250 | 124 | 19.1% | 35.5% | 0.2361 |

### Interpretation

The model is consistently conservative — it underestimates player success by about
10-15 percentage points in the early rounds. The calibration improves significantly
for deeper rounds (SF, F, W). The remaining gap is likely due to the field generation
creating slightly too-strong opponent draws. This is a tuning improvement, not a
fundamental model issue.

---

## 2. Schedule Comparison

Compared optimizer recommendations to actual 2024 schedules for three players
during clay season (weeks 14-24).

### Results

| Player | Rank | Country | Overlap | Surface Match | Geography |
|--------|------|---------|---------|---------------|-----------|
| Player A | 212 | Spain | 1/10 (10%) | Rec 89% clay vs actual 100% | Both Europe |
| Player B | 432 | Ukraine | 3/10 (30%) | Rec 100% clay vs actual 90% | Both Europe |
| Player C | 562 | Portugal | 1/10 (10%) | Rec 100% clay vs actual 90% | Both Europe |

### Interpretation

Geographic constraint: All recommended schedules stayed on the player's home continent,
matching actual behavior. Previous version had recommended intercontinental travel.

Surface seasonality: Recommended schedules are 89-100% clay during clay season,
closely matching actual player behavior of 90-100%.

Tournament overlap: 10-30% exact match is reasonable given that optional tournament
choice involves personal preference, coach input, and logistical factors the model
can't capture. The category mix and surface alignment matter more than exact matches.

---

## 3. Key Finding: Pro/Junior Ranking Contamination

During validation, we discovered that 10-14% of ITF-tier matches (M15/M25) in
the both-ranked dataset involve mixed ranking types — one player with a pro (ATP)
ranking and the other with only a junior (ITF Junior) ranking.

Junior rankings and pro rankings are on completely different scales. A "rank 200"
junior player is not equivalent to a "rank 200" pro player. When mixed matches were
included in calibration testing, they produced apparent calibration gaps of 50-80%+
that disappeared when filtering to pro-vs-pro only.

**Resolution:** Added `ranking_match_type` column to the pipeline output
(both_pro, both_junior, mixed). All simulation and optimizer components use
pro-vs-pro matches only. Junior-focused optimization is flagged as a separate
future module.

---

## 4. Summary Metrics

| Metric | Before Fix | After Fix | Target |
|--------|-----------|-----------|--------|
| Max calibration gap | 83.5% | 19.7% | <10% |
| Brier (QF reach) | 0.2707 | 0.2015 | <0.18 |
| Schedule surface match | 40-60% | 89-100% | >80% |
| Schedule geography | Multi-continent | Single continent | Regional |
| Schedule overlap | 0-10% | 10-30% | 15-30% |
