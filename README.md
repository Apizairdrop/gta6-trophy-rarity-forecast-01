# GTA VI Trophy Rarity Forecast — Week One

[![Competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Apizairdrop/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Apizairdrop/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A forecaster for the [make.fun competition `gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): for every PlayStation 5 trophy of Grand Theft Auto VI, forecast its earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of tracked players who will hold it one week after launch — as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How it forecasts

`main.py` (Python 3.13, standard library only, no network, deterministic) reads every `*.in` case in the input directory and writes one `<id>.out` per case. For each trophy it combines three views of the history:

1. **Trend.** A least-squares line of logit(rate) against log(1 + hours since release), fitted over the last 24 hours of reads, anchored at the last read's rate and read at the target.
2. **Counts, not just rates.** Earners = rate × players_tracked / 100. A second fit extrapolates earners and players separately and divides — so the model sees rates falling while earners grow when the tracked-player base grows faster than earning.
3. **Tier priors.** The fitted slope is shrunk toward a prior per tier (story bronze climbs fastest in week one, then other bronze, silver, gold, platinum barely moves). Flat history carries no slope information, so flat data is down-weighted and the prior does the work; live data overrides it.

**Refresh dilution.** Exophase refreshes on its own schedule: launch night sat at 638 tracked players for hours, then one refresh jumped to 3,868 and every rate but the 100% one fell. The arithmetic is stark — the trophy already at 100% absorbed 3,230 of the 3,333 new earners, so the other 54 trophies' rates were multiplied by about 0.22 on average. The model measures staleness (the share of consecutive reads with an unchanged players-tracked count) and, when the page is stale, shifts non-trivial medians down by log(0.25) in logit, fading with horizon; the ~100% trophy gets no shift. When the page refreshes normally, no shift is applied.

**Spread.** The logit spread is the root-sum-square of a random-walk term (0.5 × √horizon in days), the trend fit's residual, a refresh-shock term, and a 0.30 floor — so the 90% band widens honestly with horizon and with how stale the data is.

## Local validation

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

On the kit's practice cases (p00, p01): valid output (55 rows, non-decreasing quantiles, 0–100, ~2.8 KB each, well under the 65,536-byte cap), byte-identical across two runs, ~0.1 s for both cases (limit 8 s per case).

In-history holdout backtest (hold out the last 1–2 reads of each practice case, forecast them from earlier reads): mean pinball loss **0.708** vs the starter baseline's **0.014** on the same holdouts. That gap is expected and honest: the held-out reads are byte-identical to the history (the page had not refreshed), so the narrowest spread wins a holdout that contains no refresh. The model pays for refresh insurance there — and the insurance is what the real practice punishes: on a synthetic refresh built from the observed aggregate (non-trivial rates × 0.22, the 100% trophy unchanged), the model scores **0.22** with 0.98 coverage against the starter's **1.40** with 0.00 coverage (the house baseline scores ~1.51 on the real practice).

## Files

- `main.py` — the forecaster
- `README.md` — this file
- `LICENSE` — MIT

## License

MIT. See `LICENSE`.
