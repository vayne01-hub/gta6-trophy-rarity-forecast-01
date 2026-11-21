# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vayne01-hub/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vayne01-hub/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Python 3.13 program (standard library only) that forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the five quantiles (5/25/50/75/95%) of its earn rate on Exophase at the case's target. It works in logit space and models earners and players-tracked separately: a rate is earners / players-tracked, and the denominator grows fast in week one, which drags rates down even while earners rise. Each is fit log-linearly against ln(1 + hours since release) over the last 24 hours of distinct reads — the log-time axis damps the 4.5-day extrapolation — and the rate is read off their difference at the target. Piecewise-constant reads are collapsed to distinct refresh states, so the fit follows refreshes instead of chasing every 10-minute step. Growth slopes are shrunk toward the mean slope of the trophy's last-rate band (with near-ceiling trophies pooled apart: at 90% a 1-point move is ~0.1 logit, at 1% it is ~1 logit). A trophy still pinned at ~100% is diluted by forecast tracked-player growth instead (new players earn the prologue quickly but not instantly). Persistence and a plain logit-linear fit are kept as rival variants, and the program backtests all three on the case's own history — fitting prefixes at 25/50/75% of the span and forecasting the last read — picking the variant and spread multiplier with the lowest backtest pinball loss (a variant within 5% of the best counts as tied, since backtest horizons are far shorter than the real one). The spread is a random walk plus fit residual plus the median refresh jump, floored so a degenerate backtest on constant history can never collapse it to zero. Deterministic, offline, in well under a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py # reads every <id>.in in the input dir, writes <id>.out per case
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
