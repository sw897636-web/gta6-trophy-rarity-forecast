# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sw897636-web/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sw897636-web/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts each PlayStation 5 trophy of Grand Theft Auto VI's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26 00:00 UTC, as five quantiles per trophy, for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Method

`main.py` (Python 3.13, standard library only) reads one case from
`/input/<id>.in` and writes `/output/<id>.out`. For each trophy it fits a
weighted least-squares line of logit(earn rate) on log time since release
over the case's distinct reads (consecutive identical reads collapse to one
epoch, so Exophase's refresh gaps don't outweigh real movement), reads the
line at the target, and blends it slightly back towards the last read.
Trophies with no movement are forecast at their last rate.

The predictive spread on the logit scale combines a random-walk term that
grows with the horizon, the fit's residual scatter, and a refresh-jump term
(Exophase refreshes its counts in jumps), scaled to match the fitted
median's error. Quantiles are median ± z × spread on the logit scale.

Deterministic: two runs on the same case write identical bytes.

## Local check

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

On the kit's practice cases the program runs in ~0.07 s, writes valid
forecasts (55 trophies, monotone quantiles in (0, 100)), and is
byte-identical across runs. On two synthetic final-like cases (two days of
logistic trophy growth with Exophase-style refresh steps, scored against
the known day-7 rates) it scored a pinball loss of 0.22 / 0.26 against the
house persistence baseline's 0.32 / 0.36.
