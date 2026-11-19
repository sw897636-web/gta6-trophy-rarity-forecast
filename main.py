"""Forecast GTA VI's PS5 trophy earn rates at 2026-11-26 (make.fun bounty
gta6-trophy-rarity-forecast-01).

Method (computed at run time from the case's own history; nothing about the
shape of a launch is hard-coded):

Exophase's earn rate for a trophy is the share of tracked players who hold
it.  Two forces move it: players earn the trophy (numerator grows) and new
players sync profiles (denominator grows, diluting every rate).  On the
logit scale a trophy's rate grows roughly linearly in log time since
release -- fast early growth that saturates -- so for each trophy we fit a
weighted least-squares line of logit(rate) on ln(1 + hours since release)
over the case's distinct reads (consecutive identical reads collapse to
one epoch, so long flat stretches from Exophase's own refresh gaps do not
outweigh real movement), weight recent reads more, shrink the slope a
little and blend the fit back towards the last read, then read the line at
the target.  A trophy with no movement in its history is forecast at its
last rate.

Spread: the predictive logit spread combines three honest sources of
uncertainty -- a random-walk term that grows with the horizon (the house
baseline's 0.5 * sqrt(days)), the fit's residual scatter, and a
refresh-jump term (Exophase refreshes its counts in jumps, so the typical
logit jump between distinct reads, scaled by how many refreshes the
horizon should hold).  Flat histories get a wide spread, because a
refresh the history never saw is exactly what the practice cases punish.
Quantiles are median +/- z_tau * spread on the logit scale, with the
standard normal quantiles, rounded to 4 decimals.

Standard library only, deterministic, no network, no file access outside
the input/output directories.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# How much to shrink the fitted log-time slope towards zero (guards against
# a short, noisy history extrapolating wildly over a 4.5-day horizon).
SLOPE_SHRINK = 1.0
# Blend of the fitted line vs the last read for the median (0 = fit only).
PERSIST_BLEND = 0.15
# Random-walk spread, logit units per square-root day.
RW_SIGMA = 0.5
# Scale on the total predictive spread: the unscaled spread covered
# every trophy on both synthetic finals, i.e. it was wider than the
# fitted median's error; 0.8 keeps an honest band without paying
# for width at every quantile.
SPREAD_SCALE = 0.8
# Spread of a flat-history trophy: the dynamics are unobserved, so stay wide.
FLAT_RESID = 0.6
# Recency half-life for the fit's weights, in hours.
HALF_LIFE_HOURS = 24.0
# Spread bounds, in logit units.
SPREAD_MIN = 0.35
SPREAD_MAX = 2.0
# Median is capped here: a trophy printed at 100% still dilutes as new
# players join, so never forecast essentially-certain 100%.
MEDIAN_CAP = 99.0


def parse_time(text):
    """Parse an ISO 8601 timestamp (trailing Z allowed) to an aware datetime."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def distinct_epochs(history, n_trophies):
    """Collapse consecutive reads with identical rates and player counts.

    Exophase refreshes its counts hours apart, so the recorder's 10-minute
    reads repeat the same page many times.  One epoch per distinct page
    keeps the fit from overweighting flat stretches.  Returns a list of
    (hours_since_release, players_tracked, rates) oldest first.
    """
    epochs = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        rates = list(rates) + [None] * (n_trophies - len(rates))
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        at = parse_time(read.get("at"))
        if at is None:
            continue
        hours = (at - RELEASE).total_seconds() / 3600.0
        epochs.append((hours, read.get("players_tracked"), rates))
    return epochs


def wls(points):
    """Weighted least squares of y on x.  points are (x, y, w).

    Returns (intercept, slope, residual_std) or None when the fit is
    degenerate (fewer than 3 points or no spread in x).
    """
    if len(points) < 3:
        return None
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    slope = sxy / sxx
    intercept = my - slope * mx
    dof = max(len(points) - 2, 1)
    resid2 = sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in points)
    resid = math.sqrt(resid2 / (sw * dof / len(points))) if sw else 0.0
    return intercept, slope, resid


def forecast_trophy(series, target_hours, horizon_days):
    """Forecast one trophy.

    series: [(hours_since_release, rate_pct)] oldest first, distinct epochs
    only.  Returns (median_pct, spread_logit).
    """
    last_rate = series[-1][1] if series else None
    if last_rate is None:
        return 1.0, 1.5
    last_logit = logit(last_rate)

    # Typical refresh jump, from consecutive distinct reads.
    jumps = []
    for (t0, r0), (t1, r1) in zip(series, series[1:]):
        jumps.append(abs(logit(r1) - logit(r0)))
    gaps = [t1 - t0 for (t0, _), (t1, _) in zip(series, series[1:])]
    typical_jump = median(jumps) if jumps else 0.5
    typical_jump = min(typical_jump, 0.3)
    gap_hours = median(gaps) if gaps else 4.0
    if gap_hours <= 0:
        gap_hours = 4.0
    n_refresh = min(horizon_days * 24.0 / gap_hours, 8.0)
    jump_sig = math.sqrt(n_refresh) * typical_jump

    rw = RW_SIGMA * math.sqrt(horizon_days)

    rates = [r for _, r in series]
    if len(series) < 3 or min(rates) == max(rates):
        # No movement to fit: stay at the last rate, spread wide.
        spread = math.sqrt(rw * rw + FLAT_RESID ** 2 + jump_sig ** 2)
        return min(last_rate, MEDIAN_CAP), min(max(spread * SPREAD_SCALE, SPREAD_MIN), SPREAD_MAX)

    # Fit on the last 48 hours of distinct reads (all of them if fewer).
    t_last = series[-1][0]
    window = [(t, r) for t, r in series if t >= t_last - 48.0]
    if len(window) < 3:
        window = series
    points = []
    for t, r in window:
        x = math.log1p(max(t, 0.0))
        w = 0.5 ** ((t_last - t) / HALF_LIFE_HOURS)
        points.append((x, logit(r), w))
    fit = wls(points)
    if fit is None:
        spread = math.sqrt(rw * rw + FLAT_RESID ** 2 + jump_sig ** 2)
        return min(last_rate, MEDIAN_CAP), min(max(spread * SPREAD_SCALE, SPREAD_MIN), SPREAD_MAX)

    intercept, slope, resid = fit
    slope *= SLOPE_SHRINK
    x_target = math.log1p(max(target_hours, 0.0))
    fit_logit = intercept + slope * x_target
    med_logit = (1.0 - PERSIST_BLEND) * fit_logit + PERSIST_BLEND * last_logit
    med = inv_logit(med_logit)
    med = min(max(med, 0.01), MEDIAN_CAP)

    resid = min(max(resid, 0.05), 1.0)
    spread = math.sqrt(rw * rw + resid * resid + jump_sig * jump_sig)
    return med, min(max(spread * SPREAD_SCALE, SPREAD_MIN), SPREAD_MAX)


def forecast_case(case, name):
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    epochs = distinct_epochs(case["history"], n)
    if not epochs:
        raise ValueError(f"{name} has no readable history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case.get("target"))
    checkpoint = parse_time(case.get("at"))
    if target is not None:
        target_hours = (target - RELEASE).total_seconds() / 3600.0
    elif checkpoint is not None:
        target_hours = (checkpoint - RELEASE).total_seconds() / 3600.0 + case["horizon_minutes"] / 60.0
    else:
        target_hours = epochs[-1][0] + case["horizon_minutes"] / 60.0

    rows = []
    for j, trophy in enumerate(trophies):
        series = [(t, rates[j]) for t, _, rates in epochs if rates[j] is not None]
        med, spread = forecast_trophy(series, target_hours, horizon_days)
        med_logit = logit(med)
        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(med_logit + z * spread))
            q = min(max(q, 0.01), 100.0)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
