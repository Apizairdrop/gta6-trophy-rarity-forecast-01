"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at the case target -- five quantiles per trophy, scored by
pinball loss on the logit scale.  Python 3.13, standard library only, no
network, no randomness: two runs on the same case write byte-identical files.

How it forecasts
---------------
The history is a sequence of reads of one page.  Each read gives every
trophy's earn rate and the number of players Exophase tracks.  Three things
in that history matter, and the model uses all three:

1. Trend.  Per trophy, a least-squares line of logit(rate) against
   log(1 + hours since release), fitted over the last 24 hours of reads and
   anchored at the last read's rate, then read at the target.  Anchoring at
   the last rate keeps the level honest; the line only supplies direction.

2. Counts, not just rates.  Earners = rate * players_tracked / 100.  A
   second line fits log(earners) and a third fits log(players); the rate
   implied by extrapolating both is averaged (in logit space) with the
   trend line's rate.  When the tracked-player base is growing fast, rates
   can fall while earners grow -- the counts view sees that, a rate-only
   line does not.

3. Tier priors.  A flat history carries no slope information (launch
   night's reads are byte-identical: the page had not refreshed).  So the
   fitted slope is shrunk toward a prior slope per tier: story bronze
   (chapter completions) climbs fastest in week one, other bronze next,
   silver and gold slower, platinum barely moves.  The shrinkage weight
   grows with the number of reads and with how much the history actually
   varied -- flat data lets the prior do the work, live data overrides it.

Refresh dilution.  Exophase refreshes its counts on its own schedule, not
continuously: launch night sat at 638 tracked players for hours, then one
refresh jumped to 3,868 and every rate but the 100% one fell (newly
tracked players hold fewer trophies).  The arithmetic of that refresh is
stark: trophies earned grew 943 -> 4,276, but the trophy already at 100%
absorbed 3,230 of the 3,333 new earners, so the other 54 trophies shared
only ~100 new earners between them -- their rates were multiplied by
about 0.22 on average.  The model measures staleness -- the share of
consecutive read pairs with an unchanged players-tracked count -- and,
when the page is stale at the checkpoint, shifts every non-trivial
median down by log(0.25) in logit, scaled by staleness and by a horizon
factor (an overdue refresh reprices the level, then days of earning
rebuild part of it, so the shift fades from ~full at a 90-minute horizon
to ~0.37 at 4.5 days).  The trophy at ~100% gets no shift: the evidence
is that it stayed at 100% through the refresh.  When the page has been
refreshing normally, staleness is near zero and no shift is applied.

Spread.  The logit spread is the root-sum-square of a random-walk term
(0.5 * sqrt(horizon in days)), the trend fit's residual, a refresh-shock
term (0.55 * staleness) and a 0.30 floor for model error, so the 90%
band widens honestly with horizon and with how stale the data is.

Run as:  python3 -P -s main.py
with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set, else /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a helper module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# GTA VI release day, the origin of the log-time axis.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def clamp_pct(x: float) -> float:
    """A percent into the range the logit can take (0 and 100 have none)."""
    return min(99.99, max(0.01, x))


def logit_of(pct: float) -> float:
    """logit of a percent: ln(p / (1 - p)) with p = pct / 100."""
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of(lg: float) -> float:
    """Inverse logit back to a percent."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, halves away from zero (matches the house's
    JavaScript Math.round(x * 10000) / 10000); whole numbers come back
    as ints so the JSON writes 100 rather than 100.0."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_ts(text: str) -> datetime:
    """An ISO 8601 timestamp (with a trailing Z) into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment: datetime) -> float:
    """Hours from the release-day origin to a read; never negative."""
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def log_time(moment: datetime) -> float:
    """The trend axis: log(1 + hours since release).  Growth in players
    and in trophy earning is fast at first and slows, which log-time
    captures better than raw hours."""
    return math.log1p(hours_since_release(moment))


def ols_fit(xs, ys):
    """Least-squares line through (xs, ys).  Returns (slope, intercept,
    residual_std, n).  A flat or single-point series gives slope 0 and
    residual 0 rather than raising."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return 0.0, ys[0], 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return slope, intercept, resid, n

# --- Tier priors: expected logit drift per unit of log-time -----------------
# A trophy's earn rate grows fastest for the things every player meets
# early (story chapters), slower for side content, slower still for the
# long grinds, and barely at all for the platinum, which needs everything.
# These are shrinkage targets for the fitted slope, not the forecast
# itself: live data overrides them, flat data leans on them.
PRIOR_SLOPE = {
    "platinum": 0.10,
    "gold": 0.35,
    "silver": 0.55,
    "bronze": 0.60,
}
STORY_BRONZE_SLOPE = 0.90
# How strongly the prior pulls: worth 3 reads of data.
PRIOR_WEIGHT = 3.0
# A flat history is uninformative about slope (the page simply had not
# refreshed), so flat data counts at a quarter of its size.
FLAT_DATA_WEIGHT = 0.25
FLAT_RESID = 0.05

# The logit shift an overdue refresh causes on a non-trivial trophy:
# log(0.25), the ~0.22 average multiplier the first refresh produced
# (see the module docstring).  The horizon factor fades it with horizon:
# ~0.98 at 90 minutes, ~0.37 at 4.5 days.
REFRESH_LOG_FACTOR = -1.3862943611198906  # log(0.25)
REFRESH_SIGMA = 0.80
# Random-walk spread per square-root day, and the floor for model error.
WALK_SIGMA = 0.5
FLOOR_SIGMA = 0.30


def prior_slope(trophy: dict) -> float:
    """The tier prior slope for one trophy.  Bronze trophies whose
    description is a chapter completion climb fastest of all."""
    tier = (trophy.get("tier") or "bronze").lower()
    if tier == "bronze" and "chapter" in (trophy.get("description") or "").lower():
        return STORY_BRONZE_SLOPE
    return PRIOR_SLOPE.get(tier, 0.60)


def staleness(history) -> float:
    """Share of consecutive read pairs in which the players-tracked count
    did not move at all: 1.0 for a page that has not refreshed since the
    first read, ~0.0 for a page refreshing every few minutes.  The final
    case's recorder reads every 10 minutes, so a live page scores low."""
    if len(history) < 2:
        return 1.0
    frozen = 0
    for prev, cur in zip(history, history[1:]):
        if prev.get("players_tracked") == cur.get("players_tracked"):
            frozen += 1
    return frozen / (len(history) - 1)


def forecast_trophy(trophy, series, target, horizon_days, stale):
    """Forecast one trophy.  `series` is a list of (moment, rate, players)
    with rate not None, oldest first; `target` is the forecast instant."""
    idx = trophy["_index"]
    last_moment, last_rate, last_players = series[-1]
    x_last = log_time(last_moment)
    x_target = log_time(target)
    dx = max(0.0, x_target - x_last)

    # The trend window: the last 24 hours of reads, or all of them.
    window_start = series[-1][0].timestamp() - 24 * 3600.0
    window = [s for s in series if s[0].timestamp() >= window_start]
    if len(window) < 3:
        window = series
    xs = [log_time(m) for m, _, _ in window]
    ys = [logit_of(r) for _, r, _ in window]
    fit_slope, _fit_int, resid, n = ols_fit(xs, ys)

    # Shrink the fitted slope toward the tier prior.  Flat data (zero
    # residual) is down-weighted: a flat line over identical reads says
    # the page did not refresh, not that the rate stopped moving.
    info = n if resid > FLAT_RESID else n * FLAT_DATA_WEIGHT
    slope = (info * fit_slope + PRIOR_WEIGHT * prior_slope(trophy)) / (info + PRIOR_WEIGHT)
    # Trend line, anchored at the last observed rate.
    median_logit = logit_of(last_rate) + slope * dx

    # Counts view: extrapolate earners and players separately, then divide.
    earners = [max(0.0, r) * p / 100.0 for _, r, p in window]
    log_e = [math.log(e + 0.5) for e in earners]
    slope_e, _, _, _ = ols_fit(xs, log_e)
    log_p = [math.log(max(1, p)) for _, _, p in window]
    slope_p, _, _, _ = ols_fit(xs, log_p)
    e_target = math.exp(math.log(max(0.0, last_rate) * max(1, last_players) / 100.0 + 0.5) + slope_e * dx)
    p_target = math.exp(math.log(max(1, last_players)) + slope_p * dx)
    rate_counts = 100.0 * e_target / max(1.0, p_target)
    # Average the two medians in logit space: they agree when the page
    # is steady and hedge each other when players and earners diverge.
    median_logit = 0.5 * median_logit + 0.5 * logit_of(rate_counts)

    # Refresh dilution: an overdue refresh reprices the level down, and
    # the uncertainty around it widens the band.  The shift fades with
    # horizon (days of earning rebuild the level after the repricing),
    # and a trophy already at ~100% gets none of it -- the first refresh
    # left that trophy at 100%.
    horizon_factor = 0.3 + 0.7 * math.exp(-horizon_days / 2.0)
    if last_rate < 99.5:
        median_logit += REFRESH_LOG_FACTOR * stale * horizon_factor
        shock_sigma = REFRESH_SIGMA * stale * horizon_factor
    else:
        shock_sigma = 0.0

    sigma = math.sqrt(
        (WALK_SIGMA * math.sqrt(horizon_days)) ** 2
        + resid ** 2
        + shock_sigma ** 2
        + FLOOR_SIGMA ** 2
    )
    row = [round4(min(99.99, max(0.01, pct_of(median_logit + z * sigma)))) for z in Z]
    # Keep the quantiles non-decreasing after rounding.
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return {"id": trophy["id"], "q": row}


def forecast_case(case: dict, name: str) -> dict:
    """Forecast every trophy of one case."""
    levels = case.get("levels") or LEVELS
    if len(levels) != 5:
        raise ValueError(f"{name} asks for {len(levels)} levels, not 5")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    target = parse_ts(case["target"])
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    stale = staleness(history)

    parsed = []
    for read in history:
        parsed.append((parse_ts(read["at"]), read.get("players_tracked") or 0, read.get("rates") or []))
    rows = []
    for i, trophy in enumerate(trophies):
        series = [(m, rates[i], p) for m, p, rates in parsed if i < len(rates) and rates[i] is not None]
        if not series:
            # No read ever listed this trophy: fall back to a wide band
            # around a small tier prior rate.
            base = {"platinum": 0.1, "gold": 0.2, "silver": 0.5}.get(
                (trophy.get("tier") or "").lower(), 1.0
            )
            centre = logit_of(base)
            row = [round4(min(99.99, max(0.01, pct_of(centre + z * 1.5)))) for z in Z]
            rows.append({"id": trophy["id"], "q": row})
            continue
        trophy = dict(trophy)
        trophy["_index"] = i
        rows.append(forecast_trophy(trophy, series, target, horizon_days, stale))
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
