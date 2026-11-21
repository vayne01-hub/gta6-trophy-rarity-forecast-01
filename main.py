"""gta6-trophy-rarity-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five
quantiles (5/25/50/75/95%) of its earn rate on Exophase at the case's target.

Model, in logit space (logit(p) = ln(p / (100 - p)), p in percent):

  * Earners / tracked decomposition. A rate is earners / players_tracked, and
    the denominator grows fast in week one (638 -> 3,868 overnight on launch
    night), which drags rates down even while earners rise. So the model fits
    log(earners) and log(players_tracked) separately against x = ln(1 + hours
    since 2026-11-19T00:00Z) over the last 24 h of distinct reads, and reads
    the rate off their difference at the target. The log-time axis is the
    damping: extrapolating from 2.5 days of history to a 4.5-day horizon moves
    x by only ~1, so a trend cannot run away. A plain logit-linear fit on the
    same axis (the house curve baseline's shape) and persistence are kept as
    rival variants.
  * Piecewise-constant reads are collapsed: consecutive reads with identical
    players_tracked and rates carry no new information, so only the first read
    of each distinct state is fitted. The model follows refreshes, it does not
    chase every 10-minute step.
  * Pooling / shrinkage. Growth slopes are shrunk toward the mean slope of
    the trophy's last-rate band (high / mid / low), with weight
    n / (n + 3) on the trophy's own slope, n the number of distinct states
    fitted. With one distinct state (the practice cases) every slope is the
    group mean, which is flat, so every variant collapses to persistence --
    exactly what constant history justifies.
  * Ceiling trophy. A trophy pinned at ~100% (the prologue: every tracked
    player has it) can only fall as new players join tracked. Its earners are
    forecast as E_last + 0.85 * (P_target - P_last): new tracked players earn
    the prologue quickly but not instantly, so the rate tends to ~85% as the
    tracked base grows; the spread for it is widened to cover f in [0.5, 1].
    The rule fires only while history still pins the trophy at >= 99.5%; once
    its rate has started falling, the normal fit takes over.
  * Self-calibration (inside the program). The program backtests on the
    case's own history: it fits each variant on the prefix ending at
    checkpoints at 25%, 50% and 75% of the history's time span, forecasts the
    last read at the scaled horizon, and scores mean pinball loss on the logit
    scale. It picks the variant and spread multiplier (grid 0.7 / 1.0 / 1.4 /
    2.0) with the lowest backtest loss; a variant within 5% of the best is
    treated as tied and the most structured variant wins the tie (earners /
    tracked, then logit-linear, then persistence), then the multiplier
    nearest 1.0. The 5% band is deliberate: backtest horizons (hours to
    ~2 days) are far shorter than the real horizon (4.5 days), where trends
    matter more than persistence, so a small short-horizon edge for
    persistence does not transfer. With fewer than 3 distinct states the
    backtest is degenerate and the program defaults to (earners/tracked, 1.0).
  * Spread. sigma = multiplier * sqrt(rw^2 + resid^2 + refresh^2), floored at
    sqrt(rw^2 + refresh^2): rw = 0.5 * sqrt(horizon in days) is the house
    baseline's random walk; resid is the trophy's fit residual in logit,
    shrunk to its band's median; refresh is the median absolute logit jump
    between consecutive distinct states in the history (a refresh the program
    has not seen is always possible), defaulting to 0.75 logit when history
    shows no refresh at all. The floor stops a degenerate backtest on
    constant history from collapsing the spread to zero.

Deterministic: no clock, no unseeded randomness, no dict-order dependence in
output. Runtime is far under 8 s per case (a few ms on the practice cases).

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. It
reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out
to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day (house baseline).
SIGMA_RW = 0.5
# Release origin the house curve measures hours from (RULES.txt).
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Fit window: the last 24 h of distinct reads (house curve baseline's window).
WINDOW_HOURS = 24.0
# Shrinkage strength for a trophy's own slope toward its band mean (in distinct states).
SHRINK_K = 3.0
# Spread multiplier grid the self-calibration backtest searches.
MULT_GRID = (0.7, 1.0, 1.4, 2.0)
# Refresh allowance in logit when history shows no refresh at all.
REFRESH_DEFAULT = 0.75
# A trophy at or above this rate is treated as pinned at the ceiling.
CEILING_RATE = 99.5
# Share of newly tracked players assumed to earn a ceiling trophy by target.
CEILING_NEW_EARN = 0.85
# Extra logit variance (std) for a ceiling trophy's uncertain dilution.
CEILING_EXTRA_SIGMA = 1.0


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_origin(t: datetime) -> float:
    return max(-0.99, (t - ORIGIN).total_seconds() / 3600.0)


def x_of(t: datetime) -> float:
    return math.log1p(hours_since_origin(t))


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y ~ a + b*x. Returns (a, b, rms_residual)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class Case:
    """One parsed case: distinct refresh states plus per-trophy series."""

    def __init__(self, raw: dict, name: str):
        if raw["levels"] != LEVELS:
            raise ValueError(f"{name} asks for levels {raw['levels']}, not {LEVELS}")
        if not raw["history"]:
            raise ValueError(f"{name} has no history")
        self.name = name
        self.raw = raw
        self.trophies = raw["trophies"]
        self.n = len(self.trophies)
        self.horizon_days = raw["horizon_minutes"] / 1440.0
        self.target_x = x_of(parse_time(raw["target"]))
        # Collapse piecewise-constant reads to distinct states (first read of each run).
        states = []
        prev_key = None
        for read in raw["history"]:
            key = (read["players_tracked"], tuple(read["rates"]))
            if key != prev_key:
                states.append(read)
                prev_key = key
        self.states = states
        self.times = [parse_time(r["at"]) for r in states]
        self.xs = [x_of(t) for t in self.times]
        self.players = [float(r["players_tracked"]) for r in states]
        # Per-trophy rate series (None where the read lacked the trophy).
        self.rates: list[list[float | None]] = [
            [states[j]["rates"][i] for j in range(len(states))] for i in range(self.n)
        ]
        self.last_rate: list[float] = []
        for i in range(self.n):
            vals = [r for r in self.rates[i] if r is not None]
            self.last_rate.append(vals[-1] if vals else 0.5)
        # Refresh jumps in logit between consecutive distinct states, per trophy.
        self.refresh_jump = self._refresh_jump()
        # Per-prefix fit cache, filled lazily by _fit_context.
        self._fit_cache: dict[int, dict] = {}

    def _refresh_jump(self) -> float:
        """Median logit jump at actual refresh boundaries (nonzero changes only).

        Consecutive reads inside one refresh are identical, so the median over
        all consecutive pairs would be 0; only the changes between distinct
        states measure refresh noise.
        """
        jumps = []
        for i in range(self.n):
            series = self.rates[i]
            for j in range(1, len(series)):
                if series[j] is not None and series[j - 1] is not None:
                    d = abs(logit(series[j]) - logit(series[j - 1]))
                    if d > 1e-9:
                        jumps.append(d)
        if not jumps:
            return REFRESH_DEFAULT
        return min(1.2, median(jumps))

    def band(self, i: int) -> str:
        """Last-rate band, for pooling growth slopes.

        Near-ceiling trophies live in their own bands: at 90% a 1-point rate
        move is a ~0.1 logit move, at 1% it is ~1 logit, so their logit
        slopes must never be pooled with ordinary trophies' (trophy 2's
        -1.05 logit slope once dragged a whole band's mean to -0.58).
        """
        r = self.last_rate[i]
        if r >= CEILING_RATE:
            return "ceil"
        if r >= 10.0:
            return "high"
        if r >= 1.2:
            return "hi"
        if r >= 0.85:
            return "mid"
        return "lo"

    def window_idx(self, upto: int, hours: float = WINDOW_HOURS) -> list[int]:
        """Distinct-state indices within the last `hours` before state `upto` (inclusive)."""
        t_end = self.times[upto]
        idx = [j for j in range(upto + 1)
               if (t_end - self.times[j]).total_seconds() / 3600.0 <= hours]
        if len(idx) < 3:
            idx = list(range(upto + 1))
        return idx

    def _fit_context(self, upto: int) -> dict:
        """Cached per-prefix fit context: window indices, each trophy's raw
        logit-linear slope over that window, and each band's mean slope.

        Computing these once per prefix (instead of once per trophy per fit)
        is what keeps the backtest fast on final-size histories.
        """
        ctx = self._fit_cache.get(upto)
        if ctx is not None:
            return ctx
        idx = self.window_idx(upto)
        raw_logit: dict[int, float | None] = {}
        raw_lograte: dict[int, float | None] = {}
        for k in range(self.n):
            xs, ys, yl = [], [], []
            for j in idx:
                if self.rates[k][j] is not None:
                    xs.append(self.xs[j])
                    ys.append(logit(self.rates[k][j]))
                    yl.append(math.log(max(self.rates[k][j], 0.01)))
            raw_logit[k] = ols(xs, ys)[1] if len(xs) >= 2 else None
            raw_lograte[k] = ols(xs, yl)[1] if len(xs) >= 2 else None
        band_mean_logit: dict[str, float] = {}
        band_mean_lograte: dict[str, float] = {}
        for band in ("ceil", "high", "hi", "mid", "lo"):
            peers_l = [raw_logit[k] for k in range(self.n)
                       if self.band(k) == band and raw_logit[k] is not None]
            peers_r = [raw_lograte[k] for k in range(self.n)
                       if self.band(k) == band and raw_lograte[k] is not None]
            if peers_l:
                band_mean_logit[band] = sum(peers_l) / len(peers_l)
            if peers_r:
                band_mean_lograte[band] = sum(peers_r) / len(peers_r)
        ctx = {"idx": idx, "raw_logit": raw_logit, "raw_lograte": raw_lograte,
               "band_mean_logit": band_mean_logit,
               "band_mean_lograte": band_mean_lograte}
        self._fit_cache[upto] = ctx
        return ctx

    def fit_trophy(self, i: int, upto: int, variant: int) -> tuple[float, float]:
        """Forecast trophy i's logit rate at a target x, fitting states <= upto.

        variant 0 = persistence, 1 = logit-linear, 2 = earners/tracked split.
        Returns (logit_median, rms_residual_in_logit).
        """
        series = self.rates[i]
        # The last observed rate at or before `upto` (not the case's last
        # rate: in a backtest the prefix must not see the target read).
        last_upto = self.last_rate[i]
        for j in range(upto, -1, -1):
            if series[j] is not None:
                last_upto = series[j]
                break
        if variant == 0:
            resid = self._scatter(i, upto)
            return logit(last_upto), resid
        idx = self._fit_context(upto)["idx"]
        if variant == 1:
            ys = [logit(series[j]) for j in idx if series[j] is not None]
            xs_f = [self.xs[j] for j in idx if series[j] is not None]
            if len(xs_f) < 2:
                return logit(last_upto), self._scatter(i, upto)
            a, b, resid = ols(xs_f, ys)
            b = self._shrunk_slope(i, upto, b, len(xs_f), "logit")
            return a + b * self.target_x, resid
        # variant 2: log earners and log tracked separately.
        ys_e, ys_p, xs_f = [], [], []
        for j in idx:
            if series[j] is None:
                continue
            earners = series[j] * self.players[j] / 100.0
            ys_e.append(math.log(max(earners, 0.0) + 0.5))
            ys_p.append(math.log(max(self.players[j], 1.0)))
            xs_f.append(self.xs[j])
        if len(xs_f) < 2:
            return logit(last_upto), self._scatter(i, upto)
        a_e, b_e, r_e = ols(xs_f, ys_e)
        a_p, b_p, r_p = ols(xs_f, ys_p)
        slope = self._shrunk_slope(i, upto, b_e - b_p, len(xs_f), "lograte")
        log_rate = (a_e - a_p) + slope * self.target_x + math.log(100.0)
        # Re-anchor the intercept so the fitted curve passes through the last
        # observed rate at the last fitted x (guards against level drift when
        # the two fits' intercepts disagree).
        last_x = xs_f[-1]
        fitted_log_rate_last = (a_e - a_p) + (b_e - b_p) * last_x + math.log(100.0)
        observed_log_rate_last = math.log(max(series[idx[-1]], 0.01)) if series[idx[-1]] is not None else fitted_log_rate_last
        log_rate += observed_log_rate_last - fitted_log_rate_last
        rate = math.exp(log_rate)
        resid = math.sqrt(r_e * r_e + r_p * r_p)
        return logit(rate), resid

    def _scatter(self, i: int, upto: int) -> float:
        """RMS logit scatter of trophy i's rates around their mean, states <= upto."""
        vals = [logit(r) for r in self.rates[i][: upto + 1] if r is not None]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    def _shrunk_slope(self, i: int, upto: int, slope: float, n_fit: int,
                      kind: str) -> float:
        """Shrink trophy i's slope toward its band's mean slope (cached per prefix).

        `kind` is "logit" for a logit-space slope (variant 1) or "lograte" for
        a log-rate slope (variant 2's earners-minus-tracked slope): the two
        live on different scales and must never be pooled together.
        """
        key = "band_mean_logit" if kind == "logit" else "band_mean_lograte"
        group = self._fit_context(upto)[key].get(self.band(i), slope)
        w = n_fit / (n_fit + SHRINK_K)
        return w * slope + (1.0 - w) * group

    def ceiling_forecast(self, i: int) -> tuple[float, float]:
        """Forecast for a trophy pinned at ~100%: dilution by new tracked players.

        Returns (logit_median, extra_sigma). Fires only while history pins the
        trophy at >= CEILING_RATE; otherwise the normal fit is used.
        """
        series = self.rates[i]
        if min(r for r in series if r is not None) < CEILING_RATE:
            return None  # type: ignore[return-value]
        # Forecast players_tracked at target from its log-linear fit on all states.
        a_p, b_p, _ = ols(self.xs, [math.log(max(p, 1.0)) for p in self.players])
        p_target = math.exp(a_p + b_p * self.target_x)
        p_last = self.players[-1]
        e_last = self.last_rate[i] * p_last / 100.0
        e_target = e_last + CEILING_NEW_EARN * max(p_target - p_last, 0.0)
        rate = 100.0 * e_target / max(p_target, 1.0)
        return logit(rate), CEILING_EXTRA_SIGMA


def spread_sigma(case: Case, horizon_days: float, resid: float, mult: float,
                 extra: float = 0.0) -> float:
    """Total logit sigma for a horizon, with the floor that stops collapse."""
    rw = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    refresh = case.refresh_jump * math.sqrt(max(horizon_days, 1.0))
    refresh = min(refresh, 1.5)
    base = math.sqrt(rw * rw + resid * resid + refresh * refresh + extra * extra)
    floor = math.sqrt(rw * rw + refresh * refresh)
    return max(mult * base, floor)


def pinball_loss(y: float, qs: list[float]) -> float:
    """Mean pinball loss over the five levels, logit scale, y and qs in percent."""
    ly = logit(y)
    total = 0.0
    for tau, q in zip(LEVELS, qs):
        d = ly - logit(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def backtest(case: Case) -> tuple[int, float, dict]:
    """Self-calibration: fit on prefixes, forecast the last read, pick variant + multiplier.

    Returns (variant, multiplier, report) where report maps (variant, mult) to
    mean pinball loss. With fewer than 3 distinct states the backtest cannot
    run and the default (2, 1.0) is returned.
    """
    n_states = len(case.states)
    if n_states < 3:
        return 2, 1.0, {"degenerate": True, "states": n_states}
    t0 = case.times[0]
    t_last = case.times[-1]
    span_h = (t_last - t0).total_seconds() / 3600.0
    report: dict = {}
    n_cp = 0
    for frac in (0.25, 0.5, 0.75):
        cp_time = t0.timestamp() + frac * span_h * 3600.0
        upto = max(j for j in range(n_states) if case.times[j].timestamp() <= cp_time)
        if upto < 1 or upto >= n_states - 1:
            continue
        n_cp += 1
        horizon_bt_days = (t_last - case.times[upto]).total_seconds() / 86400.0
        for variant in (0, 1, 2):
            # Fit once per (checkpoint, variant): the fit does not depend on
            # the spread multiplier, only the pinball scoring does.
            fits = []
            for i in range(case.n):
                # Backtest target is the last read: fit on the prefix, then
                # read the median off at the last read's x.
                fits.append(_median_at_x(case, i, upto, variant, case.xs[-1]))
            for mult in MULT_GRID:
                losses = []
                for i, (lg_bt, resid) in enumerate(fits):
                    sig = spread_sigma(case, horizon_bt_days, resid, mult)
                    qs = [inv_logit(lg_bt + z * sig) for z in Z]
                    y = case.rates[i][-1]
                    if y is None:
                        continue
                    losses.append(pinball_loss(y, qs))
                if losses:
                    key = (variant, mult)
                    report.setdefault(key, []).append(sum(losses) / len(losses))
    if not report:
        return 2, 1.0, {"degenerate": True, "states": n_states}
    mean_loss = {k: sum(v) / len(v) for k, v in report.items()}
    best = min(mean_loss.values())
    # Backtest horizons (hours to ~2 days) are far shorter than the real
    # horizon (4.5 days), where trends matter more than persistence; so a
    # variant within 5% of the best is treated as tied, and the most
    # structured variant wins the tie, then the multiplier nearest 1.0.
    cands = [k for k, v in mean_loss.items() if v <= best * 1.05]
    cands.sort(key=lambda k: (-k[0], abs(k[1] - 1.0)))
    variant, mult = cands[0]
    detail = {f"v{v}_x{m}": round(vv, 6) for (v, m), vv in sorted(mean_loss.items())}
    detail["checkpoints"] = n_cp
    return variant, mult, detail


def _median_at_x(case: Case, i: int, upto: int, variant: int, x_target: float) -> tuple[float, float]:
    """fit_trophy's (median, resid) recomputed at an arbitrary x (for the backtest)."""
    saved = case.target_x
    case.target_x = x_target
    try:
        return case.fit_trophy(i, upto, variant)
    finally:
        case.target_x = saved


def forecast_case(raw: dict, name: str) -> tuple[dict, dict]:
    """Forecast one case. Returns (output_dict, info_dict for reporting)."""
    case = Case(raw, name)
    variant, mult, bt = backtest(case)
    # Group-median residual per band, for shrinking per-trophy residuals.
    band_resid: dict[str, list[float]] = {}
    rows = []
    for i, trophy in enumerate(case.trophies):
        ceil = case.ceiling_forecast(i) if case.last_rate[i] >= CEILING_RATE else None
        if ceil is not None:
            lg, extra = ceil
            resid = 0.0
        else:
            lg, resid = case.fit_trophy(i, len(case.states) - 1, variant)
            extra = 0.0
        band_resid.setdefault(case.band(i), []).append(resid)
        rows.append((i, trophy["id"], lg, resid, extra))
    band_med = {b: median(v) for b, v in band_resid.items()}
    out_rows = []
    for i, tid, lg, resid, extra in rows:
        resid_s = 0.5 * resid + 0.5 * band_med.get(case.band(i), resid)
        resid_s = min(max(resid_s, 0.05), 1.0)
        med = min(max(inv_logit(lg), 0.02), 99.5)
        lg_med = logit(med)
        sig = spread_sigma(case, case.horizon_days, resid_s, mult, extra)
        qs = [inv_logit(lg_med + z * sig) for z in Z]
        qs = [min(max(q, 0.01), 99.99) for q in qs]
        qs.sort()
        out_rows.append({"id": tid, "q": [round4(q) for q in qs]})
    info = {
        "variant": variant,
        "multiplier": mult,
        "backtest": bt,
        "states": len(case.states),
        "reads": len(raw["history"]),
        "refresh_jump": round(case.refresh_jump, 4),
        "horizon_days": round(case.horizon_days, 4),
    }
    return {"trophies": out_rows}, info


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        forecast, _info = forecast_case(raw, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
