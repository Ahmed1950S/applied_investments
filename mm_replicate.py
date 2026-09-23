"""
mm_replicate.py
---------------
Moreira & Muir (2017), "Volatility-Managed Portfolios", JF 72(4).
Replication + post-publication out-of-sample test.

Output sections:
  1  Replication check against MM Table I Panel A
  2  Spanning alphas, paper vs OOS
  3  Real-time implementable strategy at three leverage caps (MM Table V)
  4  OOS verdict, with block-bootstrap p-values and a multiple-testing screen
  5  Drawdowns, controlled for average exposure
  6  Confound check: is dSR just tracking the factor's own performance?
  7  Episodes, decomposed into mechanical de-risking vs genuine timing

Sections 5 and 7 both apply the same control: a strategy that simply holds
less on average will cut drawdowns and "win" in falling markets without any
timing skill at all. Every headline number here is net of that.

pip install pandas numpy statsmodels scipy pyarrow openpyxl
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA, OUT = Path("data"), Path("output")
OUT.mkdir(exist_ok=True)

# French: MktRF SMB HML MOM RMW CMA | global-q: ROE IA EG | AQR: BAB
# EG (Hou-Mo-Xue-Zhang 2021) postdates the paper: an out-of-sample test in the
# FACTOR dimension, not just the time dimension.
FACTORS = ["MktRF", "SMB", "HML", "MOM", "RMW", "CMA", "ROE", "IA", "EG", "BAB"]

PAPER_END = pd.Period("2015-12", "M")
OOS_START = pd.Period("2017-01", "M")

MIN_DAYS, MIN_OBS_RT = 15, 120
GAMMA, COST_BPS = 5.0, 10.0
LEVERAGE_CAPS = [None, 1.5, 1.0]        # MM Table V
N_BOOT, BLOCK = 5000, 6
MAIN_CAP = 1.0                          # no-leverage strategy used in 4,5,6,7

DSR_MIN = 0.10                          # below this, dSR is noise over ~9y

MM_TABLE1 = {
    "MktRF": (4.86, 1065), "SMB": (-0.58, 1065), "HML": (1.97, 1065),
    "MOM":  (12.51, 1060), "RMW": (2.44, 621),   "CMA": (0.38, 621),
    "ROE":  (5.48, 575),   "IA":  (1.55, 575),   "BAB": (5.67, 996),
}


# ======================================================================
# presentation helpers
# ======================================================================
def banner(n: int, title: str, sub: str = "") -> None:
    print(f"\n\n{'=' * 96}\n{n}.  {title}")
    if sub:
        print(f"    {sub}")
    print("=" * 96)


def stars(p: float) -> str:
    if pd.isna(p):
        return "   "
    return "***" if p < 0.01 else "** " if p < 0.05 else "*  " if p < 0.10 else "   "


def show(df: pd.DataFrame, fmt: dict = None) -> None:
    """Format numeric columns only; string columns (stars, verdicts) pass through."""
    d = df.copy()
    for c, f in (fmt or {}).items():
        if c not in d.columns or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        d[c] = d[c].map(lambda v: "" if pd.isna(v) else f.format(v))
    print(d.to_string())


# ======================================================================
# core construction
# ======================================================================
def realized_variance(daily: pd.Series, fixed_22: bool = True) -> pd.Series:
    """RV_t = sum_{d in t}(f_d - mean_t)^2.  MM eq.(2) divides by 22 whatever
    the true day count; the difference is ~0.01pp on alpha."""
    s = daily.dropna()
    if s.empty:
        return pd.Series(dtype=float, name="RV")
    g = s.groupby(pd.Grouper(freq="ME"))
    rv = (g.apply(lambda x: ((x - x.sum() / 22.0) ** 2).sum()) if fixed_22
          else g.apply(lambda x: ((x - x.mean()) ** 2).sum()))
    rv = rv[g.size() >= MIN_DAYS]
    rv.index = rv.index.to_period("M")
    return rv.rename("RV")


def build_managed(monthly: pd.Series, rv: pd.Series):
    """f^sigma_{t+1} = (c/RV_t) f_{t+1}, c set so vols match within window."""
    df = pd.concat([monthly.rename("f"), rv.shift(1).rename("RV_lag")],
                   axis=1).dropna()
    df = df[df["RV_lag"] > 0]
    if df.empty:
        return None, None
    raw = df["f"] / df["RV_lag"]
    c = df["f"].std() / raw.std()
    return (c * raw).rename("managed"), df["f"].rename("f")


def spanning(managed: pd.Series, orig: pd.Series) -> dict:
    df = pd.concat([managed, orig], axis=1).dropna()
    res = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit(cov_type="HC1")
    a, se = res.params.iloc[0], res.bse.iloc[0]
    return {"alpha": a * 12 * 100, "se": se * 12 * 100, "t": a / se,
            "beta": res.params.iloc[1],
            "appraisal": (a / np.sqrt(res.mse_resid)) * np.sqrt(12),
            "N": int(res.nobs)}


def realtime_managed(monthly: pd.Series, rv: pd.Series, w_cap: float = None):
    """
    Expanding-window c, so this is something an investor could have held.
    w_cap replicates MM Table V. Their uncapped weights reach a 99th percentile
    of 6.4, and they show alphas survive caps of 1.0 and 1.5. Running uncapped
    only would blame the strategy for leverage the paper never claimed.
    """
    df = pd.concat([monthly.rename("f"), rv.shift(1).rename("RV_lag")],
                   axis=1).dropna()
    df = df[df["RV_lag"] > 0]
    if len(df) <= MIN_OBS_RT:
        return None, None, None

    raw = df["f"] / df["RV_lag"]
    rets, wts = {}, {}
    for i in range(MIN_OBS_RT, len(df)):
        c_t = df["f"].iloc[:i].std() / raw.iloc[:i].std()
        w_t = c_t / df["RV_lag"].iloc[i]
        if w_cap is not None:
            w_t = min(w_t, w_cap)
        rets[df.index[i]] = w_t * df["f"].iloc[i]
        wts[df.index[i]] = w_t
    return (pd.Series(rets, name="rt"), df["f"].rename("f"),
            pd.Series(wts, name="w"))


def net_of_costs(w: pd.Series, gross: pd.Series, bps=COST_BPS) -> pd.Series:
    return (gross - w.diff().abs().fillna(0.0) * bps / 1e4).rename("rt_net")


# ======================================================================
# statistics
# ======================================================================
def sharpe(x) -> float:
    x = pd.Series(x).dropna()
    return np.nan if len(x) < 2 or x.std() == 0 else x.mean() / x.std() * np.sqrt(12)


def ceq(x, g=GAMMA) -> float:
    return x.mean() * 12 - 0.5 * g * x.var() * 12


def max_dd(x) -> float:
    cum = (1 + pd.Series(x).dropna()).cumprod()
    return float((cum / cum.cummax() - 1).min())


def boot_dsharpe(a: pd.Series, b: pd.Series, n_boot=N_BOOT, block=BLOCK, seed=0):
    """Circular block bootstrap for SR(a)-SR(b). Blocks preserve the volatility
    clustering the strategy exploits; an iid bootstrap would destroy exactly
    the dependence that matters."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 36:
        return np.nan, np.nan
    A, B = df["a"].to_numpy(), df["b"].to_numpy()
    n = len(A)
    obs = sharpe(A) - sharpe(B)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    draws = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.concatenate(
            [np.arange(s, s + block) % n for s in rng.integers(0, n, nb)])[:n]
        draws[k] = sharpe(A[idx]) - sharpe(B[idx])
    return obs, float(np.nanmean(np.abs(draws - np.nanmean(draws)) >= abs(obs)))


def fisher_compare(r1: float, n1: int, r2: float, n2: int):
    """Are two correlations from independent samples actually different?
    With n=9 per window, r=+0.21 and r=+0.68 are NOT distinguishable."""
    if min(n1, n2) < 4:
        return np.nan, np.nan
    z = (np.arctanh(r2) - np.arctanh(r1)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


# ======================================================================
# plumbing
# ======================================================================
def available(daily, monthly) -> list:
    out = []
    for f in FACTORS:
        hd = f in daily.columns and daily[f].notna().any()
        hm = f in monthly.columns and monthly[f].notna().any()
        if hd and hm:
            out.append(f)
        else:
            miss = "/".join(x for x, ok in [("daily", hd), ("monthly", hm)] if not ok)
            print(f"   [skip] {f}: no {miss} data")
    return out


def clip(s, lo=None, hi=None):
    if lo is not None:
        s = s[s.index >= lo]
    if hi is not None:
        s = s[s.index <= hi]
    return s


def windows(end):
    return {"paper": (None, PAPER_END), "oos": (OOS_START, end)}


# ======================================================================
# tables
# ======================================================================
def table_spanning(daily, monthly, facs, end) -> pd.DataFrame:
    rows = []
    for f in facs:
        rv = realized_variance(daily[f])
        for wn, (lo, hi) in windows(end).items():
            m, r = clip(monthly[f].dropna(), lo, hi), clip(rv, lo, hi)
            if len(m) < 36:
                continue
            managed, orig = build_managed(m, r)
            if managed is None or len(managed) < 36:
                continue
            res = spanning(managed, orig)
            res.update(factor=f, window=wn,
                       SR_bh=sharpe(orig), SR_vm=sharpe(managed))
            rows.append(res)
    return pd.DataFrame(rows).set_index(["factor", "window"])


def table_realtime(daily, monthly, facs, end, w_cap=None, boot=False) -> pd.DataFrame:
    rows = []
    for f in facs:
        rv = realized_variance(daily[f])
        rt, orig, w = realtime_managed(monthly[f].dropna(), rv, w_cap=w_cap)
        if rt is None:
            continue
        net = net_of_costs(w, rt)
        for wn, (lo, hi) in windows(end).items():
            a, b = clip(net, lo, hi), clip(orig.reindex(rt.index), lo, hi)
            if len(a) < 36:
                continue
            row = {"factor": f, "window": wn, "N": len(a),
                   "SR_bh": sharpe(b), "SR_vm": sharpe(a),
                   "dSR": sharpe(a) - sharpe(b),
                   "CEQ_bh": ceq(b) * 100, "CEQ_vm": ceq(a) * 100,
                   "DD_bh": max_dd(b) * 100, "DD_vm": max_dd(a) * 100,
                   "turn": w.reindex(a.index).diff().abs().mean(),
                   "avg_w": w.reindex(a.index).mean(),
                   "w_p99": w.reindex(a.index).quantile(0.99)}
            if boot:
                row["p"] = boot_dsharpe(a, b)[1]
            rows.append(row)
    return pd.DataFrame(rows).set_index(["factor", "window"])


# ======================================================================
# sections
# ======================================================================
def sec1_replication(span):
    banner(1, "REPLICATION CHECK",
           "MM Table I Panel A. Within ~0.5pp on alpha is a success: French "
           "has rebuilt\n    the library repeatedly since 2016 and CRSP restates.")
    p = span.xs("paper", level="window")
    rows = []
    for f, (a_mm, n_mm) in MM_TABLE1.items():
        if f not in p.index:
            continue
        rows.append({"factor": f, "alpha": p.loc[f, "alpha"], "MM": a_mm,
                     "diff": p.loc[f, "alpha"] - a_mm, "t": p.loc[f, "t"],
                     "N": int(p.loc[f, "N"]), "N_MM": n_mm,
                     "dN": int(p.loc[f, "N"]) - n_mm})
    out = pd.DataFrame(rows).set_index("factor")
    show(out, {"alpha": "{:7.2f}", "MM": "{:7.2f}", "diff": "{:+6.2f}",
               "t": "{:6.2f}"})
    print(f"\n    max |diff| = {out['diff'].abs().max():.2f}pp   "
          f"max |dN| = {int(out['dN'].abs().max())} months")
    print("    A CONSTANT dN within each data source is a vintage/convention "
          "offset,\n    not a per-factor problem.")
    return out


def sec2_spanning(span):
    banner(2, "SPANNING ALPHAS, PAPER vs OUT-OF-SAMPLE",
           "alpha in % p.a.  */**/*** = 10/5/1% on the White t-stat")
    u = span.unstack("window")
    rows = []
    for f in u.index:
        rec = {"factor": f}
        for wn, tag in [("paper", "in"), ("oos", "oos")]:
            if ("alpha", wn) not in u.columns or pd.isna(u.loc[f, ("alpha", wn)]):
                continue
            t = u.loc[f, ("t", wn)]
            rec[f"alpha_{tag}"] = u.loc[f, ("alpha", wn)]
            rec[f"t_{tag}"] = t
            rec[f"sig_{tag}"] = stars(2 * (1 - stats.norm.cdf(abs(t))))
            rec[f"SRbh_{tag}"] = u.loc[f, ("SR_bh", wn)]
        rec["change"] = rec.get("alpha_oos", np.nan) - rec.get("alpha_in", np.nan)
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("factor")
    cols = ["alpha_in", "t_in", "sig_in", "alpha_oos", "t_oos", "sig_oos",
            "change", "SRbh_in", "SRbh_oos"]
    out = out[[c for c in cols if c in out.columns]]
    fmt = {c: "{:7.2f}" for c in out.columns
           if c not in ("change", "sig_in", "sig_oos")}
    fmt["change"] = "{:+7.2f}"
    show(out, fmt)
    return out


def sec3_realtime(rt_by_cap):
    banner(3, "REAL-TIME IMPLEMENTABLE STRATEGY, NET OF COSTS",
           "Expanding-window c. dSR = Sharpe(managed) - Sharpe(buy-and-hold).\n"
           "    Caps follow MM Table V. READ THE IN-SAMPLE, CAP 1.0 COLUMN "
           "FIRST: it asks\n    whether the paper's own window delivered a "
           "tradeable gain under the\n    leverage constraint the paper itself "
           "proposes.")
    for wn, label in [("paper", "IN-SAMPLE (to 2015-12)"), ("oos", "OUT-OF-SAMPLE")]:
        rows = []
        for cap, t in rt_by_cap.items():
            if wn not in t.index.get_level_values("window"):
                continue
            s = t.xs(wn, level="window")
            for f in s.index:
                rows.append({"factor": f, "cap": cap, "dSR": s.loc[f, "dSR"],
                             "p": s.loc[f, "p"] if "p" in s.columns else np.nan})
        if not rows:
            continue
        d = pd.DataFrame(rows).pivot(index="factor", columns="cap")
        d = d.reindex(columns=list(rt_by_cap), level=1)
        print(f"\n    --- {label} ---")
        print(d.round(3).to_string())


def sec4_verdict(span, rt_cap, facs):
    k = len(facs)
    t_bonf = stats.norm.ppf(1 - 0.025 / k)
    p_bonf = 0.05 / k
    banner(4, "OUT-OF-SAMPLE VERDICT",
           f"{k} factors tested. Bonferroni: |t| > {t_bonf:.2f} on alpha, "
           f"p < {p_bonf:.4f} on dSR.\n    dSR from the no-leverage strategy "
           f"(cap {MAIN_CAP}). Report bonf_dSR honestly:\n    a nominal 5% "
           f"result among 9 tests is weak evidence.")
    s = span.xs("oos", level="window")
    r = rt_cap.xs("oos", level="window")
    d = s[["alpha", "t", "appraisal"]].join(
        r[["SR_bh", "SR_vm", "dSR", "p"]], how="inner")

    sig_a = d["t"].abs() > 1.96
    sig_a_b = d["t"].abs() > t_bonf
    trade = (d["dSR"] > DSR_MIN) & (d["p"] < 0.05)
    trade_b = (d["dSR"] > DSR_MIN) & (d["p"] < p_bonf)

    d["verdict"] = np.where(sig_a_b & trade, "SURVIVES (nominal dSR)",
                   np.where(sig_a & trade, "survives (nominal)",
                   np.where(d["t"] < -1.96, "REVERSES",
                   np.where(d["alpha"] > 0, "alpha>0, not tradeable",
                            "no effect"))))
    d["bonf_dSR"] = np.where(trade_b, "yes", "no")
    show(d, {"alpha": "{:7.2f}", "t": "{:6.2f}", "appraisal": "{:6.2f}",
             "SR_bh": "{:6.2f}", "SR_vm": "{:6.2f}", "dSR": "{:+6.2f}",
             "p": "{:6.3f}"})
    n_bonf = int((d["bonf_dSR"] == "yes").sum())
    print(f"\n    {n_bonf} of {k} factors clear Bonferroni on dSR.")
    return d


def sec5_drawdowns(daily, monthly, facs, end, w_cap=MAIN_CAP):
    """
    Drawdown reduction is meaningless without an exposure control: a CONSTANT
    weight of avg_w cuts drawdown on its own. DD_const is that counterfactual.
    timing_DD = DD_vm - DD_const; negative means the strategy beat a passive
    de-risking of the same average size.
    """
    banner(5, "DRAWDOWNS, CONTROLLED FOR AVERAGE EXPOSURE",
           "Max drawdown %, no-leverage cap. DD_const = a constant weight of "
           "avg_w.\n    timing_DD = DD_vm - DD_const. Negative means genuine "
           "skill; near zero\n    means the strategy simply held less.")
    rows = []
    for f in facs:
        rv = realized_variance(daily[f])
        rt, orig, w = realtime_managed(monthly[f].dropna(), rv, w_cap=w_cap)
        if rt is None:
            continue
        for wn, (lo, hi) in windows(end).items():
            a = clip(rt, lo, hi)
            b = clip(orig.reindex(rt.index), lo, hi)
            if len(a) < 36:
                continue
            aw = w.reindex(a.index).mean()
            rows.append({"factor": f, "window": wn, "avg_w": aw,
                         "DD_bh": max_dd(b) * 100,
                         "DD_const": max_dd(aw * b) * 100,
                         "DD_vm": max_dd(a) * 100})
    d = pd.DataFrame(rows).set_index(["factor", "window"])
    d["timing_DD"] = d["DD_vm"] - d["DD_const"]

    for wn, label in [("paper", "in-sample"), ("oos", "out-of-sample")]:
        if wn not in d.index.get_level_values("window"):
            continue
        s = d.xs(wn, level="window")
        print(f"\n    --- {label} ---")
        show(s, {"avg_w": "{:6.2f}", "DD_bh": "{:8.1f}", "DD_const": "{:9.1f}",
                 "DD_vm": "{:8.1f}", "timing_DD": "{:+10.1f}"})
        print(f"      mean timing_DD = {s['timing_DD'].mean():+.1f}pp   "
              f"negative in {int((s['timing_DD'] < 0).sum())}/{len(s)} factors")
    return d


def sec6_confound(span, rt_cap):
    banner(6, "CONFOUND CHECK",
           "Is dSR measuring volatility timing, or tracking how the underlying\n"
           "    factor did? Value, size and investment had a miserable "
           "2017-2025 as\n    UNMANAGED factors. With n=9 per window, treat "
           "these correlations as\n    suggestive: the Fisher test says whether "
           "the windows actually differ.")
    res, store = {}, {}
    for wn in ["paper", "oos"]:
        if wn not in span.index.get_level_values("window"):
            continue
        s, r = span.xs(wn, level="window"), rt_cap.xs(wn, level="window")
        d = s[["alpha"]].join(r[["SR_bh", "dSR"]], how="inner").dropna()
        if len(d) < 4:
            continue
        print(f"\n    --- {wn} (n={len(d)} factors) ---")
        res[wn] = {}
        for y in ["dSR", "alpha"]:
            sl, ic, rr, p, _ = stats.linregress(d["SR_bh"], d[y])
            res[wn][y] = (rr, len(d))
            print(f"      {y:5s} = {ic:+6.2f} {sl:+6.2f} * SR_bh     "
                  f"r = {rr:+.2f}   p = {p:.3f}")
        store[wn] = d

    if "paper" in res and "oos" in res:
        print("\n    Fisher z, are the two windows different?")
        for y in ["dSR", "alpha"]:
            (r1, n1), (r2, n2) = res["paper"][y], res["oos"][y]
            z, p = fisher_compare(r1, n1, r2, n2)
            verdict = "DIFFERENT" if p < 0.05 else "not distinguishable"
            print(f"      {y:5s}: r_paper={r1:+.2f}  r_oos={r2:+.2f}  "
                  f"z={z:+.2f}  p={p:.3f}   -> {verdict}")

    if "oos" in store:
        print("\n    OOS scatter (check for two clusters rather than a line):")
        show(store["oos"].sort_values("SR_bh")[["SR_bh", "dSR", "alpha"]],
             {"SR_bh": "{:6.2f}", "dSR": "{:+6.2f}", "alpha": "{:7.2f}"})
    return store


def sec7_episodes(daily, monthly, facs, w_cap=MAIN_CAP):
    eps = {"2017-2019": ("2017-01", "2019-12"),
           "COVID 2020": ("2020-01", "2020-12"),
           "2022 bear": ("2022-01", "2022-12")}
    rows = []
    for f in facs:
        rv = realized_variance(daily[f])
        rt, orig, w = realtime_managed(monthly[f].dropna(), rv, w_cap=w_cap)
        if rt is None:
            continue
        for name, (lo, hi) in eps.items():
            lo, hi = pd.Period(lo, "M"), pd.Period(hi, "M")
            a, b = clip(rt, lo, hi), clip(orig.reindex(rt.index), lo, hi)
            if len(a) < 6:
                continue
            bh = ((1 + b).prod() - 1) * 100
            vm = ((1 + a).prod() - 1) * 100
            aw = w.reindex(a.index).mean()
            mech = ((1 + aw * b).prod() - 1) * 100
            rows.append({"factor": f, "episode": name, "bh": bh, "vm": vm,
                         "gap": vm - bh, "avg_w": aw,
                         "mech": mech - bh, "timing": vm - mech})
    ep = pd.DataFrame(rows).set_index(["factor", "episode"])

    banner(7, "EPISODES MM NEVER SAW",
           "Cumulative %, no-leverage cap.  gap = mech + timing, where mech is "
           "what a\n    CONSTANT exposure of avg_w would have delivered. Only "
           "'timing' reflects\n    WHEN the strategy was in and out.")
    for name in eps:
        if name not in ep.index.get_level_values("episode"):
            continue
        s = ep.xs(name, level="episode")
        print(f"\n    --- {name} ---")
        show(s, {"bh": "{:8.1f}", "vm": "{:8.1f}", "gap": "{:+8.1f}",
                 "avg_w": "{:6.2f}", "mech": "{:+8.1f}", "timing": "{:+8.1f}"})
        print(f"      mean timing = {s['timing'].mean():+.1f}pp   "
              f"mean |timing| = {s['timing'].abs().mean():.1f}pp")

    print("\n    Mechanics: regress gap on bh across factors within each episode.")
    print("    R2 near 1 means the episode reveals only position SIZE, not skill.")
    mrows = []
    for name, g in ep.groupby(level="episode"):
        if len(g) < 4:
            continue
        sl, _, rr, p, _ = stats.linregress(g["bh"], g["gap"])
        mrows.append({"episode": name, "slope": sl, "R2": rr ** 2,
                      "implied_exposure": 1 + sl, "avg_w": g["avg_w"].mean(),
                      "mean_timing": g["timing"].mean(),
                      "abs_timing": g["timing"].abs().mean(),
                      "reads_as": "MECHANICAL" if rr ** 2 > 0.90 else "has timing"})
    show(pd.DataFrame(mrows).set_index("episode"),
         {"slope": "{:+6.2f}", "R2": "{:5.3f}", "implied_exposure": "{:6.2f}",
          "avg_w": "{:6.2f}", "mean_timing": "{:+7.1f}", "abs_timing": "{:6.1f}"})
    return ep


# ======================================================================
def main():
    daily = pd.read_parquet(DATA / "factors_daily.parquet")
    monthly = pd.read_parquet(DATA / "factors_monthly.parquet")
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("=" * 96)
    print("MOREIRA & MUIR (2017) - REPLICATION AND POST-PUBLICATION TEST")
    print("=" * 96)
    facs = available(daily, monthly)
    end = min(monthly[f].dropna().index.max() for f in facs)
    print(f"   factors   : {', '.join(facs)}")
    print(f"   in-sample : ...{PAPER_END}")
    print(f"   OOS       : {OOS_START}...{end}  ({(end - OOS_START).n + 1} months)")
    print(f"   costs {COST_BPS:.0f}bp | gamma {GAMMA:.0f} | "
          f"bootstrap {N_BOOT} draws, {BLOCK}m blocks")

    span = table_spanning(daily, monthly, facs, end)

    rt_by_cap = {}
    for cap in LEVERAGE_CAPS:
        label = "uncapped" if cap is None else f"cap{cap}"
        print(f"   ... real-time, {label}")
        rt_by_cap[label] = table_realtime(daily, monthly, facs, end,
                                          w_cap=cap, boot=True)
    main_rt = rt_by_cap[f"cap{MAIN_CAP}"]

    rep = sec1_replication(span)
    sp = sec2_spanning(span)
    sec3_realtime(rt_by_cap)
    ver = sec4_verdict(span, main_rt, facs)
    dd = sec5_drawdowns(daily, monthly, facs, end)
    sec6_confound(span, main_rt)
    ep = sec7_episodes(daily, monthly, facs)

    with pd.ExcelWriter(OUT / "mm_results.xlsx") as xl:
        rep.to_excel(xl, sheet_name="1_replication")
        sp.to_excel(xl, sheet_name="2_spanning")
        span.to_excel(xl, sheet_name="2_spanning_raw")
        for lbl, t in rt_by_cap.items():
            t.to_excel(xl, sheet_name=f"3_rt_{lbl}"[:31])
        ver.to_excel(xl, sheet_name="4_verdict")
        dd.to_excel(xl, sheet_name="5_drawdowns")
        ep.to_excel(xl, sheet_name="7_episodes")
    print(f"\n\nWritten to {OUT / 'mm_results.xlsx'}")


if __name__ == "__main__":
    main()