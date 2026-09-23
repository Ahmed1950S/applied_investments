"""
mm_data_extra.py
----------------
Adds the non-French factors to the panels built by mm_data.py:
  * IA, ROE (and EG, bonus) from the Hou-Xue-Zhang q-factor library, global-q.org
  * BAB (USA) from AQR's Betting Against Beta equity factors

Daily series are used ONLY to build realized variance; monthly series are the
returns being scaled. The two are never compounded into each other, because a
daily-rebalanced long-short is not the same object as a monthly-held one.

pip install requests pandas openpyxl pyarrow

BAB must be downloaded by hand (AQR gates it behind a click-through). Get BOTH:
  https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Daily
  https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly
Drop them in ./data/ keeping AQR's filenames.

Downloads are cached in ./data/cache/. Delete that folder to force a refresh.
If your network kills the TLS handshake repeatedly (some university proxies do,
Weebly-hosted global-q is a common casualty), just open the two CSV URLs in a
browser, save them anywhere, and set GLOBALQ_LOCAL below.
"""

import hashlib
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT = Path("data")
OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research replication)"}
GLOBALQ_PAGE = "https://global-q.org/factors.html"

# Manual override: set these to local file paths if the download keeps failing.
GLOBALQ_LOCAL = {"daily": None, "monthly": None}

BAB_DAILY_XLSX = OUT / "Betting Against Beta Equity Factors Daily.xlsx"
BAB_MONTHLY_XLSX = OUT / "Betting Against Beta Equity Factors Monthly.xlsx"

FACTORS = ["MktRF", "SMB", "HML", "MOM", "RMW", "CMA", "ROE", "IA", "EG", "BAB"]

# Factors whose daily version rebalances more often than the monthly version.
# Compounded-daily vs native-monthly correlation is legitimately below 1 for
# these. It is not an error and does not affect the replication.
DAILY_REBALANCED = {"MOM", "SMB", "HML", "BAB"}


# ======================================================================
# networking
# ======================================================================
def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5,
        backoff_factor=1.5,                 # 0, 1.5, 3, 6, 12 seconds
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )))
    s.headers.update(HEADERS)
    return s


SESSION = _session()


def fetch_bytes(url: str, timeout: int = 120, use_cache: bool = True) -> bytes:
    """
    GET with retries and a local cache. urllib3's Retry does not cover a
    connection reset during the TLS handshake, which is exactly what
    global-q throws on rapid successive requests, so wrap it in an outer loop.
    """
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    ext = ".csv" if url.lower().endswith(".csv") else ".html"
    path = CACHE / f"{key}{ext}"

    if use_cache and path.exists() and path.stat().st_size > 0:
        print(f"   [cache] {url.split('/')[-1]}")
        return path.read_bytes()

    last = None
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            if not r.content:
                raise requests.exceptions.ConnectionError("empty body")
            path.write_bytes(r.content)
            return r.content
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            last = e
            wait = 2.0 * (attempt + 1)
            print(f"   !! {type(e).__name__} on {url.split('/')[-1]}, "
                  f"retry {attempt + 1}/4 in {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch {url} after 4 attempts. Download it in a browser "
        f"and set GLOBALQ_LOCAL at the top of this file."
    ) from last


# ======================================================================
# global-q.org
# ======================================================================
def globalq_urls() -> dict:
    """
    Scrape the Factors page for the current q-factor CSV links. The vintage
    year is in the filename and changes with each release, so never hardcode.
    """
    html = fetch_bytes(GLOBALQ_PAGE).decode("utf-8", errors="replace")

    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    hrefs = [h if h.startswith("http") else "https://global-q.org/" + h.lstrip("/")
             for h in hrefs]
    csvs = sorted({h for h in hrefs if h.lower().endswith(".csv")})

    # q5_factors_* in current vintages, q_factors_* in older ones.
    # Either contains R_IA and R_ROE, which is all we need.
    out = {
        "daily":   [h for h in csvs if re.search(r"q5?_factors_daily", h, re.I)],
        "monthly": [h for h in csvs if re.search(r"q5?_factors_monthly", h, re.I)],
    }

    if not out["daily"] or not out["monthly"]:
        print("--- could not identify factor CSVs. All CSV links found: ---")
        for h in csvs:
            print("   ", h)
        raise RuntimeError("global-q link pattern changed; inspect list above.")

    return out


def load_globalq(source: str, freq: str) -> pd.DataFrame:
    """
    Read a global-q factor CSV from a URL or a local path. Schema is detected,
    not assumed: the daily file carries an ISO 'date' string, the monthly file
    carries integer year/month columns, older vintages used YYYYMMDD.
    """
    if source.lower().startswith("http"):
        raw = io.BytesIO(fetch_bytes(source))
        label = source.split("/")[-1]
    else:
        raw = Path(source)
        label = raw.name

    df = pd.read_csv(raw)
    df.columns = [c.strip() for c in df.columns]

    lower = {c.lower(): c for c in df.columns}
    print(f"   {label}: columns = {list(df.columns)}")

    # ---- index ----
    if {"year", "month", "day"} <= set(lower):
        idx = pd.to_datetime(dict(year=df[lower["year"]],
                                  month=df[lower["month"]],
                                  day=df[lower["day"]]), errors="coerce")
        if freq == "M":
            idx = pd.PeriodIndex(idx, freq="M")

    elif {"year", "month"} <= set(lower):
        # PeriodIndex(year=..., month=...) was removed in pandas 2.2.
        y = df[lower["year"]].astype(int).astype(str)
        m = df[lower["month"]].astype(int).astype(str).str.zfill(2)
        per = pd.PeriodIndex(y + "-" + m, freq="M")
        idx = per if freq == "M" else per.to_timestamp(how="end")

    elif "date" in lower:
        parsed = pd.to_datetime(df[lower["date"]].astype(str).str.strip(),
                                errors="coerce")
        idx = parsed if freq == "D" else pd.PeriodIndex(parsed, freq="M")

    else:
        raise ValueError(f"No recognisable date column. Columns: {list(df.columns)}")

    df = df.set_index(pd.Index(idx, name="date"))

    bad = int(pd.isna(df.index).sum())
    if bad:
        print(f"   !! dropping {bad} rows with unparseable dates")
        df = df[pd.notna(df.index)]

    # ---- factor columns, case-insensitive ----
    want = {"r_ia": "IA", "r_roe": "ROE", "r_eg": "EG"}
    cols = {lower[k]: v for k, v in want.items() if k in lower}
    if not cols:
        raise ValueError(f"No R_IA / R_ROE columns. Columns: {list(df.columns)}")

    out = df[list(cols)].rename(columns=cols).apply(pd.to_numeric, errors="coerce")
    return (out / 100.0).sort_index()          # global-q reports percent


# ======================================================================
# AQR BAB
# ======================================================================
def _find_header_row(path: Path, sheet: str, probe: str = "USA",
                     max_scan: int = 40) -> int:
    """
    AQR prepends a disclaimer block of variable length. Locate the row that
    actually holds the country headers rather than guessing skiprows.
    """
    head = pd.read_excel(path, sheet_name=sheet, header=None, nrows=max_scan)
    for i in range(len(head)):
        row = head.iloc[i].astype(str).str.strip().str.upper()
        if (row == probe).any():
            return i
    raise ValueError(f"No header row containing '{probe}' in the first "
                     f"{max_scan} rows of {path.name}. Open it and check the "
                     f"sheet name and layout.")


def load_aqr_bab(path: Path, freq: str, sheet: str = "BAB Factors") -> pd.Series:
    hdr = _find_header_row(path, sheet)
    df = pd.read_excel(path, sheet_name=sheet, skiprows=hdr)
    df = df.rename(columns={df.columns[0]: "date"})
    df.columns = [str(c).strip() for c in df.columns]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    if "USA" not in df.columns:
        raise ValueError(f"No USA column in {path.name}. "
                         f"Found: {list(df.columns)[:15]}")

    s = pd.to_numeric(df["USA"], errors="coerce").dropna().rename("BAB")
    if freq == "M":
        s.index = s.index.to_period("M")
    return s                                    # AQR already reports decimals


# ======================================================================
# diagnostics
# ======================================================================
def check_scale(daily: pd.DataFrame, cols) -> None:
    """Catches a stray factor-of-100. Nothing else."""
    print("\nAnnualised vol from daily data (%):")
    for c in cols:
        if c in daily.columns and daily[c].notna().any():
            v = daily[c].dropna().std() * np.sqrt(252) * 100
            flag = "" if 1.0 < v < 60.0 else "   <-- CHECK SCALING"
            print(f"   {c:8s} {v:7.2f}{flag}")


def check_alignment(daily: pd.DataFrame, monthly: pd.DataFrame, cols) -> None:
    """
    Compounded-daily vs native-monthly. This is NOT a validity test of the
    replication: the two files are separately constructed, and for daily-
    rebalanced long-shorts they legitimately differ (MOM near 0.93 is normal,
    driven by the 2009 crash). What it DOES catch is a date shift. If a lagged
    correlation beats the contemporaneous one, the index is off by a month and
    everything downstream is wrong.
    """
    print("\nDaily-vs-monthly alignment (contemporaneous corr should dominate):")
    for c in cols:
        if c not in daily.columns or c not in monthly.columns:
            continue
        d = daily[c].dropna()
        if d.empty:
            continue
        comp = (1 + d).groupby(pd.Grouper(freq="ME")).prod() - 1
        comp.index = comp.index.to_period("M")
        both = pd.concat([comp.rename("d"), monthly[c].rename("m")],
                         axis=1).dropna()
        if len(both) < 24:
            continue

        r0 = both["d"].corr(both["m"])
        rp = both["d"].shift(1).corr(both["m"])
        rm = both["d"].shift(-1).corr(both["m"])
        worst = (both["d"] - both["m"]).abs().idxmax()

        if max(abs(rp), abs(rm)) >= abs(r0):
            flag = "   <-- DATE SHIFT, investigate"
        elif r0 < 0.99 and c not in DAILY_REBALANCED:
            flag = "   <-- unexpected for this factor"
        else:
            flag = ""
        print(f"   {c:8s} corr={r0:6.4f}  lag+1={rp:6.3f}  lag-1={rm:6.3f}  "
              f"worst={worst}{flag}")


def coverage(monthly: pd.DataFrame, cols) -> pd.Period:
    print("\nCoverage:")
    ends = []
    for c in cols:
        if c in monthly.columns and monthly[c].notna().any():
            s = monthly[c].dropna()
            print(f"   {c:8s} {s.index.min()} -> {s.index.max()}  N={len(s)}")
            ends.append(s.index.max())
    return min(ends)


# ======================================================================
def main():
    daily = pd.read_parquet(OUT / "factors_daily.parquet")
    monthly = pd.read_parquet(OUT / "factors_monthly.parquet")
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")

    # ---------------- global-q ----------------
    if GLOBALQ_LOCAL["daily"] and GLOBALQ_LOCAL["monthly"]:
        src_d, src_m = GLOBALQ_LOCAL["daily"], GLOBALQ_LOCAL["monthly"]
        print("using local global-q files")
    else:
        urls = globalq_urls()
        src_d, src_m = urls["daily"][0], urls["monthly"][0]
        print("global-q daily  :", src_d)
        print("global-q monthly:", src_m)

    qd = load_globalq(src_d, "D")
    time.sleep(2)                               # don't hammer Weebly
    qm = load_globalq(src_m, "M")

    daily = daily.drop(columns=[c for c in qd.columns if c in daily.columns])
    monthly = monthly.drop(columns=[c for c in qm.columns if c in monthly.columns])
    daily = daily.join(qd, how="outer")
    monthly = monthly.join(qm, how="outer")

    # ---------------- AQR BAB ----------------
    have_daily = BAB_DAILY_XLSX.exists()
    have_monthly = BAB_MONTHLY_XLSX.exists()

    if have_daily:
        bd = load_aqr_bab(BAB_DAILY_XLSX, "D")
        daily = daily.drop(columns=["BAB"], errors="ignore").join(bd, how="outer")
        print(f"BAB daily  : {bd.index.min().date()} -> {bd.index.max().date()}, "
              f"N={len(bd)}")
    else:
        print(f"!! {BAB_DAILY_XLSX.name} not found - no RV for BAB, "
              f"factor will be skipped in the replication.")

    if have_monthly:
        bm = load_aqr_bab(BAB_MONTHLY_XLSX, "M")
        monthly = monthly.drop(columns=["BAB"], errors="ignore").join(bm, how="outer")
        print(f"BAB monthly: {bm.index.min()} -> {bm.index.max()}, N={len(bm)}")
    elif have_daily:
        print("\n" + "!" * 70)
        print("Monthly BAB file missing. Compounding the daily series as a")
        print("fallback. BAB is a LEVERED long-short: compounded daily returns")
        print("are not the monthly-held factor, and the gap is largest in")
        print("exactly the high-volatility months this paper is about.")
        print("Download the monthly file before using these results.")
        print("!" * 70 + "\n")
        bd = daily["BAB"].dropna()
        bm = (1 + bd).groupby(pd.Grouper(freq="ME")).prod() - 1
        bm.index = bm.index.to_period("M")
        monthly = monthly.drop(columns=["BAB"], errors="ignore").join(
            bm.rename("BAB"), how="outer")

    daily = daily.sort_index()
    monthly = monthly.sort_index()

    # ---------------- write ----------------
    daily.to_parquet(OUT / "factors_daily.parquet")
    monthly.to_parquet(OUT / "factors_monthly.parquet")
    daily.to_csv(OUT / "factors_daily.csv")
    monthly.to_csv(OUT / "factors_monthly.csv")

    # ---------------- diagnostics ----------------
    common_end = coverage(monthly, FACTORS)
    check_scale(daily, FACTORS)
    check_alignment(daily, monthly, FACTORS)

    print(f"\nCommon end date across all factors: {common_end}")
    print("Truncate the main table here for a like-for-like comparison, "
          "or report N per row.")
    if monthly["MktRF"].dropna().index.min() > pd.Period("1926-07", "M"):
        print("\n!! MktRF starts after 1926-07. mm_data.py is dropping the "
              "months that exist only in the FF3 file - apply the "
              "merge_panels fix.")


if __name__ == "__main__":
    main()