"""
mm_data.py
Downloads Fama-French daily + monthly factors and Momentum from Ken French's
data library. Saves tidy parquet/csv to ./data/.

pip install pandas numpy requests pyarrow
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
OUT = Path("data")
OUT.mkdir(exist_ok=True)

FILES = {
    "ff3_daily":   "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff3_monthly": "F-F_Research_Data_Factors_CSV.zip",
    "ff5_daily":   "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom_daily":   "F-F_Momentum_Factor_daily_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (research replication)"}


def fetch_zip_csv(fname: str) -> str:
    """Download a French zip and return the raw text of the single CSV inside."""
    r = requests.get(BASE + fname, headers=HEADERS, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    return z.read(name).decode("latin-1")


def parse_french(text: str, freq: str) -> pd.DataFrame:
    """
    French CSVs have a preamble, then a date-indexed block, then often an
    annual block and/or copyright footer. We keep only rows whose first field
    is a valid date of the expected width (8 for daily, 6 for monthly).
    """
    width = 8 if freq == "D" else 6
    rows, header = [], None

    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        key = parts[0]
        if key.isdigit() and len(key) == width:
            rows.append(parts)
        elif header is None and key == "" and any(parts[1:]):
            # the column-name line: leading empty field
            header = parts
        elif key.isdigit() and len(key) == 4 and freq == "M":
            # annual block has started -> stop
            break

    if header is None:
        raise ValueError("Could not locate header row")

    cols = ["date"] + [c for c in header[1:] if c != ""]
    df = pd.DataFrame(rows).iloc[:, : len(cols)]
    df.columns = cols

    fmt = "%Y%m%d" if freq == "D" else "%Y%m"
    df["date"] = pd.to_datetime(df["date"], format=fmt)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # French codes missing as -99.99 / -999
    df = df.replace([-99.99, -999, -99.99], np.nan)

    # percent -> decimal
    for c in cols[1:]:
        df[c] = df[c] / 100.0

    df = df.set_index("date").sort_index()
    df.columns = [c.replace("-", "").replace(" ", "") for c in df.columns]
    return df


def month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    """Snap monthly French dates (YYYYMM -> 1st) to month-end periods."""
    out = df.copy()
    out.index = out.index.to_period("M")
    return out


def main():
    store = {}
    for key, fname in FILES.items():
        freq = "D" if key.endswith("daily") else "M"
        print(f"downloading {fname} ...")
        df = parse_french(fetch_zip_csv(fname), freq)
        if freq == "M":
            df = month_end_index(df)
        store[key] = df
        print(f"   {key}: {df.index.min()} -> {df.index.max()}, {df.shape}")

    def merge_panels(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
        """Union the indices FIRST, then backfill. Assigning into primary's
        existing index silently drops rows that exist only in secondary."""
        idx = primary.index.union(secondary.index)
        p, s = primary.reindex(idx), secondary.reindex(idx)
        for c in s.columns:
            p[c] = p[c].combine_first(s[c]) if c in p.columns else s[c]
        return p.sort_index()

    # FF3 is the long series (1926-07); FF5 and Mom layer on top
    d = merge_panels(store["ff3_daily"], store["ff5_daily"])
    d = merge_panels(d, store["mom_daily"].rename(columns={"Mom": "MOM"}))

    m = merge_panels(store["ff3_monthly"], store["ff5_monthly"])
    m = merge_panels(m, store["mom_monthly"].rename(columns={"Mom": "MOM"}))

    d.to_parquet(OUT / "factors_daily.parquet")
    m.to_parquet(OUT / "factors_monthly.parquet")
    d.to_csv(OUT / "factors_daily.csv")
    m.to_csv(OUT / "factors_monthly.csv")

    print("\nDaily panel:", d.shape, d.index.min().date(), "->", d.index.max().date())
    print("Monthly panel:", m.shape, m.index.min(), "->", m.index.max())
    print("\nNon-null counts (monthly):")
    print(m.notna().sum())


if __name__ == "__main__":
    main()