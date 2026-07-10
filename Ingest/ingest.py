"""
ingest.py — Exchange Margin pipeline, single entry point.

Steps:
  1. Fetch new ICE margin scanning CSVs (incremental, from last local file to today)
  2. Parse all local CSVs -> margin_scanning table (DuckDB)
  3. Fetch LSEG prices (GSCI + flat) for KC/CC/CT/SB/OJ, compute realised vol -> prices table
  4. ASOF-join margin + prices, compute VaR per lot -> margin_var table
  5. Fit per-market OLS (rv60/rv120) on margin-change events -> model_margin column
  6. Export margin_scanning / prices / margin_var to parquet in ../Database/

Usage:
    python ingest.py            # incremental (fetch new CSVs + upsert prices)
    python ingest.py --full     # refetch full CSV history + full price history
"""

import argparse
import csv
import io
import pathlib
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_csv"
DB_PATH = BASE_DIR / "margin.duckdb"
OUT_DIR = BASE_DIR.parent / "Database"
RAW_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

TARGET_CODES = {"CC", "KC", "SB", "CT", "OJ"}
TARGET_TIERS = {1, 2}

MARKET_RICS = {
    "KC": (".SPGSKCP", "KCc2"),
    "CC": (".SPGSCCP", "CCc2"),
    "CT": (".SPGSCTP", "CTv1"),
    "SB": (".SPGSSBP", "SBc1"),
    "OJ": (".SPGSOJP", "OJc2"),
}

LOT_MULTIPLIERS = {"KC": 375, "SB": 1120, "CT": 500, "CC": 10}
RV_WINDOWS = [20, 60, 120, 240, 500]
PRICE_START_FULL = "2015-01-01"
SQRT252 = np.sqrt(252)
MODEL_MARKETS = ["CC", "CT", "KC", "SB"]

BASE_URL = "https://www.ice.com/publicdocs/clear_us/irmParameters/ICUS_MARGIN_SCANNING_{}.CSV"


# ---------------------------------------------------------------------------
# 1. Fetch new margin CSVs
# ---------------------------------------------------------------------------
def fetch_new_csvs(full: bool, since_years: int | None = None) -> int:
    existing_dates = sorted(
        p.stem.rsplit("_", 1)[-1] for p in RAW_DIR.glob("ICUS_MARGIN_SCANNING_*.CSV")
    )
    if full or not existing_dates:
        start = date(2015, 1, 1)
    else:
        last = existing_dates[-1]
        start = date(int(last[:4]), int(last[4:6]), int(last[6:8])) + timedelta(days=1)
    end = date.today()

    if since_years is not None:
        floor = date(end.year - since_years, end.month, end.day)
        start = max(start, floor)

    if start > end:
        print("Fetch: already up to date.")
        return 0

    print(f"Fetching margin CSVs {start} -> {end} ...")
    fetched = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            date_str = d.strftime("%Y%m%d")
            out = RAW_DIR / f"ICUS_MARGIN_SCANNING_{date_str}.CSV"
            if not out.exists():
                url = BASE_URL.format(date_str)
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as r, open(out, "wb") as f:
                        f.write(r.read())
                    print(f"  saved {date_str}")
                    fetched += 1
                    time.sleep(1)
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        pass
                    elif e.code == 429:
                        print("  rate limited, waiting 30s...")
                        time.sleep(30)
                        continue
                    else:
                        print(f"  error {date_str}: {e}")
                except Exception as e:
                    print(f"  error {date_str}: {e}")
        d += timedelta(days=1)
    print(f"Fetch done: {fetched} new files.")
    return fetched


# ---------------------------------------------------------------------------
# 2. Build margin_scanning table
# ---------------------------------------------------------------------------
def parse_csv(path: pathlib.Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [{k.strip(): v.strip() for k, v in row.items()}
            for row in csv.DictReader(io.StringIO(raw))]


def normalise(row: dict, source_file: str) -> dict | None:
    code = row.get("Logical Commodity Code", "")
    if code not in TARGET_CODES:
        return None
    try:
        tier = int(row.get("Tier", ""))
    except ValueError:
        return None
    if tier not in TARGET_TIERS:
        return None
    return {
        "effective_date": row.get("Effective Date"),
        "market": code,
        "contract_name": row.get("Contract Name"),
        "currency": row.get("Currency"),
        "tier": tier,
        "new_scanning_range": row.get("New Scanning Range"),
        "previous_scanning_range": row.get("Previous Scanning Range"),
        "new_applied_margin_rate": row.get("New Applied Margin Rate"),
        "previous_applied_margin_rate": row.get("Previous Applied Margin Rate"),
        "percentage_change": row.get("Percentage Change"),
        "margin_units": row.get("Margin Units"),
        "multiplier": row.get("Multiplier"),
        "source_file": source_file,
    }


def build_margin_scanning(con: duckdb.DuckDBPyConnection, since_years: int | None = None) -> None:
    csv_files = sorted(RAW_DIR.glob("ICUS_MARGIN_SCANNING_*.CSV"))
    if since_years is not None:
        cutoff = date.today().replace(year=date.today().year - since_years).strftime("%Y%m%d")
        csv_files = [f for f in csv_files if f.stem.rsplit("_", 1)[-1] >= cutoff]
    print(f"Found {len(csv_files)} raw CSV files"
          f"{f' (last {since_years}y)' if since_years else ''}.")

    all_rows: list[dict] = []
    for f in csv_files:
        for r in parse_csv(f):
            n = normalise(r, f.name)
            if n:
                all_rows.append(n)
    print(f"Kept {len(all_rows)} rows (CC/KC/SB/CT/OJ, tiers 1-2).")

    if not all_rows:
        raise SystemExit("No margin rows parsed — check raw_csv/ contents.")

    con.execute("DROP TABLE IF EXISTS margin_scanning")
    con.execute("""
        CREATE TABLE margin_scanning (
            effective_date               VARCHAR,
            market                       VARCHAR,
            contract_name                VARCHAR,
            currency                     VARCHAR,
            tier                         INTEGER,
            new_scanning_range           VARCHAR,
            previous_scanning_range      VARCHAR,
            new_applied_margin_rate      VARCHAR,
            previous_applied_margin_rate VARCHAR,
            percentage_change            VARCHAR,
            margin_units                 VARCHAR,
            multiplier                   VARCHAR,
            source_file                  VARCHAR
        )
    """)
    con.executemany(
        "INSERT INTO margin_scanning VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [[r["effective_date"], r["market"], r["contract_name"], r["currency"],
          r["tier"], r["new_scanning_range"], r["previous_scanning_range"],
          r["new_applied_margin_rate"], r["previous_applied_margin_rate"],
          r["percentage_change"], r["margin_units"], r["multiplier"], r["source_file"]]
         for r in all_rows],
    )
    print(con.execute("""
        SELECT market, tier, COUNT(*) AS n FROM margin_scanning
        GROUP BY market, tier ORDER BY market, tier
    """).fetchdf().to_string(index=False))


# ---------------------------------------------------------------------------
# 3. Prices + realised vol (LSEG)
# ---------------------------------------------------------------------------
def compute_rv(prices: pd.Series) -> pd.DataFrame:
    log_ret = np.log(prices / prices.shift(1))
    rv = pd.DataFrame(index=prices.index)
    for w in RV_WINDOWS:
        rv[f"rv_{w}"] = log_ret.rolling(w).std() * np.sqrt(252) * 100
    return rv


def build_prices(con: duckdb.DuckDBPyConnection, full: bool, since_years: int | None = None) -> None:
    import lseg.data as ld

    print("Opening LSEG session...")
    ld.open_session()

    if since_years is not None:
        # fetch a couple of extra years so the rv_240/rv_500 windows have warm-up data
        buffered_years = since_years + 2
        price_start = date.today().replace(year=date.today().year - buffered_years).strftime("%Y-%m-%d")
    else:
        price_start = PRICE_START_FULL
    price_end = date.today().strftime("%Y-%m-%d")

    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date        DATE    NOT NULL,
            market      VARCHAR NOT NULL,
            gsci_close  DOUBLE,
            flat_close  DOUBLE,
            rv_20       DOUBLE,
            rv_60       DOUBLE,
            rv_120      DOUBLE,
            rv_240      DOUBLE,
            rv_500      DOUBLE,
            PRIMARY KEY (market, date)
        )
    """)
    if full:
        con.execute("DELETE FROM prices")

    for market, (gsci_ric, flat_ric) in MARKET_RICS.items():
        print(f"  {market}: GSCI={gsci_ric}  flat={flat_ric}")
        try:
            gdf = ld.get_history(universe=gsci_ric, fields=["TRDPRC_1"],
                                  interval="daily", start=price_start, end=price_end)
            fdf = ld.get_history(universe=flat_ric, fields=["TRDPRC_1"],
                                  interval="daily", start=price_start, end=price_end)
        except Exception as e:
            print(f"    error fetching {market}: {e}")
            continue
        time.sleep(0.5)

        gsci = gdf["TRDPRC_1"].dropna() if gdf is not None and not gdf.empty else pd.Series(dtype=float)
        flat = fdf["TRDPRC_1"].dropna() if fdf is not None and not fdf.empty else pd.Series(dtype=float)
        gsci.index = pd.to_datetime(gsci.index)
        flat.index = pd.to_datetime(flat.index)

        if gsci.empty:
            print(f"    [{market}] no GSCI data — skipping")
            continue

        rv = compute_rv(gsci)
        all_dates = gsci.index.union(flat.index)

        def _v(series, d):
            val = series.get(d)
            return float(val) if val is not None and not pd.isna(val) else None

        rows = [
            (str(d.date()), market, _v(gsci, d), _v(flat, d),
             _v(rv["rv_20"], d), _v(rv["rv_60"], d), _v(rv["rv_120"], d),
             _v(rv["rv_240"], d), _v(rv["rv_500"], d))
            for d in all_dates
        ]
        con.executemany("""
            INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT (market, date) DO UPDATE SET
                gsci_close = EXCLUDED.gsci_close,
                flat_close = EXCLUDED.flat_close,
                rv_20 = EXCLUDED.rv_20, rv_60 = EXCLUDED.rv_60,
                rv_120 = EXCLUDED.rv_120, rv_240 = EXCLUDED.rv_240,
                rv_500 = EXCLUDED.rv_500
        """, rows)
        print(f"    upserted {len(rows):,} rows "
              f"({str(all_dates.min().date())} -> {str(all_dates.max().date())})")


# ---------------------------------------------------------------------------
# 4. margin_var (ASOF join + VaR)
# ---------------------------------------------------------------------------
def build_margin_var(con: duckdb.DuckDBPyConnection) -> None:
    print("Building margin_var...")
    multiplier_case = "\n        ".join(
        f"WHEN '{mkt}' THEN {mult}" for mkt, mult in LOT_MULTIPLIERS.items()
    )
    con.execute("DROP TABLE IF EXISTS margin_var")
    con.execute(f"""
    CREATE TABLE margin_var AS
    WITH
    clean_margin AS (
        SELECT
            STRPTIME(effective_date, '%d-%b-%y')::DATE AS eff_date,
            market, tier,
            TRY_CAST(new_applied_margin_rate AS DOUBLE) AS initial_margin,
            market || CAST(tier AS VARCHAR) AS contract
        FROM margin_scanning
        WHERE market IN ({", ".join(f"'{m}'" for m in LOT_MULTIPLIERS)})
          AND tier IN (1, 2)
          AND new_applied_margin_rate IS NOT NULL
          AND new_applied_margin_rate <> ''
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY market, tier, STRPTIME(effective_date, '%d-%b-%y')::DATE
            ORDER BY effective_date DESC
        ) = 1
    ),
    prices_tiers AS (
        SELECT p.date, p.market, p.gsci_close, p.flat_close,
               p.rv_20, p.rv_60, p.rv_120, p.rv_240, p.rv_500,
               t.tier, t.contract
        FROM prices p
        JOIN (SELECT DISTINCT market, tier, contract FROM clean_margin) t
          ON p.market = t.market
        WHERE p.market IN ({", ".join(f"'{m}'" for m in LOT_MULTIPLIERS)})
    ),
    asof_joined AS (
        SELECT pt.date, pt.market, pt.contract, pt.gsci_close, pt.flat_close,
               pt.rv_20, pt.rv_60, pt.rv_120, pt.rv_240, pt.rv_500,
               m.initial_margin
        FROM prices_tiers pt
        ASOF JOIN clean_margin m
          ON pt.market = m.market AND pt.tier = m.tier AND pt.date >= m.eff_date
    )
    SELECT
        date, market, contract, gsci_close, flat_close,
        rv_20, rv_60, rv_120, rv_240, rv_500,
        ROUND(initial_margin)::INTEGER AS initial_margin,
        ROUND(flat_close * CASE market {multiplier_case} ELSE NULL END)::INTEGER AS nominal_value,
        ROUND(flat_close * CASE market {multiplier_case} ELSE NULL END * rv_20  / 100)::INTEGER AS var_20,
        ROUND(flat_close * CASE market {multiplier_case} ELSE NULL END * rv_60  / 100)::INTEGER AS var_60,
        ROUND(flat_close * CASE market {multiplier_case} ELSE NULL END * rv_120 / 100)::INTEGER AS var_120,
        ROUND(flat_close * CASE market {multiplier_case} ELSE NULL END * rv_240 / 100)::INTEGER AS var_240,
        ROUND(flat_close * CASE market {multiplier_case} ELSE NULL END * rv_500 / 100)::INTEGER AS var_500
    FROM asof_joined
    ORDER BY date, market, contract
    """)
    n = con.execute("SELECT COUNT(*) FROM margin_var").fetchone()[0]
    print(f"  margin_var: {n:,} rows")


# ---------------------------------------------------------------------------
# 5. Model margin (per-market OLS on rv60/rv120), written back into margin_var
# ---------------------------------------------------------------------------
def build_model_margin(con: duckdb.DuckDBPyConnection) -> None:
    print("Fitting per-market model_margin...")
    events = con.execute("""
        WITH clean AS (
            SELECT
                STRPTIME(ms.effective_date, '%d-%b-%y')::DATE AS event_date,
                ms.market, ms.tier,
                ROUND(TRY_CAST(ms.new_applied_margin_rate AS DOUBLE)) AS initial_margin
            FROM margin_scanning ms
            WHERE ms.market IN ('CC', 'KC', 'SB', 'CT')
              AND ms.tier IN (1, 2)
              AND ms.new_applied_margin_rate IS NOT NULL
              AND ms.new_applied_margin_rate <> ''
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY ms.market, ms.tier,
                             STRPTIME(ms.effective_date, '%d-%b-%y')::DATE
                ORDER BY ms.effective_date DESC
            ) = 1
        )
        SELECT c.event_date, c.market, c.tier, c.initial_margin,
               ROUND(mv.var_60  / ?) AS var_daily_60,
               ROUND(mv.var_120 / ?) AS var_daily_120
        FROM clean c
        ASOF JOIN margin_var mv
          ON c.market = mv.market
         AND c.tier = CAST(SUBSTR(mv.contract, -1) AS INTEGER)
         AND c.event_date >= mv.date
        WHERE c.initial_margin > 0 AND mv.var_60 IS NOT NULL AND mv.var_120 IS NOT NULL
    """, [SQRT252, SQRT252]).fetchdf()

    coeffs = {}
    for market in MODEL_MARKETS:
        sub = events[events.market == market].copy()
        if sub.empty:
            continue
        coeffs[market] = {}
        for w in [60, 120]:
            mod = smf.ols(f"initial_margin ~ var_daily_{w}", data=sub).fit()
            coeffs[market][w] = (mod.params["Intercept"], mod.params[f"var_daily_{w}"])

    mv = con.execute("""
        SELECT date, market, contract, var_60, var_120 FROM margin_var
    """).fetchdf()
    mv["var_daily_60"] = mv["var_60"] / SQRT252
    mv["var_daily_120"] = mv["var_120"] / SQRT252

    def model_margin(row):
        mkt = row["market"]
        if mkt not in coeffs:
            return pd.NA
        v60, v120 = row["var_daily_60"], row["var_daily_120"]
        if pd.isna(v60) or pd.isna(v120):
            return pd.NA
        i60, s60 = coeffs[mkt][60]
        i120, s120 = coeffs[mkt][120]
        return int(round(((i60 + s60 * v60) + (i120 + s120 * v120)) / 2))

    mv["model_margin"] = mv.apply(model_margin, axis=1).astype("Int64")

    con.execute("ALTER TABLE margin_var DROP COLUMN IF EXISTS model_margin")
    con.execute("ALTER TABLE margin_var ADD COLUMN model_margin INTEGER")
    con.register("_model_df", mv[["date", "market", "contract", "model_margin"]])
    con.execute("""
        UPDATE margin_var SET model_margin = t.model_margin
        FROM _model_df t
        WHERE margin_var.date = t.date AND margin_var.market = t.market
          AND margin_var.contract = t.contract
    """)
    print("  model_margin written.")


# ---------------------------------------------------------------------------
# 6. Export to parquet
# ---------------------------------------------------------------------------
def export_parquet(con: duckdb.DuckDBPyConnection) -> None:
    for table in ["margin_scanning", "prices", "margin_var"]:
        out = OUT_DIR / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{out}' (FORMAT PARQUET)")
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:20s} -> {out.name}  ({n:,} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="full rebuild (CSVs + prices)")
    parser.add_argument("--years", type=int, default=None,
                         help="only backfill margin CSVs for the last N years (speeds up --full)")
    parser.add_argument("--skip-fetch", action="store_true", help="skip fetching new CSVs")
    parser.add_argument("--skip-prices", action="store_true", help="skip LSEG price refresh")
    args = parser.parse_args()

    if not args.skip_fetch:
        fetch_new_csvs(full=args.full, since_years=args.years)

    con = duckdb.connect(str(DB_PATH))
    build_margin_scanning(con, since_years=args.years)

    if not args.skip_prices:
        build_prices(con, full=args.full, since_years=args.years)

    build_margin_var(con)
    build_model_margin(con)

    print("\nExporting to parquet...")
    export_parquet(con)
    con.close()
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
