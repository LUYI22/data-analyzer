#!/usr/bin/env python3
"""
Correlation + next-day prediction for 10 large caps vs the S&P 500 (SPY proxy),
using the latest 100 daily bars from Alpha Vantage (TIME_SERIES_DAILY, compact).

Usage:
    export ALPHAVANTAGE_API_KEY=your_key
    python3 market_analysis.py                      # real data
    python3 market_analysis.py --synthetic          # pipeline test, no API calls
    python3 market_analysis.py --symbols NVDA,MSFT,AAPL,AMZN,GOOGL,META,AVGO,TSLA,AMD,ORCL

Outputs go to ./output/: summary.csv, correlation_matrix.csv, predictions.csv, charts (png).
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import binomtest
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BENCH = "SPY"  # ETF proxy for the S&P 500 (the index endpoint is premium)
DEFAULT_SYMBOLS = "NVDA,MSFT,AAPL,AMZN,GOOGL,META,AVGO,TSLA,AMD,ORCL"
N_POINTS = 100
MIN_TRAIN = 60
URL = "https://www.alphavantage.co/query"

INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#e4e7eb"
BLUE = "#2f6db5"
ORANGE = "#d9822b"


# ----------------------------------------------------------------------------- data
def fetch_daily(symbol: str, key: str, cache_dir: Path) -> pd.Series:
    """Latest 100 daily closes (raw, as traded). Cached per symbol per day."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{symbol}_{dt.date.today().isoformat()}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        r = requests.get(
            URL,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": key,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if "Time Series (Daily)" not in data:
            msg = data.get("Error Message") or data.get("Note") or data.get("Information") or str(data)[:200]
            raise RuntimeError(f"{symbol}: {msg}")
        cache.write_text(json.dumps(data))
    ts = data["Time Series (Daily)"]
    s = pd.Series({pd.Timestamp(d): float(v["4. close"]) for d, v in ts.items()}).sort_index()
    return s.rename(symbol)


def synthetic_prices(symbols, n=N_POINTS, seed=7) -> pd.DataFrame:
    """Fake correlated prices so the pipeline can be tested offline."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    mkt = rng.normal(0.0004, 0.009, n)
    out = {BENCH: 450 * np.exp(np.cumsum(mkt))}
    for s in symbols:
        beta = rng.uniform(0.6, 1.8)
        idio = rng.normal(0, rng.uniform(0.008, 0.02), n)
        out[s] = 100 * np.exp(np.cumsum(beta * mkt + idio))
    return pd.DataFrame(out, index=idx)


def load_prices(symbols, key, sleep, cache_dir) -> pd.DataFrame:
    cols = []
    for i, sym in enumerate([BENCH] + list(symbols)):
        cols.append(fetch_daily(sym, key, cache_dir))
        print(f"  fetched {sym} ({len(cols[-1])} bars)")
        if i < len(symbols):
            time.sleep(sleep)
    return pd.concat(cols, axis=1).dropna().tail(N_POINTS)


# ----------------------------------------------------------------------------- analysis
def correlation_table(rets: pd.DataFrame) -> pd.DataFrame:
    stocks = [c for c in rets.columns if c != BENCH]
    mkt = rets[BENCH]
    rows = []
    for s in stocks:
        corr = rets[s].corr(mkt)
        beta = rets[s].cov(mkt) / mkt.var()
        rows.append(
            {
                "symbol": s,
                "corr_with_SPY": corr,
                "beta": beta,
                "R2": corr**2,
                "ann_vol": rets[s].std() * np.sqrt(252),
                # does today's move in the stock say anything about SPY tomorrow?
                "corr_stock_t_vs_SPY_t+1": rets[s].corr(mkt.shift(-1)),
                "total_return_%": (np.exp(rets[s].sum()) - 1) * 100,
            }
        )
    return pd.DataFrame(rows).set_index("symbol").sort_values("corr_with_SPY", ascending=False)


def walk_forward(rets: pd.DataFrame):
    """Predict SPY's NEXT-day log return from today's returns of the 10 stocks + SPY.
    Expanding-window ridge, refit every day; baseline = expanding mean of past SPY returns."""
    X = rets.copy()
    y = rets[BENCH].shift(-1)
    data = pd.concat([X, y.rename("target")], axis=1).dropna()
    Xv, yv = data.drop(columns="target").values, data["target"].values
    preds, base, dates = [], [], []
    for i in range(MIN_TRAIN, len(data)):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 25)))
        model.fit(Xv[:i], yv[:i])
        preds.append(model.predict(Xv[i : i + 1])[0])
        base.append(yv[:i].mean())
        dates.append(data.index[i])
    res = pd.DataFrame({"predicted": preds, "baseline": base, "actual": yv[MIN_TRAIN:]}, index=dates)
    # final model on everything -> forecast for the next trading day
    final = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 25))).fit(Xv, yv)
    next_pred = final.predict(rets.iloc[[-1]].values)[0]
    coefs = pd.Series(
        final[-1].coef_ / final[0].scale_, index=rets.columns, name="ridge_coef"
    ).sort_values(key=np.abs, ascending=False)
    return res, next_pred, coefs, final[-1].alpha_


def score(res: pd.DataFrame) -> dict:
    rmse = lambda p: float(np.sqrt(np.mean((res["actual"] - p) ** 2)))
    hits = int((np.sign(res["predicted"]) == np.sign(res["actual"])).sum())
    n = len(res)
    return {
        "test_days": n,
        "rmse_model": rmse(res["predicted"]),
        "rmse_baseline": rmse(res["baseline"]),
        "directional_accuracy": hits / n,
        "binom_p_vs_coinflip": binomtest(hits, n, 0.5).pvalue,
        "corr_pred_vs_actual": float(res["predicted"].corr(res["actual"])),
    }


# ----------------------------------------------------------------------------- charts
def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def chart_heatmap(corr: pd.DataFrame, path: Path):
    order = corr[BENCH].drop(BENCH).sort_values(ascending=False).index.tolist()
    cols = [BENCH] + order
    m = corr.loc[cols, cols]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_title("Daily-return correlations (last 100 trading days)", loc="left", color=INK, fontsize=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_beta(table: pd.DataFrame, path: Path):
    t = table.sort_values("beta")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(t.index, t["beta"], color=BLUE, height=0.6)
    ax.axvline(1, color=MUTED, linewidth=1, linestyle="--")
    ax.text(1.01, -0.7, "market (β = 1)", color=MUTED, fontsize=8)
    for y, (b, c) in enumerate(zip(t["beta"], t["corr_with_SPY"])):
        ax.text(b + 0.02, y, f"β {b:.2f} · ρ {c:.2f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, t["beta"].max() * 1.3)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title("Beta to the S&P (SPY) and correlation, last 100 days", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_pred(res: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(res.index, res["actual"] * 100, color=MUTED, linewidth=1.5)
    ax.plot(res.index, res["predicted"] * 100, color=ORANGE, linewidth=2)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.text(res.index[-1], res["actual"].iloc[-1] * 100, "  actual", color=MUTED, fontsize=9, va="center")
    ax.text(res.index[-1], res["predicted"].iloc[-1] * 100, "  predicted", color=ORANGE, fontsize=9, va="center")
    style(ax)
    ax.set_ylabel("next-day SPY log return (%)", color=MUTED, fontsize=9)
    ax.set_title("Walk-forward prediction vs actual (out-of-sample days only)", loc="left", color=INK, fontsize=12)
    ax.set_xlim(res.index[0], res.index[-1] + pd.Timedelta(days=8))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS, help="comma-separated, 10 large caps")
    ap.add_argument("--synthetic", action="store_true", help="offline test with fake data")
    ap.add_argument("--sleep", type=float, default=15.0, help="seconds between API calls (rate limits)")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.synthetic:
        print("Using SYNTHETIC data (pipeline test only; numbers are meaningless).")
        prices = synthetic_prices(symbols)
    else:
        key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not key:
            sys.exit("Set ALPHAVANTAGE_API_KEY (free key: https://www.alphavantage.co/support/#api-key)")
        print("Fetching daily bars from Alpha Vantage...")
        prices = load_prices(symbols, key, args.sleep, out / "cache")

    rets = np.log(prices).diff().dropna()
    print(f"\n{len(prices)} prices -> {len(rets)} daily returns "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    table = correlation_table(rets)
    corr = rets.corr()
    table.to_csv(out / "summary.csv")
    corr.to_csv(out / "correlation_matrix.csv")
    print("\nCorrelation with the S&P (ranked):")
    print(table.round(3).to_string())

    res, next_pred, coefs, alpha = walk_forward(rets)
    res.to_csv(out / "predictions.csv")
    sc = score(res)
    print(f"\nWalk-forward SPY next-day prediction (ridge alpha={alpha:.1f}, {sc['test_days']} out-of-sample days):")
    for k, v in sc.items():
        print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    verdict = "beats" if sc["rmse_model"] < sc["rmse_baseline"] else "does NOT beat"
    print(f"  -> model {verdict} the 'predict the average' baseline on RMSE.")
    print(f"\nLargest ridge coefficients (per 1.00 of daily log return):\n{coefs.head(5).round(4).to_string()}")
    print(f"\nModel's forecast for SPY's next trading day: {next_pred * 100:+.3f}% "
          "(treat as a sanity-check output, not a trading signal)")

    chart_heatmap(corr, out / "correlation_heatmap.png")
    chart_beta(table, out / "beta_vs_spy.png")
    chart_pred(res, out / "prediction_vs_actual.png")
    print(f"\nSaved tables and charts to {out.resolve()}")


if __name__ == "__main__":
    main()