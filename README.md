# Market Data Analysis

This Python project downloads recent daily stock prices for 10 large-cap stocks and compares them with the SPY S&P 500 proxy. It calculates correlations, betas, annualized volatility, and walk-forward ridge-regression predictions for the next-day SPY return.

> The forecast is a pipeline sanity check, not an investment or trading signal.

## What it creates

The script writes these files to `output/`:

- `summary.csv` - correlation, beta, R-squared, volatility, and return metrics
- `correlation_matrix.csv` - correlations among all symbols
- `predictions.csv` - walk-forward predicted and actual SPY returns
- `forecast_history.csv` - dated model forecasts and scoring metrics from each run
- `correlation_heatmap.png` - correlation heatmap
- `beta_vs_spy.png` - stock beta and correlation chart
- `prediction_vs_actual.png` - out-of-sample prediction chart
- `cache/<collection-date>/` - Alpha Vantage responses grouped by collection date

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- requests
- SciPy
- scikit-learn

An existing virtual environment is included at `.venv/`.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib requests scipy scikit-learn
```

If `.venv/` already exists, activate it and skip installation unless a package is missing:

```bash
source .venv/bin/activate
```

On macOS/Linux, command-line activation uses `source`. If you close or open a new terminal, activate the environment again.

## Run an offline test

Synthetic mode does not use the internet or an API key:

```bash
python market_analysis.py --synthetic
```

This is useful for confirming that the pipeline and charts work. The generated values are artificial.

## Run with real market data

1. Create a free Alpha Vantage API key at https://www.alphavantage.co/support/#api-key.
2. Set the key in your terminal:

```bash
export ALPHAVANTAGE_API_KEY="your_api_key"
```

3. Run the script:

```bash
python market_analysis.py
```

Alpha Vantage free-tier requests are rate-limited. The default run waits 15 seconds between calls, and successful responses are cached for the current day in `output/cache/`.

## Customize the symbols

Pass a comma-separated list with `--symbols`:

```bash
python market_analysis.py --symbols NVDA,MSFT,AAPL,AMZN,GOOGL,META,AVGO,TSLA,AMD,ORCL
python market_analysis.py --synthetic --symbols NVDA,MSFT,AAPL
```

SPY is always added automatically as the benchmark.

## Other options

```bash
python market_analysis.py --out results
python market_analysis.py --sleep 20
python market_analysis.py --help
```

- `--out` changes the output directory. It defaults to `output`.
- `--sleep` changes the delay in seconds between real-data API calls.
- `--synthetic` uses generated prices instead of Alpha Vantage.
- `--symbols` changes the analyzed stock list.

## Example workflow

```bash
cd /Users/annalu/Documents/GitHub/data-analyzer
source .venv/bin/activate
python market_analysis.py --synthetic
```

Then open the generated PNG files in `output/` and inspect the CSV files with a spreadsheet or Python.
