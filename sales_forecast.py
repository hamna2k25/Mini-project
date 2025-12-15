#!/usr/bin/env python3
"""
Sales Forecasting Dashboard - CLI tool

Usage examples:
  python sales_forecast.py --input sample_data.csv --method linear --periods 3 --all
  python sales_forecast.py -i sample_data.csv -m arima -p 2 -P PROD_A --save-charts
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional dependency; ARIMA is used when requested and available
try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

from sklearn.linear_model import LinearRegression  # type: ignore

CHARTS_DIR_DEFAULT = Path("charts")
OUTPUT_DIR_DEFAULT = Path("output")


def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    required = {"Date", "ProductID", "SalesAmount"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Input CSV must contain columns: {required}. Found: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Drop rows with invalid dates or missing ProductID
    df = df.dropna(subset=["Date", "ProductID"])
    # Coerce sales to numeric and fill missing with 0
    df["SalesAmount"] = pd.to_numeric(df["SalesAmount"], errors="coerce").fillna(0.0)
    # Sort
    df = df.sort_values(["ProductID", "Date"])
    return df


def aggregate_monthly(df: pd.DataFrame, product_id: Optional[str] = None) -> pd.Series:
    df = df.copy()
    if product_id is not None:
        df = df[df["ProductID"].astype(str) == str(product_id)]
        if df.empty:
            raise ValueError(f"No data for product '{product_id}'")
    # Group by month (end of month)
    df.set_index("Date", inplace=True)
    monthly = df["SalesAmount"].resample("M").sum()
    # Fill missing months with 0
    monthly = monthly.asfreq("M", fill_value=0.0)
    return monthly


def forecast_linear(series: pd.Series, periods: int) -> pd.Series:
    """
    Simple linear regression on time index to forecast next `periods` months.
    series: pandas Series indexed by Timestamp (monthly frequency)
    returns: pandas Series of predicted values indexed by future monthly timestamps
    """
    if len(series.dropna()) < 2:
        raise ValueError("Not enough data points for linear forecasting (need >=2).")
    # Convert timestamps to integer feature (months since epoch)
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    preds = model.predict(future_X)
    last_index = series.index[-1]
    future_index = pd.date_range(start=last_index + pd.offsets.MonthBegin(1), periods=periods, freq="M")
    return pd.Series(data=np.maximum(preds, 0.0), index=future_index)


def forecast_arima(series: pd.Series, periods: int) -> pd.Series:
    """
    Forecast using ARIMA if statsmodels is available.
    Falls back to linear if ARIMA is not available or fails.
    """
    if not _HAS_ARIMA:
        raise RuntimeError("statsmodels is not installed; ARIMA is unavailable.")
    # If too short, raise to let caller fallback
    if len(series.dropna()) < 6:
        raise ValueError("Not enough data points for ARIMA forecasting (recommend >=6).")
    try:
        # Simple ARIMA(1,1,1) - can be improved
        model = ARIMA(series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit()
        fc = fitted.get_forecast(steps=periods)
        idx = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="M")
        preds = fc.predicted_mean
        preds = pd.Series(data=np.maximum(preds.values, 0.0), index=idx)
        return preds
    except Exception as e:
        raise RuntimeError(f"ARIMA forecasting failed: {e}")


def plot_history_and_forecast(history: pd.Series, forecast: pd.Series, product_id: str, charts_dir: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(history.index, history.values, label="Historical Sales", marker="o")
    plt.plot(forecast.index, forecast.values, label="Forecasted Sales", marker="o", linestyle="--")
    plt.title(f"Sales Forecast for Product {product_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales Amount")
    plt.legend()
    plt.grid(True)
    fname = charts_dir / f"{product_id}_forecast.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    # Also save a history-only chart
    plt.figure(figsize=(10, 4))
    plt.plot(history.index, history.values, label="Historical Sales", marker="o")
    plt.title(f"Historical Sales for Product {product_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales Amount")
    plt.grid(True)
    fname2 = charts_dir / f"{product_id}_history.png"
    plt.tight_layout()
    plt.savefig(fname2)
    plt.close()
    return fname, fname2


def build_forecast_for_product(series: pd.Series, product_id: str, method: str, periods: int, charts_dir: Path) -> pd.DataFrame:
    # Choose method
    forecast_series = None
    if method == "arima":
        try:
            forecast_series = forecast_arima(series, periods)
        except Exception as e:
            print(f"[warning] ARIMA failed for product {product_id}: {e}. Falling back to linear.")
            forecast_series = forecast_linear(series, periods)
    else:
        forecast_series = forecast_linear(series, periods)

    # Plot
    try:
        plot_history_and_forecast(series, forecast_series, product_id, charts_dir)
    except Exception as e:
        print(f"[warning] Plotting failed for product {product_id}: {e}")

    # Build a dataframe for output rows
    records = []
    for date, value in forecast_series.items():
        records.append({"Date": date.strftime("%Y-%m-%d"), "ProductID": product_id, "PredictedSales": float(round(value, 2))})
    return pd.DataFrame.from_records(records)


def run_cli(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ensure_dirs(CHARTS_DIR_DEFAULT, OUTPUT_DIR_DEFAULT)

    df = load_data(str(input_path))
    df = clean_data(df)

    products = []
    if args.all:
        products = sorted(df["ProductID"].astype(str).unique().tolist())
    elif args.product:
        products = [str(args.product)]
    else:
        # Default: all
        products = sorted(df["ProductID"].astype(str).unique().tolist())

    all_forecasts = []
    for pid in products:
        try:
            monthly = aggregate_monthly(df, pid)
            # If series has trailing zeros (no recent sales) it's still ok
            forecast_df = build_forecast_for_product(monthly, pid, args.method, args.periods, CHARTS_DIR_DEFAULT)
            all_forecasts.append(forecast_df)
            print(f"[ok] Forecast completed for product {pid}")
        except Exception as e:
            print(f"[error] Skipping product {pid}: {e}")

    if not all_forecasts:
        print("No forecasts were generated.")
        return

    result_df = pd.concat(all_forecasts, ignore_index=True)
    output_file = OUTPUT_DIR_DEFAULT / f"forecast_{args.method}_{args.periods}m.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Forecast CSV saved to: {output_file}")
    print(f"Charts saved to: {CHARTS_DIR_DEFAULT.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sales Forecasting Dashboard (CLI)")
    p.add_argument("-i", "--input", required=True, help="Path to input CSV with columns: Date, ProductID, SalesAmount")
    p.add_argument("-P", "--product", help="ProductID to forecast (default: all products)")
    p.add_argument("-a", "--all", action="store_true", help="Forecast for all products (overrides --product)")
    p.add_argument("-m", "--method", choices=["linear", "arima"], default="linear", help="Forecast method (default: linear). ARIMA requires statsmodels.")
    p.add_argument("-p", "--periods", type=int, default=3, help="Number of future months to predict (default: 3)")
    p.add_argument("--save-charts", action="store_true", help="Save charts to charts/ folder (default: saved anyway)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(args)