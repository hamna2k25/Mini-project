# Sales Forecasting Dashboard

A learning-focused CLI tool to predict future sales from past sales data and visualize trends. Intended for small business owners and students exploring applied AI/business analytics.

Features
- Load CSV with columns: `Date`, `ProductID`, `SalesAmount`.
- Basic data cleaning (date parsing, missing values).
- Generate simple time-series forecasts using:
  - Linear regression (default)
  - ARIMA (optional; requires `statsmodels`)
- Visualize results:
  - Line chart of historical sales.
  - Line chart showing future predicted sales for the next 1–3 months.
- Save forecast CSV and charts as PNG files.

Repo structure
- `sales_forecast.py` — main CLI script.
- `README.md` — this file.
- `sample_data.csv` — small sample dataset (~50 rows).
- `charts/` — directory where generated charts are saved.
- `output/` — directory where forecast CSV is saved.
- `tests/` — optional unit tests (requires `pytest`).
- `LICENSE` — MIT license.

Quickstart / Requirements
- Python 3.8+
- Recommended packages:
  - pandas, numpy, matplotlib, scikit-learn
- Optional (for ARIMA): statsmodels
- Optional (for tests): pytest

Install packages (example)
```bash
pip install pandas numpy matplotlib scikit-learn
# For ARIMA:
pip install statsmodels
# For tests:
pip install pytest
```

Usage examples
- Forecast all products for next 3 months using linear regression:
```bash
python sales_forecast.py --input sample_data.csv --method linear --periods 3 --all
```
- Forecast a single product (`PROD_A`) for next 2 months using ARIMA:
```bash
python sales_forecast.py -i sample_data.csv -m arima -p 2 -P PROD_A
```

What the script does
1. Loads CSV and checks required columns (`Date`, `ProductID`, `SalesAmount`).
2. Cleans data:
   - Converts `Date` to datetime, drops rows with invalid dates or missing ProductID.
   - Coerces `SalesAmount` to numeric and fills missing values with 0.
3. Aggregates sales to monthly totals per product.
4. Forecasts next N months using the chosen method:
   - `linear`: fits a linear regression on the monthly time index.
   - `arima`: fits an ARIMA(1,1,1) model (if `statsmodels` present). Falls back to linear if ARIMA fails.
5. Saves:
   - Forecast CSV at `output/forecast_{method}_{periods}m.csv`.
   - Charts for each product at `charts/{ProductID}_forecast.png` and `{ProductID}_history.png`.

Output CSV schema
- `Date` (YYYY-MM-DD) — predicted month end date
- `ProductID`
- `PredictedSales` (float, rounded to 2 decimals)

Example command & expected files
```bash
python sales_forecast.py -i sample_data.csv -m linear -p 3 --all
```
This will create:
- `output/forecast_linear_3m.csv`
- `charts/{ProductID}_forecast.png` for each product
- `charts/{ProductID}_history.png` for each product

Forecast method notes
- Linear regression is simple and robust for a first pass.
- ARIMA can capture more complex temporal patterns but needs more data and the `statsmodels` package.
- This tool is intended for learning and demonstration; consider more advanced models (Prophet, SARIMA, LSTM) for production.

Testing
- A simple unit test is included in `tests/test_forecast.py` to validate CSV loading and the linear forecast function.
- Run tests with:
```bash
pytest
```

Sample data
- `sample_data.csv` contains ~50 rows of example sales for a few products to try the tool quickly.

License
- MIT License. See `LICENSE` file.

Acknowledgements
- Built as an educational example to demonstrate applied forecasting with Python for business users and learners.
