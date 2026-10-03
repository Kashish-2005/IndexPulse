# Enhanced Indexing Architecture & Contract

## System Specifications
- Universe: Nifty 50 constituents via yfinance (`.NS` tickers)
- Rebalancing Frequency: Month-end (`ME`)
- Turnover Definition: `0.5 * sum(|w_new - w_old|)`
- Execution: Trades execute at close of `t+1` based on signals computed at close of `t`

## Function Signatures

### 1. Data (`src/eindex/data.py`)
- `load_prices(tickers: list[str], start: str) -> pd.DataFrame`
- `load_market_caps(tickers: list[str]) -> pd.Series`
- `load_sectors(tickers: list[str]) -> pd.Series`
- `build_benchmark_weights(prices: pd.DataFrame, shares: pd.Series, dates: list[pd.Timestamp]) -> pd.DataFrame`

### 2. Alpha Signals (`src/eindex/signals.py`)
- `compute_alpha(prices: pd.DataFrame, date: pd.Timestamp, cfg: Config) -> pd.Series`

### 3. Risk Model (`src/eindex/risk.py`)
- `ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame`
- `psd_factor(cov: pd.DataFrame) -> np.ndarray`

### 4. Optimizer (`src/eindex/optimizer.py`)
- `optimize_portfolio(alpha: pd.Series, cov_annual: pd.DataFrame, w_bench: pd.Series, w_prev: pd.Series, sectors: pd.Series, cfg: Config) -> tuple[pd.Series, dict]`

### 5. Backtest (`src/eindex/backtest.py`)
- `run_backtest(prices: pd.DataFrame, benchmark_weights: pd.DataFrame, sectors: pd.Series, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]`

### 6. Analytics (`src/eindex/analytics.py`)
- `compute_metrics(returns_df: pd.DataFrame, weights_df: pd.DataFrame, cfg: Config) -> dict`

### 7. Pipeline (`src/eindex/pipeline.py`)
- `run_all(config_path: str = "config.yaml") -> None`
  - Output files:
    - `results/returns.csv` (columns: `date`, `portfolio`, `benchmark`)
    - `results/weights.csv` (columns: `date`, `ticker`, `sector`, `weight`, `bench_weight`)
    - `results/latest_trades.csv` (columns: `ticker`, `current_weight`, `target_weight`, `trade`)
    - `results/summary.json`