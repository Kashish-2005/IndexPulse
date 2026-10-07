# Enhanced Indexing: Portfolio Optimization

[![CI](https://github.com/REPLACE_WITH_YOUR_USERNAME/enhanced-index/actions/workflows/ci.yml/badge.svg)](https://github.com/REPLACE_WITH_YOUR_USERNAME/enhanced-index/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An institutional-grade quantitative research framework implementing **Enhanced Indexing** on India's flagship **Nifty 50** universe. The strategy aims to harvest risk-premia and generate systematic excess returns (alpha) while maintaining strict tracking error and active-risk constraints relative to the market-cap benchmark.

> **Disclaimer:** This project is strictly for academic research and educational purposes. It does not constitute financial or investment advice.

---

## 🏛️ System Architecture

The pipeline follows a modular, schedule-driven execution model with zero external database dependencies:

```mermaid
flowchart TD
    subgraph Data Layer
        YF[yfinance API] -->|Prices & Shares| DL[Data Ingestion & Cache]
        CSV[(static_sectors.csv)] -->|Fallback Sectors| DL
    end

    subgraph Quant Engine
        DL -->|Adjusted Close| SIG[Alpha Signal Engine\nMomentum + Low-Vol + Reversal]
        DL -->|Daily Returns| RSK[Risk Model\nLedoit-Wolf Shrinkage Covariance]
        DL -->|Market Cap Proxy| BMK[Benchmark Weight Construction]
        
        SIG --> OPT[Convex Optimization\nCVXPY + CLARABEL]
        RSK --> OPT
        BMK --> OPT
    end

    subgraph Execution & Tracking
        OPT -->|Target Weights| BT[Walk-Forward Backtest\nt+1 Close Execution & Friction]
        BT --> OUT[Results Artifacts\nreturns.csv | weights.csv | latest_trades.csv]
    end

    subgraph Deployment
        OUT --> GH[GitHub Repository]
        GH -->|Automated Read| DSH[Streamlit Community Cloud]
    end