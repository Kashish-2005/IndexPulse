"""Portfolio optimizer: convex optimization with fallback ladder."""

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from eindex.risk import psd_factor

logger = logging.getLogger(__name__)

_SOLVERS = [cp.CLARABEL, cp.OSQP, cp.SCS]
_CLIP_TOL = 1e-9
_CONSTRAINT_TOL = 1e-6


def _align_inputs(
    alpha: pd.Series,
    cov_annual: pd.DataFrame,
    w_bench: pd.Series,
    w_prev: pd.Series,
    sectors: pd.Series,
) -> list[str]:
    """Return common ticker list; raise on any mismatch."""
    idx_alpha = set(alpha.index)
    idx_cov = set(cov_annual.index)
    idx_bench = set(w_bench.index)
    idx_prev = set(w_prev.index)
    idx_sec = set(sectors.index)

    if not (idx_alpha == idx_cov == idx_bench == idx_prev == idx_sec):
        raise ValueError(
            "Index mismatch across inputs. "
            f"alpha={sorted(idx_alpha)}, cov={sorted(idx_cov)}, "
            f"bench={sorted(idx_bench)}, prev={sorted(idx_prev)}, "
            f"sectors={sorted(idx_sec)}"
        )
    return sorted(alpha.index)


def _solve(prob: cp.Problem) -> str:
    """Try solvers in priority order, return status of first that works."""
    for solver in _SOLVERS:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return prob.status
        except cp.SolverError:
            continue
    return prob.status


def _build_and_solve(
    alpha_arr: np.ndarray,
    L: np.ndarray,
    w_bench_arr: np.ndarray,
    w_prev_arr: np.ndarray,
    sector_ids: np.ndarray,
    unique_sectors: list,
    cfg,
    turnover_mult: float = 1.0,
    sector_mult: float = 1.0,
) -> tuple[cp.Problem, cp.Variable]:
    n = len(alpha_arr)
    w = cp.Variable(n)
    a = w - w_bench_arr

    objective = (
        alpha_arr @ a
        - cfg.risk_aversion * cp.sum_squares(L.T @ a)
        - (cfg.cost_bps / 1e4) * cp.norm1(w - w_prev_arr)
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.abs(a) <= cfg.max_active_stock,
        cp.sum_squares(L.T @ a) <= cfg.max_te_annual**2,
        cp.norm1(w - w_prev_arr) <= 2 * cfg.max_turnover_oneway * turnover_mult,
    ]

    for sec in unique_sectors:
        mask = sector_ids == sec
        constraints.append(cp.abs(cp.sum(a[mask])) <= cfg.max_active_sector * sector_mult)

    prob = cp.Problem(cp.Maximize(objective), constraints)
    _solve(prob)
    return prob, w


def optimize_portfolio(
    alpha: pd.Series,
    cov_annual: pd.DataFrame,
    w_bench: pd.Series,
    w_prev: pd.Series,
    sectors: pd.Series,
    cfg,
) -> tuple[pd.Series, dict]:
    """Solve the enhanced-index portfolio optimization problem.

    Returns:
        (weights, diagnostics) where diagnostics contains status, step,
        realized_te, turnover, and objective.
    """
    tickers = _align_inputs(alpha, cov_annual, w_bench, w_prev, sectors)
    alpha_arr = alpha.loc[tickers].to_numpy(dtype=float)
    w_bench_arr = w_bench.loc[tickers].to_numpy(dtype=float)
    w_prev_arr = w_prev.loc[tickers].to_numpy(dtype=float)
    sectors_aligned = sectors.loc[tickers]
    sector_ids = sectors_aligned.to_numpy()
    unique_sectors = sorted(sectors_aligned.unique())

    L = psd_factor(cov_annual.loc[tickers, tickers])

    # Fallback ladder
    ladder = [
        (1.0, 1.0, "optimal"),
        (2.0, 1.0, "relax_turnover_x2"),
        (2.0, 1.5, "relax_sector_x1.5"),
    ]

    for to_mult, sec_mult, step in ladder:
        prob, w_var = _build_and_solve(
            alpha_arr,
            L,
            w_bench_arr,
            w_prev_arr,
            sector_ids,
            unique_sectors,
            cfg,
            turnover_mult=to_mult,
            sector_mult=sec_mult,
        )
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            logger.info("Optimizer converged at step '%s' (status=%s).", step, prob.status)
            w_arr = np.array(w_var.value).flatten()
            w_arr = np.where(w_arr > -_CLIP_TOL, np.maximum(w_arr, 0.0), w_arr)
            w_arr = w_arr / w_arr.sum()

            active = w_arr - w_bench_arr
            realized_te = float(np.sqrt(active @ (L @ L.T) @ active))
            turnover = float(0.5 * np.sum(np.abs(w_arr - w_prev_arr)))

            assert np.abs(w_arr.sum() - 1.0) < _CONSTRAINT_TOL
            assert (w_arr >= -_CONSTRAINT_TOL).all()

            weights = pd.Series(w_arr, index=tickers, name="weight")
            diag = {
                "status": prob.status,
                "step": step,
                "realized_te": realized_te,
                "turnover": turnover,
                "objective": float(prob.value),
            }
            return weights, diag

        logger.warning("Step '%s' infeasible (status=%s), trying next.", step, prob.status)

    logger.warning("All relaxation steps failed; returning benchmark weights.")
    weights = pd.Series(w_bench_arr, index=tickers, name="weight")
    diag = {
        "status": "fallback_benchmark",
        "step": "benchmark_fallback",
        "realized_te": 0.0,
        "turnover": float(0.5 * np.sum(np.abs(w_bench_arr - w_prev_arr))),
        "objective": 0.0,
    }
    return weights, diag
