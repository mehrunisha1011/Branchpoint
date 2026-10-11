"""
simulation.py - 10-year Monte Carlo projection of money, income, expenses and runway.

Self-contained: needs only numpy. No API keys. Runs locally and inside a sandbox.

Usage (local test):
    python simulation.py

Usage (from other code):
    from simulation import run_simulation, SCENARIOS
    result = run_simulation(SCENARIOS["quit_school_startup"])

Output shape (JSON-serialisable):
{
  "name": str,
  "seed": int, "n_sims": int, "years": int,
  "yearly": [
     {"year": 1,
      "cash":     {"low": x, "mid": x, "high": x},
      "income":   {"low": x, "mid": x, "high": x},   # monthly income at year end
      "expenses": {"low": x, "mid": x, "high": x},   # monthly expenses at year end
      "runway_months": {"low": x, "mid": x, "high": x},
      "prob_ruined_by_now": 0.12},
     ...
  ],
  "summary": {...}
}
low = 10th percentile (bad luck), mid = median, high = 90th percentile (good luck).
"Ruined" means cash dropped below zero at some point. After that the simulation lets cash
continue negative (debt) so you can see how deep the hole goes.
"""

import json
import numpy as np

MONTHS = 12
RUNWAY_CAP = 120.0  # runway is capped at 10 years (cash that never runs out)

# ---------------------------------------------------------------------------
# Scenario format. All money is in one currency, per month unless stated.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "scenario",
    "years": 10,
    "n_sims": 10000,
    "seed": 42,
    "starting_cash": 5000.0,
    "monthly_income": 2000.0,
    "income_growth": 0.03,        # expected yearly growth of income (0.03 = 3%)
    "income_volatility": 0.10,    # yearly randomness of income (std dev of log growth)
    "monthly_expenses": 1800.0,
    "expense_growth": 0.03,       # yearly inflation / lifestyle growth
    "expense_volatility": 0.03,
    # Random bad events (medical bill, car repair, layoff cost...)
    "shock_prob_per_year": 0.30,
    "shock_size_months": 2.0,     # average shock = this many months of expenses
    # Income collapse (lose job / customers): income drops for a while
    "income_loss_prob_per_year": 0.08,
    "income_loss_fraction": 0.60,  # income falls by this fraction (0.6 = -60%)
    "income_loss_months": 6,
    # Big upside (promotion, startup traction): income jumps and stays
    "breakout_prob_per_year": 0.05,
    "breakout_multiplier": 2.0,
}

# Three hand-made scenarios for testing the engine.
SCENARIOS = {
    "stay_and_finish": {
        "name": "Stay in school and finish, then take a salaried job",
        "starting_cash": 4000, "monthly_income": 800, "income_growth": 0.20,
        "income_volatility": 0.08, "monthly_expenses": 750, "expense_growth": 0.03,
        "shock_prob_per_year": 0.25, "shock_size_months": 1.5,
        "income_loss_prob_per_year": 0.05, "income_loss_fraction": 0.5,
        "income_loss_months": 4, "breakout_prob_per_year": 0.06,
        "breakout_multiplier": 1.5,
    },
    "quit_school_startup": {
        "name": "Quit school and build a startup full-time",
        "starting_cash": 20000, "monthly_income": 500, "income_growth": 0.20,
        "income_volatility": 0.70, "monthly_expenses": 1200, "expense_growth": 0.04,
        "shock_prob_per_year": 0.35, "shock_size_months": 2.0,
        "income_loss_prob_per_year": 0.20, "income_loss_fraction": 0.7,
        "income_loss_months": 8, "breakout_prob_per_year": 0.08,
        "breakout_multiplier": 4.0,
    },
    "steady_job_baseline": {
        "name": "Keep a steady job (low risk baseline)",
        "starting_cash": 10000, "monthly_income": 3500, "income_growth": 0.04,
        "income_volatility": 0.04, "monthly_expenses": 2500, "expense_growth": 0.03,
        "shock_prob_per_year": 0.25, "shock_size_months": 1.0,
        "income_loss_prob_per_year": 0.04, "income_loss_fraction": 0.5,
        "income_loss_months": 4, "breakout_prob_per_year": 0.03,
        "breakout_multiplier": 1.4,
    },
}


def _pct(arr):
    """Low / mid / high = 10th / 50th / 90th percentile across simulations."""
    lo, mid, hi = np.percentile(arr, [10, 50, 90])
    return {"low": round(float(lo), 2), "mid": round(float(mid), 2), "high": round(float(hi), 2)}


def _clean(scenario):
    """Merge with defaults and validate so bad input gives a clear error."""
    s = dict(DEFAULTS)
    s.update({k: v for k, v in scenario.items() if v is not None})
    for key in ("years", "n_sims", "income_loss_months"):
        s[key] = int(s[key])
    if not 1 <= s["years"] <= 30:
        raise ValueError("years must be between 1 and 30")
    if not 100 <= s["n_sims"] <= 200000:
        raise ValueError("n_sims must be between 100 and 200000")
    for key in ("shock_prob_per_year", "income_loss_prob_per_year", "breakout_prob_per_year"):
        s[key] = min(max(float(s[key]), 0.0), 1.0)
    s["income_loss_fraction"] = min(max(float(s["income_loss_fraction"]), 0.0), 1.0)
    for key in ("starting_cash", "monthly_income", "monthly_expenses", "income_growth",
                "expense_growth", "income_volatility", "expense_volatility",
                "shock_size_months", "breakout_multiplier"):
        s[key] = float(s[key])
    if s["monthly_expenses"] < 0 or s["monthly_income"] < 0:
        raise ValueError("income and expenses cannot be negative")
    return s


def run_simulation(scenario):
    """Run the Monte Carlo and return yearly low/mid/high paths plus a summary."""
    s = _clean(scenario)
    rng = np.random.default_rng(s["seed"])
    n, years = s["n_sims"], s["years"]
    months = years * MONTHS

    cash = np.full(n, s["starting_cash"])
    income = np.full(n, s["monthly_income"])
    expenses = np.full(n, s["monthly_expenses"])
    ruined = np.zeros(n, dtype=bool)
    ruin_month = np.full(n, np.nan)
    loss_left = np.zeros(n, dtype=int)  # months of income collapse remaining

    # Convert yearly numbers to monthly ones
    inc_drift = np.log1p(s["income_growth"]) / MONTHS
    exp_drift = np.log1p(s["expense_growth"]) / MONTHS
    inc_vol = s["income_volatility"] / np.sqrt(MONTHS)
    exp_vol = s["expense_volatility"] / np.sqrt(MONTHS)
    p_shock = 1 - (1 - s["shock_prob_per_year"]) ** (1 / MONTHS)
    p_loss = 1 - (1 - s["income_loss_prob_per_year"]) ** (1 / MONTHS)
    p_break = 1 - (1 - s["breakout_prob_per_year"]) ** (1 / MONTHS)

    yearly = []
    for m in range(1, months + 1):
        # Income random walk with drift (a zero income stays zero unless a breakout hits)
        income *= np.exp(inc_drift - 0.5 * inc_vol ** 2 + inc_vol * rng.standard_normal(n))
        expenses *= np.exp(exp_drift - 0.5 * exp_vol ** 2 + exp_vol * rng.standard_normal(n))

        # Breakout: income jumps (from zero it jumps to a multiple of starting expenses)
        breakout = rng.random(n) < p_break
        jump_base = np.where(income > 0, income, s["monthly_expenses"] * 0.5)
        income = np.where(breakout, jump_base * s["breakout_multiplier"], income)

        # Income collapse: starts randomly, lasts income_loss_months
        start_loss = (rng.random(n) < p_loss) & (loss_left == 0)
        loss_left = np.where(start_loss, s["income_loss_months"], loss_left)
        effective_income = np.where(loss_left > 0, income * (1 - s["income_loss_fraction"]), income)
        loss_left = np.maximum(loss_left - 1, 0)

        # Expense shocks: lognormal, average = shock_size_months of expenses
        shock_hit = rng.random(n) < p_shock
        shock_cost = shock_hit * expenses * s["shock_size_months"] * rng.lognormal(-0.125, 0.5, n)

        cash = cash + effective_income - expenses - shock_cost

        newly_ruined = (cash < 0) & ~ruined
        ruin_month[newly_ruined] = m
        ruined |= cash < 0

        if m % MONTHS == 0:
            net_burn = expenses - effective_income
            runway = np.where(net_burn > 0, np.maximum(cash, 0) / np.where(net_burn > 0, net_burn, 1), RUNWAY_CAP)
            runway = np.minimum(runway, RUNWAY_CAP)
            yearly.append({
                "year": m // MONTHS,
                "cash": _pct(cash),
                "income": _pct(effective_income),
                "expenses": _pct(expenses),
                "runway_months": _pct(runway),
                "prob_ruined_by_now": round(float(ruined.mean()), 4),
            })

    ruined_months = ruin_month[~np.isnan(ruin_month)]
    summary = {
        "prob_ruin_ever": round(float(ruined.mean()), 4),
        "median_month_of_ruin": float(np.median(ruined_months)) if ruined_months.size else None,
        "final_cash": _pct(cash),
        "prob_final_cash_above_start": round(float((cash > s["starting_cash"]).mean()), 4),
        "disclaimer": "Simulation output, not financial advice.",
    }
    return {"name": s["name"], "seed": s["seed"], "n_sims": n, "years": years,
            "yearly": yearly, "summary": summary}


if __name__ == "__main__":
    for key, scenario in SCENARIOS.items():
        r = run_simulation(scenario)
        print(f"\n=== {r['name']} ===")
        print("year  cash(low / mid / high)            ruined%")
        for y in r["yearly"]:
            c = y["cash"]
            print(f"{y['year']:>4}  {c['low']:>10,.0f} {c['mid']:>10,.0f} {c['high']:>10,.0f}   {y['prob_ruined_by_now']*100:>6.1f}")
        print("summary:", json.dumps(r["summary"]))
