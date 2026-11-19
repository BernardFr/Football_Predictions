# Dixon-Coles Poisson Model: Theory, Math, Evaluation & Backtesting

## 1. Overview
The **Dixon-Coles model** is an extension of the standard independent Poisson distribution used in association football (soccer) analytics to predict match scores and win/draw/loss probabilities more accurately.

Originally introduced by statisticians Mark Dixon and Stuart Coles in 1997, the model addresses a well-known flaw in basic Poisson modeling: the false assumption that home and away goals are completely independent.

### What Is Wrong with Standard Poisson?
* **Independence Assumption:** A standard Poisson model multiplies the independent probability of the home team scoring $x$ goals by the away team scoring $y$ goals ($P(\text{home}=x) \times P(\text{away}=y)$).
* **The Draw Deficit:** In real football matches, when both teams play conservatively or defensively, low-scoring games (especially 0–0 and 1–1) happen more frequently than a naive Poisson model predicts.

## How the Dixon-Coles Model Fixes It

The Dixon-Coles model retains the core Poisson framework for team strength parameters (attack a, defense b, and home-field advantage γ), but introduces two major adjustments: [[1](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/), [2](https://www.ajbuckeconbikesail.net/wkpapers/Airports/MVPoisson/soccer_betting.pdf)]

1. **The Low-Scoring Correction Factor (τ):**

   1. A dependence parameter, **ρ (rho)**, is added to adjust the probabilities of only four specific low-scoring scorelines:
      - **0 - 0**
      - **1 - 0**
      - **0 - 1**
      - **1 - 1**

   1. All other scorelines remain completely unchanged, and the marginal goal distributions still average out to the expected goals λ and μ.

   1. The modified probability formula is: [[1](https://statsultra.com/dixon-coles-model/), [2](https://impliedscore.com/dixon-coles-model/)]
      \(P^{*}(x,y)=P(x)\times P(y)\times \tau (x,y,\lambda ,\mu ,\rho )\)

1. **Time-Decay Weighting (ξ):**
   - Older matches are less representative of a team's current form than recent matches.
   - Dixon and Coles introduced an exponential time-decay weighting function (\(e^{-\xi t}\)) into the Maximum Likelihood Estimation (MLE) so that more recent results carry a higher weight when estimating team attack and defense parameters. [[1](https://www.ajbuckeconbikesail.net/wkpapers/Airports/MVPoisson/soccer_betting.pdf), [2](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)]

## Key Components Estimated in the Model

- **Attack Parameters (\(a_{i}\)):** How many goals a team is expected to score relative to the league average.
- **Defense Parameters (\(b_{i}\)):** How resilient a team is at preventing goals.
- **Home Advantage (γ):** An uplift factor for teams playing at their home stadium.
- **Dependence (ρ):** Typically negative (often around -0.08 to -0.13, depending on the league and dataset), which boosts the likelihood of 0–0 and 1–1 draws. [[1](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/), [2](https://docs.pena.lt/y/models/dixon_coles.html), [3](https://www.youtube.com/watch?v=vg5BxFCdYnE), [4](https://www.ajbuckeconbikesail.net/wkpapers/Airports/MVPoisson/soccer_betting.pdf)]

You can read a detailed breakdown of the math and implementation via [ImpliedScore's Dixon-Coles Guide](https://impliedscore.com/dixon-coles-model/) or explore open-source implementations like Python's `penaltyblog` library. [[1](https://docs.pena.lt/y/models/dixon_coles.html), [2](https://impliedscore.com/dixon-coles-model/)]

---

# Key Formulas

Here is the complete breakdown of the **τ(x, y) formulas**, how **Maximum Likelihood Estimation (MLE)** fits the model, and a working **Python implementation**. [[1](https://github.com/dashee87/blogScripts/blob/master/Jupyter/2018-09-13-predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting.ipynb), [2](https://predictionengine.app/learn/dixon-coles-soccer-model)]

------

1. ## The Exact Formulas for τ(x, y)

Let λ be the expected goals for the home team and μ be the expected goals for the away team. The adjustment function τ(x, y) modifies the joint probability of the four lowest scorelines using the dependence parameter ρ: [[1](https://predictionengine.app/learn/dixon-coles-soccer-model)]

| Home Goals (x)       | Away Goals (y)       | τ(x, y) Formula | Effect when ρ < 0            |
| -------------------- | -------------------- | --------------- | ---------------------------- |
| **0**                | **0**                | 1 - λ μ ρ       | Increases probability of 0–0 |
| **1**                | **0**                | 1 + μ ρ         | Decreases probability of 1–0 |
| **0**                | **1**                | 1 + λ ρ         | Decreases probability of 0–1 |
| **1**                | **1**                | 1 - ρ           | Increases probability of 1–1 |
| **All other scores** | **All other scores** | 1               | No change                    |

*Note on Parameter Bounds:* Because probabilities cannot be negative, τ(x, y) must always be greater than 0. During optimization, ρ is bounded by \(\max(-1/\lambda, -1/\mu) < \rho < \min(1/(\lambda\mu), 1)\) so that τ stays strictly positive. [[1](https://opisthokonta.net/?p=1685), [2](https://impliedscore.com/dixon-coles-model/)]

------

2. ## Fitting Parameters via Maximum Likelihood Estimation (MLE)

To find the optimal attack (\(\alpha _{i}\)), defense (\(\beta _{i}\)), home advantage (γ), and dependence (ρ) parameters across N historical matches, we maximize the **time-weighted log-likelihood** function: [[1](https://predictionengine.app/learn/dixon-coles-soccer-model)]

\(\ln L(\alpha ,\beta ,\gamma ,\rho )=\sum _{k=1}^{N}w_{k}\Big[\ln \tau (x_{k},y_{k})+\ln P(x_{k}\mid \lambda _{k})+\ln P(y_{k}\mid \mu _{k})\Big]\)

- **Expected Goal Rates:** For match k between home team i and away team j, the expected goals are parameterized exponentially to ensure they remain positive: [[1](https://github.com/dashee87/blogScripts/blob/master/Jupyter/2018-09-13-predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting.ipynb), [2](https://predictionengine.app/learn/dixon-coles-soccer-model)]
  \(\lambda _{k}=\exp (\alpha _{i}+\beta _{j}+\gamma )\)
  \(\mu _{k}=\exp (\alpha _{j}+\beta _{i})\)
- **Time-Decay Weight (\(w_{k}\)):** Each match is weighted by how many days (\(t - t_k\)) ago it was played, controlled by decay rate ξ: [[1](https://www.tamnguyen.io/writing/dixon-coles), [2](https://predictionengine.app/learn/dixon-coles-soccer-model)]
  \(w_{k}=\exp \big(-\xi \cdot (t-t_{k})\big)\)
- **Identifiability Constraint:** Because adding a constant to all attack ratings and subtracting it from all defense ratings produces the exact same λ and μ, we apply a sum-to-zero constraint (e.g., \(\sum \alpha_i = 0\) or \(\sum \beta_i = 0\)) so the optimizer converges to a unique solution. [[1](https://opisthokonta.net/?p=1685)]

---

## 4. Model Evaluation & Tuning $\xi$ with Ranked Probability Score (RPS)

Because win/draw/loss outcomes are ordered categories (Home Win > Draw > Away Win), **Ranked Probability Score (RPS)** is the standard metric for evaluating football match predictions:

$$
\text{RPS} = \frac{1}{R - 1} \sum_{i=1}^{R - 1} \left( \sum_{j=1}^{i} (p_j - e_j) \right)^2
$$


* For $R = 3$ outcomes (Home Win, Draw, Away Win), $p = (p_H, p_D, p_A)$ is the predicted probability vector and $e = (e_H, e_D, e_A)$ is the one-hot encoded actual result.
* **Lower RPS is better** ($0$ represents a perfect prediction, $1$ represents the worst possible prediction).
* **Rolling Backtesting:** Rather than optimizing $\xi$ directly inside the MLE objective (which would push $\xi \to 0$), we evaluate candidate values of $\xi$ over expanding/rolling time windows—training on matches prior to cutoff date $t$ and scoring out-of-sample predictions in $[t, t + \Delta t)$ via mean RPS.

------



## Practical Tips for Production Use

- **Tuning ξ (Time-Decay):** Rather than optimizing ξ inside the same MLE step (which would trivially push ξ → 0 to maximize historical likelihood), find ξ via cross-validation using **Ranked Probability Score (RPS)** or out-of-sample log-loss on rolling matchweeks. Typical values for daily decay in domestic leagues sit around ξ ≈ 0.0018 to 0.003. [[1](https://www.tamnguyen.io/writing/dixon-coles), [2](https://impliedscore.com/dixon-coles-model/), [3](https://predictionengine.app/learn/dixon-coles-soccer-model)]
- **Interpretation of Defense (β):** In this formulation, a **negative** \(\beta _{i}\) indicates a strong defense (it reduces the opponent's expected goal rate λ), while a **positive** \(\beta _{i}\) indicates a weak defense. [[1](https://predictionengine.app/learn/dixon-coles-soccer-model)]



------

## Breakdown of the Parameters

In the Dixon-Coles model, **λ (lambda)** and **μ (mu)** represent the **expected number of goals** scored by the home team and the away team, respectively, during a match.

- **λ (Lambda):** The expected goal rate (scoring intensity) for the **home team**. It is modeled using the home team's attacking strength, the away team's defensive strength, and a home-field advantage factor.
- **μ (Mu):** The expected goal rate (scoring intensity) for the **away team**. It is modeled using the away team's attacking strength and the home team's defensive strength.

### Mathematical Formulation

In a standard Dixon-Coles setup, these expected goal rates are calculated using exponential functions to ensure they always remain positive:

- **Home Expected Goals:** \(\lambda_{ij} = \exp(\alpha_i + \beta_j + \gamma)\)
- **Away Expected Goals:** \(\mu_{ij} = \exp(\alpha_j + \beta_i)\)

Where:

- **α (alpha):** Team attacking strength parameter (i for home, j for away).
- **β (beta):** Team defensive weakness parameter (a higher value means conceding more goals).
- **γ (gamma):** Home-field advantage parameter, applied only to the home team's goal rate.

If you'd like, let me know:

- Would you like to see how **α, β, and γ are constrained** (identifiability constraints) during model fitting?
- Do you want a **Python example** showing how to compute λ, μ, and the full score probability matrix?

------

To fit the Dixon-Coles model cleanly and generate score predictions, you need to apply **identifiability constraints** so the optimizer converges on a unique solution, and then combine the Poisson probabilities with the **τ (tau) adjustment**.

1. Identifiability Constraints on α, β, and γ

Without constraints, you could add a constant c to every attacking parameter (α) and subtract c from every defensive parameter (β), resulting in the exact same λ and μ values.

To prevent this infinite-solutions problem during optimization, models enforce one of two standard rules:

- **Sum-to-Zero Constraint (Most Common):** The average attacking strength across all N teams is fixed to zero.
  \(\sum _{i=1}^{N}\alpha _{i}=0\)
  *(Alternatively, in the original 1997 Dixon-Coles paper, the average attacking strength is constrained to sum to N, or \(\frac{1}{N}\sum \alpha_i = 1\), though summing to zero on the log-scale is standard in modern implementations).*
- **Reference Team Constraint:** One arbitrary team's attacking parameter is pinned to zero (α₁ = 0), and all other teams are measured relative to that baseline team.
- **No Constraint Needed on β or γ:** Once α is anchored, the defensive parameters (β) and the home advantage parameter (γ) naturally scale around it without ambiguity.

------

2. Python Example: Computing λ, μ, and the Score Matrix

The following script uses `numpy` and `scipy` to calculate expected goals, apply the τ correlation adjustment, and output a 6×6 exact scoreline probability matrix.

python

```
import numpy as np
from scipy.stats import poisson

def tau_adjustment(x, y, lam, mu, rho):
    """Dixon-Coles low-score dependence adjustment."""
    if x == 0 and y == 0:
        return 1 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1 + (lam * rho)
    elif x == 1 and y == 0:
        return 1 + (mu * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def predict_score_matrix(alpha_home, beta_home, alpha_away, beta_away, gamma, rho, max_goals=5):
    """Generates a (max_goals+1) x (max_goals+1) scoreline probability matrix."""
    # 1. Calculate expected goals (lambda and mu)
    lam = np.exp(alpha_home + beta_away + gamma)
    mu = np.exp(alpha_away + beta_home)
    
    # 2. Compute independent Poisson probabilities
    home_probs = poisson.pmf(np.arange(max_goals + 1), lam)
    away_probs = poisson.pmf(np.arange(max_goals + 1), mu)
    matrix = np.outer(home_probs, away_probs)
    
    # 3. Apply the tau adjustment to the 4 low-scoring cells
    for x in range(2):
        for y in range(2):
            matrix[x, y] *= tau_adjustment(x, y, lam, mu, rho)
            
    # Normalize matrix so probabilities sum strictly to 1.0
    matrix /= np.sum(matrix)
    return lam, mu, matrix

# Example parameters (after fitting with sum(alpha) = 0)
alpha_h, beta_h = 0.25, -0.15   # Strong home team
alpha_a, beta_a = -0.10, 0.10   # Average/slightly weak away team
gamma = 0.28                    # Home advantage (~32% boost in goals)
rho = -0.11                     # Typical negative rho in football

lam, mu, score_matrix = predict_score_matrix(alpha_h, beta_h, alpha_a, beta_a, gamma, rho)

print(f"Home Expected Goals (λ): {lam:.2f}")
print(f"Away Expected Goals (μ): {mu:.2f}\n")
print("Scoreline Probability Matrix (Rows = Home 0..5, Cols = Away 0..5):")
print(np.round(score_matrix, 4))

# Extract 1X2 Match Outcome Probabilities
home_win = np.sum(np.tril(score_matrix, -1))
draw = np.sum(np.trace(score_matrix))
away_win = np.sum(np.triu(score_matrix, 1))

print(f"\nHome Win: {home_win:.1%} | Draw: {draw:.1%} | Away Win: {away_win:.1%}")
```

Use code with caution.

*(Note on notation: In the original 1997 paper, τ(0,1) = 1 + λρ and τ(1,1) = 1 - ρ, so a negative ρ increases 0-0 and 1-1 probabilities while decreasing 1-0 and 0-1. The code above uses this exact standard convention.)*

If you'd like, let me know:

- Do you want to see the **maximum likelihood optimization code** (`scipy.optimize.minimize`) to estimate α, β, γ, and ρ from historical match results?
- Are you incorporating **time-decay weights** (ξ) so recent matches carry more weight than older ones?



------

To fit the Dixon-Coles model to historical data with **time decay**, you maximize a **weighted pseudo-log-likelihood** function. Recent matches receive higher weights so the model adapts to current team form.

How Time Decay Works

In the 1997 Dixon-Coles paper, each historical match k is weighted exponentially based on how many days ago (\(t - t_k\)) it was played relative to the prediction date t:

\(\phi (t-t_{k})=\exp \big(-\xi \cdot (t-t_{k})\big)\)

- **ξ (Xi) Decay Rate:** Controls how fast older matches lose influence. A typical daily decay rate in top-flight football is around **ξ ≈ 0.0018 to 0.0030** (which gives a half-life of roughly 230 to 385 days).
- **Log-Likelihood Weighting:** During optimization, the log-probability of each historical match is multiplied by \(\phi(t - t_k)\).

------

Python Implementation: MLE Fitting with Time Decay

This script builds a complete, vectorized maximum likelihood estimator using `scipy.optimize.minimize` with the **SLSQP** solver to enforce the \(\sum \alpha_i = 0\) constraint.

python

```
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

def dc_log_likelihood(params, home_idx, away_idx, home_goals, away_goals, weights, n_teams):
    """Computes the negative weighted log-likelihood for the Dixon-Coles model."""
    alpha = params[:n_teams]
    beta = params[n_teams:2*n_teams]
    gamma = params[-2]
    rho = params[-1]
    
    # Expected goal rates for all matches
    lam = np.exp(alpha[home_idx] + beta[away_idx] + gamma)
    mu = np.exp(alpha[away_idx] + beta[home_idx])
    
    # Vectorized tau adjustment
    tau = np.ones_like(lam)
    
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)
    
    tau[mask_00] = 1 - lam[mask_00] * mu[mask_00] * rho
    tau[mask_01] = 1 + lam[mask_01] * rho
    tau[mask_10] = 1 + mu[mask_10] * rho
    tau[mask_11] = 1 - rho
    
    # Guard against invalid log(<= 0) during optimizer line searches
    if np.any(tau <= 1e-10):
        return 1e10
    
    # Log-likelihood of independent Poisson + log(tau)
    log_lik = (
        poisson.logpmf(home_goals, lam)
        + poisson.logpmf(away_goals, mu)
        + np.log(tau)
    )
    
    # Return negative weighted sum (since scipy minimizes)
    return -np.sum(weights * log_lik)

def fit_dixon_coles(df, xi=0.002, ref_date=None):
    """Fits Dixon-Coles parameters on a DataFrame with Date, HomeTeam, AwayTeam, FTHG, FTAG."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if ref_date is None:
        ref_date = df["Date"].max()
    else:
        ref_date = pd.to_datetime(ref_date)
        
    # Calculate time-decay weights
    days_elapsed = (ref_date - df["Date"]).dt.days.values
    weights = np.exp(-xi * np.maximum(days_elapsed, 0))
    
    # Map teams to integer indices
    teams = sorted(list(set(df["HomeTeam"]).union(set(df["AwayTeam"]))))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    
    home_idx = df["HomeTeam"].map(team_to_idx).values
    away_idx = df["AwayTeam"].map(team_to_idx).values
    home_goals = df["FTHG"].values
    away_goals = df["FTAG"].values
    
    # Initial parameter guesses: alpha=0, beta=0, gamma=0.25, rho=-0.05
    init_params = np.zeros(2 * n_teams + 2)
    init_params[-2] = 0.25
    init_params[-1] = -0.05
    
    # Constraint: sum of attacking parameters (alpha) equals 0
    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n_teams])}]
    
    # Bounds keep parameters stable and ensure rho stays within valid probability bounds
    bounds = [(-3, 3)] * (2 * n_teams) + [(-1, 1), (-0.99, 0.99)]
    
    result = minimize(
        dc_log_likelihood,
        init_params,
        args=(home_idx, away_idx, home_goals, away_goals, weights, n_teams),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8}
    )
    
    params = result.x
    team_stats = pd.DataFrame({
        "Team": teams,
        "Attack_Alpha": params[:n_teams],
        "Defense_Beta": params[n_teams:2*n_teams]
    })
    
    return {
        "team_stats": team_stats,
        "gamma": params[-2],
        "rho": params[-1],
        "success": result.success
    }

# --- Example Usage ---
sample_data = pd.DataFrame({
    "Date": ["2026-08-15", "2026-08-22", "2026-09-01", "2026-09-12", "2026-10-05", "2026-11-01"],
    "HomeTeam": ["Arsenal", "Chelsea", "Liverpool", "Arsenal", "Chelsea", "Liverpool"],
    "AwayTeam": ["Chelsea", "Liverpool", "Arsenal", "Liverpool", "Arsenal", "Chelsea"],
    "FTHG":,
    "FTAG": [0, 1, 1, 1, 0, 1]
})

fitted = fit_dixon_coles(sample_data, xi=0.002)
print(f"Converged: {fitted['success']}")
print(f"Home Advantage (γ): {fitted['gamma']:.4f}")
print(f"Correlation (ρ): {fitted['rho']:.4f}\n")
print(fitted["team_stats"].to_string(index=False))
```

Use code with caution.

Key Implementation Notes

- **Penalty Guard (`tau <= 1e-10`):** During line-search steps, `scipy` may briefly test values of ρ that make 1 - λμρ ≤ 0. Returning a large penalty (`1e10`) keeps the optimizer inside the valid domain without throwing `NaN` errors.
- **Tuning ξ:** You do not estimate ξ inside the same log-likelihood function (otherwise the optimizer would set ξ → ∞ to ignore all past matches except the latest day). Instead, tune ξ via **out-of-sample cross-validation** using a scoring rule like **Ranked Probability Score (RPS)** or log-loss.

If you'd like, let me know:

- Do you want to see how to run a **grid search to tune the optimal ξ value** using Ranked Probability Score (RPS)?
- Would you like to export this complete pipeline (fitting, time decay, and prediction) into a **downloadable `.py` file**?



------



**`dixon_coles_model.py`** standalone Python script provides a complete, end-to-end Dixon-Coles pipeline with time-decay tuning and probabilistic match prediction:

- **Maximum Likelihood Estimation (`fit_dixon_coles`):** Fits team attacking (α), defensive (β), home advantage (γ), and low-score correlation (ρ) parameters using `scipy.optimize.minimize` (SLSQP) with the \(\sum \alpha_i = 0\) identifiability constraint and exponential time-decay weights (ξ).
- **Walk-Forward Cross-Validation (`tune_xi_grid_search`):** Evaluates candidate time-decay rates (ξ) chronologically across rolling time windows to prevent data leakage, selecting the parameter that minimizes the out-of-sample **Ranked Probability Score (RPS)**.
- **Match Prediction (`predict_score_matrix` & `predict_1x2`):** Computes expected goal rates (λ, μ), applies the τ adjustment to the 0-0, 0-1, 1-0, and 1-1 cells, and outputs normalized exact scoreline matrices alongside 1X2 outcome probabilities.
- **Runnable Example (`__main__`):** Includes a self-contained demonstration that simulates league match data, runs the ξ grid search, fits the optimal model, and prints sample match forecasts.

If you'd like, I can help you adapt this script to:

- Load and clean a **specific historical dataset** (such as Football-Data.co.uk CSVs)
- Add **Asian Handicap or Over/Under goal line** probability helpers

------

## SLSQP (Sequential Least Squares Programming)

is an iterative, gradient-based optimization algorithm designed to solve nonlinear programming problems subject to bounds, equality constraints, and inequality constraints. Originally developed by Dieter Kraft in 1988, it is the underlying Fortran routine wrapped by `scipy.optimize.minimize(..., method="SLSQP")`. [[1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html), [2](https://github.com/relf/slsqp/blob/main/README.md)]

### How SLSQP Works

Instead of solving a complex nonlinear problem all at once, SLSQP breaks it down into a sequence of simpler subproblems at each iteration: [[1](https://search.r-project.org/CRAN/refmans/nloptr/html/slsqp.html)]

- **Quadratic Approximation of the Objective:** At the current parameter estimate \(x_{k}\), SLSQP approximates the nonlinear objective function (such as the negative log-likelihood in Dixon-Coles) using a second-order Taylor expansion. It estimates the curvature (Hessian matrix) iteratively using **BFGS quasi-Newton updates** so you do not have to compute second derivatives manually. [[1](http://jacobwilliams.github.io/slsqp/proc/slsqp.html), [2](https://search.r-project.org/CRAN/refmans/nloptr/html/slsqp.html), [3](https://stackoverflow.com/questions/59808494/how-does-the-slsqp-optimization-algorithm-work)]
- **Linear Approximation of Constraints:** Any nonlinear equality or inequality constraints are linearized around \(x_{k}\) using their first-order derivatives (Jacobians). [[1](https://search.r-project.org/CRAN/refmans/nloptr/html/slsqp.html)]
- **Solving the Least-Squares Subproblem:** Combining a quadratic objective with linear constraints produces a standard Quadratic Programming (QP) subproblem. Kraft's implementation transforms this QP subproblem into an equivalent **linear least-squares subproblem** (giving the algorithm its name) to compute the next search direction \(d_{k}\). [[1](https://docs.scipy.org/doc/scipy-1.18.0/reference/optimize.minimize-slsqp.html), [2](https://search.r-project.org/CRAN/refmans/nloptr/html/slsqp.html), [3](http://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf)]
- **Line Search with Merit Function:** Once the search direction \(d_{k}\) is found, SLSQP performs a line search to choose a step size \(\alpha _{k}\) that decreases an **\(L_{1}\) merit function**. This ensures the algorithm balances reducing the objective function while staying within or moving toward valid constraint boundaries. [[1](http://jacobwilliams.github.io/slsqp/proc/slsqp.html)]
- **Convergence:** The algorithm repeats these steps until the change in parameters, the change in objective value (`ftol`), and the Karush-Kuhn-Tucker (KKT) optimality conditions fall below specified tolerances. [[1](https://docs.scipy.org/doc/scipy-1.18.0/reference/optimize.minimize-slsqp.html)]

### Why It Fits the Dixon-Coles Model Well

- **Smoothness:** The Poisson log-likelihood and exponential goal rates are smooth and twice continuously differentiable, matching SLSQP's mathematical assumptions.
- **Exact Constraint Handling:** It enforces the \(\sum \alpha_i = 0\) identifiability constraint via Lagrange multipliers while simultaneously respecting parameter bounds (such as keeping \(\rho \) within \((-1, 1)\)). [[1](https://arxiv.org/pdf/2408.13420), [2](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.SLSQP.html), [3](https://stackoverflow.com/questions/59808494/how-does-the-slsqp-optimization-algorithm-work)]

### Key References

- **Original Technical Report:** Dieter Kraft, *"A software package for sequential quadratic programming,"* Technical Report DFVLR-FB 88-28 (1988), outlining the foundational framework and available via DFVLR-FB 88-28 PDF.
- **Journal Publication:** Dieter Kraft, *"Algorithm 733: TOMP–Fortran modules for optimal control calculations,"* *ACM Transactions on Mathematical Software*, 20(3), pp. 262–281 (1994).
- **Software Documentation & Modern Overviews:** Implementation specifics and tuning options are available in the [SciPy SLSQP Reference Manual](https://docs.scipy.org/doc/scipy-1.18.0/reference/optimize.minimize-slsqp.html), alongside a contemporary analysis of its internal mechanics in the [PySLSQP Paper on arXiv](https://arxiv.org/html/2408.13420v1). [[1](http://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf), [2](https://github.com/relf/slsqp/blob/main/README.md), [3](https://docs.scipy.org/doc/scipy-1.18.0/reference/optimize.minimize-slsqp.html), [4](https://arxiv.org/html/2408.13420v1)]

Further assistance is available to demonstrate passing analytical gradients (`jac`) for faster model fitting or comparing SLSQP against alternative routines like `trust-constr` and `L-BFGS-B`.

## Poisson log probability mass function

The Poisson log probability mass function (`logpmf`) is the natural logarithm of the Poisson probability mass function, given by \(\ln f(k; \mu) = -\mu + k \ln \mu - \ln(k!)\) for integer counts k ≥ 0 and rate parameter μ ≥ 0. [[1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html), [2](https://en.wikipedia.org/wiki/Poisson_distribution)]

### Formula & Properties

- **PMF:** \(f(k; \mu) = \frac{e^{-\mu}\mu^k}{k!}\) for \(k = 0, 1, 2, \dots\)
- **Log-PMF:** \(\ln f(k; \mu) = -\mu + k \ln \mu - \ln \Gamma(k + 1)\)
- **Why use `logpmf`:** Summing log-probabilities avoids numerical underflow when computing log-likelihoods over many observations or extreme counts. [[1](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm), [2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html), [3](https://en.wikipedia.org/wiki/Poisson_distribution)]

### Python Usage (`scipy.stats`)

python

```
from scipy.stats import poisson

# Log probability of observing k = 3 events when mean/rate mu = 2.5
val = poisson.logpmf(k=3, mu=2.5)
```

Use code with caution.

See the [scipy.stats.poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html) manual or [JAX poisson.logpmf](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.poisson.logpmf.html) for array broadcasting, `loc` shifting, and differentiable implementations. [[1](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.poisson.logpmf.html), [2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)]

