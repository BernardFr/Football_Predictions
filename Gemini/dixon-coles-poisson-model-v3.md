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
