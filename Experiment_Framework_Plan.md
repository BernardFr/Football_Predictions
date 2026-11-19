# Plan: a framework for evaluating models and tuning hyperparameters

Status: approved with the decisions in section 11. No code yet.

## 1. Goal

Make "is model B better than model A?" a one-command question with a trustworthy answer:

- every model variant is evaluated the same way, on the same matches, with the same cost functions
- the answer says whether a difference is real or noise
- experiments are written as small config files, not code changes
- results land in one Excel workbook, with suggested next experiments

The first experiment it has to support: **does adding each team's recent Form improve the current Dixon-Coles model?**

## 2. What we already have (and reuse)

| Piece | Where | Reuse as |
|---|---|---|
| Walk-forward backtest: refit before each matchday on earlier matches only, so nothing from the future leaks in | [backtest.py](src/football_predictions/backtest.py) `walk_forward` | the evaluation loop |
| Grid search over `half_life_days`, `ridge`, `feeder_weight`, `cup_weight` on the previous season | `backtest.tune`, `[tune]` in the TOML | the hyperparameter search, generalized |
| Cost functions: RPS, log loss, Brier, accuracy, GD error, 80% interval coverage, bookmaker and base-rate comparison | [metrics.py](src/football_predictions/metrics.py) | the cost function library |
| A second-step adjustment on top of the base model: features → multipliers on expected goals, coefficients fitted walk-forward with ridge | [lineups.py](src/football_predictions/lineups.py), `DixonColes.expected_goals(adjust=...)` | the pattern for Form |

The gap: each of these is wired to one specific use. There is no common way to describe a model variant, run it over several seasons, compare it against a baseline with confidence intervals, and keep a record of results.

## 3. Data split

### Three roles, not two

With walk-forward evaluation, "training set" means two different things. Keeping them apart avoids the most common mistake, which is tuning on the test season.

| Role | What it is | Example |
|---|---|---|
| **Fit window** | matches the model is fitted on before each matchday. Always "everything before this date", weighted by recency. | automatic |
| **Tuning seasons** | seasons whose walk-forward scores pick hyperparameters and fit adjustment coefficients (e.g. the Form weight) | 2024-25, 2025-26 |
| **Test season(s)** | seasons scored once per final decision, never used to choose anything | 2026-27 |

### Recommendation: rolling origin, plus a live holdout

2026-27 has about 100 games so far. That is too few to separate two good models, because differences between them are usually a few thousandths of RPS. Our own lineup test is an example: 0.2115 vs. 0.2116 over 279 games is not a difference.

So the default split is:

1. **Rolling-origin test** (main decision tool): for each test season *k* in 2023-24, 2024-25, 2025-26, tune on the two seasons before *k*, then test on *k*. That gives about 1,140 test games instead of 100. It needs results back to 2020-21 (approved): one config change (`history_seasons`) plus one download.
2. **2026-27 as a live holdout**: a variant that wins step 1 is scored on 2026-27 once, as a final check. The live ledger keeps adding to this holdout every week.

The split is set in the experiment file (section 6), so your example split (tune on 2024-25 + 2025-26, test on 2026-27) is a one-line choice.

## 4. Cost functions

| Cost | Measures | Role |
|---|---|---|
| **RPS** (ranked probability score) | quality of the H/D/A probabilities; rewards being "close" (a home win forecast punished less for a draw than for an away win) | **primary**: models are ranked on it, already used by `tune` |
| Log loss | punishes confident wrong forecasts hard | secondary, catches overconfidence |
| Accuracy (correct W/D/L pick) | share of matches where the most likely outcome happened | reported, never used to choose: it ignores probabilities, and the model almost never picks draws |
| GD MAE | mean absolute error of expected goal difference | secondary, for the goal difference forecast |
| GD 80% coverage | share of results inside the 80% goal difference interval; should be about 80% | calibration check |
| Goal log-likelihood | probability given to the exact score | secondary, rewards a good score grid |
| Bookmaker gap closed | on games with odds: (baseline RPS − variant RPS) / (baseline RPS − bookmaker RPS). 0% = no better than the current model, 100% = as good as the bookmakers, above 100% = better than them. | secondary, reported for every variant |

Bookmaker probabilities are the margin-free ones `metrics.implied_probs` already computes. The gap measure needs an average over many games, so it is reported per variant and per test season rather than per match, and it is undefined (shown blank) for a season where the baseline already beats the bookmakers.

**Adoption rule:** a variant is adopted when it beats the current model on RPS (section 8). The bookmaker gap is context, not a requirement.

Each cost is a small class with one method, `score(predictions) -> per-match array`. Returning one value per match, instead of an average, is what makes the significance test in section 7 possible.

## 5. Architecture

```
src/football_predictions/
  experiments/
    __init__.py
    forecaster.py   Forecaster base class + registry; DixonColesForecaster wraps today's model
    features.py     Form (and later other) features, computed only from games before kickoff
    adjusted.py     AdjustedForecaster: base model + fitted multipliers on expected goals
    costs.py        CostFunction base class; RPS, LogLoss, Accuracy, GdMae, ... (thin wrappers on metrics.py)
    evaluate.py     walk-forward loop for any Forecaster; per-match predictions out
    compare.py      paired comparison vs. baseline, bootstrap confidence intervals
    advisor.py      rule-based "what to try next"
    runner.py       reads an experiment TOML, runs the grid, writes results
experiments/          (project root)
  form.toml           one file per experiment
  results/            one Excel workbook per run + results_log.xlsx (all runs)
```

The key abstraction:

```python
class Forecaster(ABC):
    """Anything that can be fitted on past matches and forecast future ones."""
    name: str

    @abstractmethod
    def fit(self, matches: pd.DataFrame, asof: pd.Timestamp) -> None: ...

    @abstractmethod
    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """One row per fixture: xg_home, xg_away, p_home, p_draw, p_away, exp_gd, ..."""
```

- `DixonColesForecaster` wraps today's `DixonColes` and `Hyper`, with no change to the model.
- `AdjustedForecaster(base, feature_builder)` adds multipliers to expected goals, the way `lineups.py` does. Form is the first feature builder. Lineups can move here later.
- **Speed:** the base model's walk-forward predictions depend only on its hyperparameters, so they are cached per (hyperparameters, season). A grid over 24 Form variants then costs one base backtest plus 24 cheap second steps, not 24 full refits.

`main.py` gets one new command, `football-predictions experiment experiments/form.toml`. Existing commands are unchanged. Once a variant wins, it is promoted into the live `predict` path by a config setting.

## 6. Setting up an experiment

One TOML file per experiment. A grid is written as a list, and the runner tries every combination.

```toml
# experiments/form.toml
name = "form-v1"
question = "Does recent form improve forecasts beyond Dixon-Coles ratings?"
league = "EPL"

[split]
mode = "rolling"                     # "rolling" or "fixed"
test_seasons = ["2023-24", "2024-25", "2025-26"]
tuning_seasons_per_test = 2          # rolling: tune on the 2 seasons before each test season
# mode = "fixed"; tuning_seasons = ["2024-25", "2025-26"]; test_seasons = ["2026-27"]

[costs]
primary = "rps"
report = ["log_loss", "accuracy", "gd_mae", "gd_80_coverage", "bookmaker_gap_closed"]

[baseline]
forecaster = "dixon_coles"           # tuned hyperparameters from state/EPL/hyper.json

[[variants]]
forecaster = "dixon_coles+form"
form.points = ["3-1-0", "1-0-(-1)"]
form.window = [3, 5, 8]
form.weights = ["equal", "linear", "exponential"]
form.relative_to = ["raw", "expected"]      # see 7.2
# 2 x 3 x 3 x 2 = 36 variants
```

Running it writes `experiments/results/form-v1_<date>.xlsx` and appends one row per variant to `experiments/results/results_log.xlsx`. The log is the long-term record: what was tried, when, on which split, with what result, and the git commit it ran on.

## 7. First experiment: Form

### 7.1 Computing Form

For each team, before each match, using only matches already played:

```
form = sum(w_i * points_i) / sum(w_i)      over the team's last N matches, i = 1 (most recent) ... N
```

**Points scheme.** The two schemes you suggested differ only in what a draw is worth, because the Form coefficient is fitted and absorbs the scale:

| Scheme | W / D / L | Draw as a share of a win (after rescaling to 0-1) |
|---|---|---|
| 3-1-0 | 3 / 1 / 0 | 1/3 |
| +1 / 0 / -1 | 1 / 0 / -1 | 1/2 |

So the real question is "is a draw worth a third or half of a win, as a signal of form?" Both go in the grid. If one wins, a draw value of 0.4 is a cheap follow-up.

**Window and weights.** N is the number of past matches and `w_i` the weight of the i-th most recent one:

| Weights | w_1 ... w_N | Notes |
|---|---|---|
| equal | 1, 1, ..., 1 | your starting idea |
| last 3 only | equal with N = 3 | covered by the window setting, no separate option needed |
| linear | N, N-1, ..., 1 | your "increasing from 5th to most recent game" |
| exponential | 0.7^0, 0.7^1, ... | same idea, with one parameter (0.7) that can itself be tuned |

Windows of 3, 5 and 8 bracket your 5, so the advisor can tell whether the best window sits at the edge of the grid.

**Rules:**

- Which games count: **league games only** (decided). Cups are left out, since rotated cup line-ups make their results a noisy form signal.
- Season boundaries: **every team resets to neutral form at the start of each season** (decided). Only the current season's league games count, so a promoted team's form never comes from Championship games. As a result, Form has little to say in the first few matchdays. Those games stay in the test set, since live predictions have to cover them too; the per-match Predictions sheet lets us check whether they dilute the effect.
- Fewer than N games played: **average over the games available**, and treat a team with no games as neutral.
- Home and away form: **one combined form** first. A home/away split is a follow-up only if Form helps at all.

### 7.2 The important twist: raw form vs. form relative to expectation

The current model already weights recent games more (180-day half-life), so team ratings already contain much of what raw form measures. Man City winning 4 of their last 5 is not news to the model. Man City winning 4 of 5 against stronger opponents than usual, or Coventry winning 4 of 5 when the model expected 1.5 wins, might be.

So each points scheme is tested two ways:

- **raw:** form computed from actual points, as above
- **relative to expected:** actual points minus the points the model expected before each of those games (from its pre-match probabilities), i.e. "how much is this team beating its own ratings lately?"

My expectation, to be confirmed or refuted by the data: raw form adds little or nothing; form relative to expected is the version with a real chance of helping. The new Stats tab already shows over- and under-performance by season. This is the same idea, applied to the last few games.

### 7.3 How Form enters the model

Same two-step approach as lineups, so the base model is untouched:

```
xg_home_adj = xg_home * exp(+beta * (form_home - form_away))
xg_away_adj = xg_away * exp(-beta * (form_home - form_away))
```

- Form is standardized (mean 0, standard deviation 1 over the tuning seasons) so that `beta` means the same thing across schemes.
- `beta` is fitted on the tuning seasons by maximizing the Poisson likelihood of actual goals, with a ridge penalty pulling it toward 0. `beta = 0` gives back the current model exactly, so Form has to earn a non-zero weight.
- The fitted `beta` and its confidence interval are reported. A `beta` near 0 is itself the answer "form adds nothing".

## 8. Comparing models: is the difference real?

Every variant is scored on exactly the same test matches as the baseline, so the comparison is paired:

- **Per-match difference** in RPS (variant minus baseline), averaged over the test seasons.
- **95% confidence interval** by bootstrap, resampling whole matchdays, since games on the same weekend share conditions.
- **Verdict:**

| CI of the RPS difference | Verdict |
|---|---|
| entirely below 0 | better |
| entirely above 0 | worse |
| contains 0 | no detectable difference |

- **Selection effect:** with 36 variants, the best one will look a little better than it really is, by chance alone. Two guards: the winner is scored once on the 2026-27 holdout, and the verdict is shown next to the spread of all variants' results.

## 9. Viewing results

One Excel workbook per run, formatted with `excel_utilities.py`:

| Sheet | Contents |
|---|---|
| Summary | the question, split, baseline score, best variant, verdict, and the advisor's recommendations |
| Leaderboard | every variant: RPS and the secondary costs (including bookmaker gap closed), difference vs. baseline with 95% CI, verdict, fitted coefficient; sorted by RPS, with conditional colors |
| Parameter effects | for each grid setting (window, weights, ...), the average RPS of all variants with that value. Shows which settings matter and which don't. |
| By season | each variant's difference vs. baseline in each test season, as a check that one season isn't carrying the result |
| Calibration | reliability table for the baseline and the best variant |
| Predictions | per-match predictions for the baseline and the best variant, for drilling down |

Plus `results_log.xlsx`: one row per variant per run, across all experiments.

### Recommendations for further experiments

`advisor.py` turns the results into suggestions with simple, transparent rules. Examples:

| Observation | Recommendation |
|---|---|
| best value of a setting is at the edge of its grid (e.g. window = 8, the largest tried) | "extend the grid: try window 10, 13" |
| a setting changes RPS by less than the CI width across all its values | "drop this setting; it doesn't matter" |
| best variant's CI contains 0 | "no detectable gain; more test seasons would narrow the CI to ±x" |
| better on one test season, worse on the others | "unstable; don't adopt" |
| better, and the 2026-27 holdout agrees | "adopt: set `form = ...` in football_predictions.toml" |
| fitted `beta` is near 0 for all raw-form variants but not for the relative ones | "raw form is already in the ratings; continue with relative form" |

These are rules, not an LLM. Each recommendation shows the numbers that triggered it.

## 10. Implementation phases

Each phase ends with a check, and tests go in `tests/` as usual.

1. **Evaluation core.** `Forecaster`, `DixonColesForecaster`, `CostFunction` classes, `evaluate.py`.
   Check: running the baseline through the framework reproduces today's `backtest` numbers exactly.
2. **Comparison and reporting.** `compare.py` (paired bootstrap), the Excel workbook, `results_log.xlsx`.
   Check: comparing the baseline with itself gives a difference of exactly 0 and the verdict "no detectable difference"; a deliberately degraded model (e.g. all predictions set to base rates) comes out "worse".
3. **Experiment runner.** TOML format, grid expansion, rolling-origin split, caching of base predictions, the `experiment` command.
   Check: the current `[tune]` grid, re-run as an experiment file, picks the same hyperparameters as `tune`.
4. **Form.** `features.py` (form computation, raw and relative to expected), `AdjustedForecaster`, `experiments/form.toml`.
   Check: unit tests on hand-computed form for a small fixture list, including a check that a match never sees its own result or any later one; and `beta = 0` reproduces the baseline exactly.
5. **Advisor.** The rules in section 9.
   Check: unit tests with synthetic leaderboards that trigger each rule.
6. **Run form-v1 and decide.**

Phases 1-3 are the framework; 4 is the first experiment. Lineups can move into the framework later as a second `AdjustedForecaster`. That's optional and not part of this plan.

## 11. Decisions

1. **More history:** download 2020-21 to 2022-23 for the rolling-origin split. Approved.
2. **Form scope:** league games only, current season only: every team starts each season with neutral form (section 7.1).
3. **Base model settings:** kept fixed at the current tuned hyperparameters (`state/<league>/hyper.json`) during the Form experiment. Only the Form settings and `beta` are fitted.
4. **Adoption:** beating the current model on RPS is enough. The bookmaker comparison is a reported cost, "bookmaker gap closed" (section 4), not a requirement.
