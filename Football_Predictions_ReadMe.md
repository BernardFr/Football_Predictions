# Football_Predictions

Win/draw/loss probabilities and goal-difference forecasts for league football, using a
time-weighted Dixon-Coles model. It's generic across leagues (EPL, Bundesliga, La Liga, Serie A
and Ligue 1 are configured; enable them in `src/football_predictions.toml`).

## Setup (once)

```bash
cd ~/Documents/Code/Football_Predictions
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"                    # installs the football-predictions command into the venv
football-predictions download     # EPL + Championship, 2023-24 to now, from football-data.co.uk
football-predictions tune         # picks decay / shrinkage / cup weight on last season (a few minutes)
football-predictions update       # backtest this season, predict upcoming games, write the report
```

The report is `reports/EPL_report.xlsx`. Add `--all` to run every enabled league, or `--league BUN` for one.

## The feedback loop

**At the end of each matchweek**, run these commands:

```bash
football-predictions matchweek                       # about 20 seconds once the ESPN data is cached
git add site reports state && git commit -m "Matchweek update" && git push   # publishes the website
```

`matchweek` does these steps, in this sequence:

1. Downloads the latest results.
2. Scores every prediction logged before kickoff (`state/EPL/ledger.csv`): the live track record and the
   website's Last matchweek.
3. Refits on all matches played so far, with recent games weighted most, and reruns the walk-forward backtest
   of the season (the grades and the predictability ranking).
4. Predicts the games in the next 10 days (from football-data's `fixtures.csv` and ESPN's schedule) and logs them.
5. Refreshes the ESPN line-ups, goals and keeper and defender records (the Impact Players tables). If ESPN
   fails, the command logs a warning, keeps the previous player data and continues.
6. Rebuilds the Excel report and the website.

`matchweek` is the same as `update`, then `lineups-backtest`, then `site`, with one report build instead of
two. `update` alone does steps 1 to 4 and builds the report. It is enough between matchweeks (for example
daily), when the player data does not change.

About an hour before kickoff, run `football-predictions lineups` to adjust for the announced starting XIs
(see **Enhanced predictions from starting XIs**). Re-tune once a season (`tune`), for example in July after the
previous season is complete.

## What each prediction contains

| Column | Meaning |
|---|---|
| pick | Most likely outcome (home team, Draw, or away team) |
| confidence, band | Probability of the pick. High >= 60%, Medium >= 45%, Low < 45% (set in config) |
| p_home / p_draw / p_away | Full W/D/L probabilities |
| exp_gd | Expected goal difference, home minus away |
| gd_range_80 | 80% interval for the goal difference |
| likely_score | Single most likely scoreline |
| bk_home / bk_draw / bk_away | Bookmaker probabilities (margin removed), for comparison |

## The report

`reports/EPL_report.xlsx` has these sheets. A sheet is left out when its data does not exist yet.

| Sheet | Content |
|---|---|
| Upcoming | The predictions for the games in the next 10 days, most confident first |
| Backtest | How much to trust the predictions: the model vs. closing bookmaker odds and a base rate (RPS, lower is better; log loss; accuracy), the hit rate of each confidence band, how often results fell inside the 80% interval, and the season grade |
| Grades | The grade of the season, of each matchweek and of each game of the backtest (see **Grading predictions**) |
| Predictability | The teams from most to least predictable |
| Calibration | Does a 30% forecast happen 30% of the time? |
| Backtest detail | Every backtest prediction with its result |
| Stats | Teams that beat or fell short of their predictions; best and worst attacks and defences |
| Lineup predictions, Lineup backtest | The lineup-adjusted predictions and their test (see **Enhanced predictions from starting XIs**) |
| Impact Players | The players who matter most to each team |
| Live track record | The predictions logged before kickoff, scored on the results |
| Team ratings, Tuning | The fitted ratings and the tuning grid |

The backtest predicts every game of the season so far with only the results known before that game
(walk-forward), so its scores are an honest test of the model.

## Grading predictions

The grade (`grading.py`) tells how good one prediction was, or a group of predictions. It compares the
predicted score probabilities with the league's usual scorelines (the previous seasons):

- 0 means no better than the usual scorelines. +100 means certain and right. Below 0 means worse.
- It weighs the result (50%), the goal difference (25%) and each side's goals (25%). Change the weights
  in `PART_WEIGHTS` in `grading.py`.
- Each part uses the ranked probability score (RPS), so probability placed near the actual score also earns
  credit. A pick of +1 goal difference scores better than a pick of -1 when the result is +2.

A group of games (a matchweek, a team's games, the season) adds up the RPS of its games before the comparison,
so one odd game does not dominate it. One game's grade is mostly luck. Judge the model on the matchweek grade
and the season grade. For comparison, the model graded +6 after 11 matchweeks of 2026-27, and bookmakers
usually do about 10% better than base rates.

**Matchweeks.** The data has no matchweek numbers, so the dates set the matchweeks. A new matchweek starts after 2 or more
days with no games, or on a date when a team would play a second time in the matchweek.

**Predictability.** A team's grade is the grade of the predictions for the games it played. A predictable
team plays to its rating. The grade belongs to the game, not only to the team: a low grade can also come
from the team's opponents.

The grades show in the report (Backtest, Grades and Predictability sheets) and on the website (the Last
matchweek table and the Season predictions tab).

## How the model works

- Each team has an attack and a defence rating, and each division has a scoring level and a home
  advantage. Goals are Poisson, with the Dixon-Coles correction for 0-0, 1-0, 0-1 and 1-1.
- Matches are weighted by age (half-life tuned per league; about 150 days for the EPL) and ratings
  are shrunk slightly toward average.
- **Promoted teams:** the second division is fitted on the same rating scale. Teams that move up
  and down link the two divisions, so a promoted side starts with its Championship rating
  translated to Premier League level. No hand-set "promotion penalty" is needed.
- **One model per league:** the code is shared, and each league gets its own fit and its own tuned
  settings (`state/<league>/hyper.json`), because ratings only mean something within one league.

## Cup games (FA Cup, EFL Cup, Community Shield)

Cup results come from ESPN (`cups` under the league in `src/football_predictions.toml`) and are
added to the fitting data. They help most where the league can't: a Premier League side against a
Championship side directly measures the gap between the divisions, which sharpens promoted-team
ratings. Because cup ties are noisier, they are handled differently from league games:

- **Scored at 90 minutes.** Ties that went to extra time or penalties count as draws at the
  90-minute score.
- **Neutral venues** (Wembley semi-finals and finals, the Community Shield) get no home advantage.
- **Only games between rated teams** (Premier League and Championship) are used; a tie against a
  League Two or non-league side is skipped.
- **Weighted:** `cup_weight` (0 = ignore cups) is chosen by `tune` from the backtest, so cups only
  count as much as they actually help. Cup games are never predicted, and lineup features still use
  league games only, since cup line-ups are often rotated.

`football-predictions download` lists how many cup games were found per season and competition.
For other leagues, add their ESPN cup slugs to `cups` (e.g. `ger.dfb_pokal` for the DFB-Pokal).

## Manual adjustments (and lineups)

Create `state/EPL/adjustments.csv` to nudge a team's expected goals, for example when you know a
key player is out:

```csv
team,attack,defence,until,note
Man City,0.90,1.00,2026-11-30,striker injured
Arsenal,1.00,1.08,,first-choice GK out
```

`attack` multiplies the goals the team is expected to score. `defence` multiplies the goals it is
expected to concede (above 1 is worse). Rows expire after `until`, and adjusted predictions are
flagged in the report. This is also the hook an automated starting-XI adjustment would use.

## Enhanced predictions from starting XIs (ESPN)

ESPN's public feed (no key) lists each match's starters, substitutions, goals and assists. It's
undocumented, so it may change without notice. Two commands use it:

```bash
football-predictions lineups-backtest   # once, then occasionally: does the lineup adjustment help?
football-predictions lineups            # ~45-60 min before kickoff, after the XIs are announced
```

The first `lineups-backtest` downloads two seasons of ESPN data, about 1,000 requests or roughly
10 minutes at the polite default pace. After that everything is cached, and later runs fetch only
new matches.

**How it works.** For each team, the announced XI is compared with how the team has lined up in its
last 19 matches (recent ones weighted more). Only players who have played for the team this season
count, so a player who left the club is not counted as missing:

| Feature | Meaning |
|---|---|
| `miss` | Share of the team's usual playing time missing from the XI (0 = full strength) |
| `att` | Attacking output of the XI vs. its usual XIs (goals + assists per 90, shrunk toward position averages) |
| `gk` | Keeper quality of the XI vs. the usual keepers: log of the goals-allowed rate (1 − save %, shrunk toward the league average of 67%). 0 = the usual keeper; above 0 = a weaker keeper |

These scale the base model's expected goals for both sides. The size of each effect is estimated
from past matches, shrunk toward "no effect", then refitted before every match date. Each effect is
held to its sensible sign: a missing regular can't help the team, and a weaker keeper can't concede
fewer goals. The
**Lineup backtest** sheet compares base vs. adjusted on the same matches (RPS, log loss, accuracy,
and goal log-likelihood, which reacts sooner than W/D/L scores). Lineup predictions are logged to
the ledger as variant `lineup`, so **Live track record** scores them separately from the base model.

The **Impact Players** sheet has one column per team. It shows these tables:

- **Usual XI**: the 11 players with the most recent minutes, by position.
- **Most impact**: the 3 players whose absence costs the most expected points per match. The
  team's average expected goals are scaled as if the player misses the usual XI and the best
  backup at the same position starts. Expected points = 3 x P(win) + P(draw). The change in
  expected goal difference is also shown.
- **No impact**: players under 0.005 points per match. Players outside the usual XI always have
  no impact, because the model only uses the starting XI.
- **Top 5 scorers**: this season's goals per player. Own goals are not counted.
- **Valuable scorers**: this season's top 3 players for first goals (the team's first goal of the
  match), equalizers (the goal levels the score) and go-ahead goals (the goal puts the team ahead
  from a level score). **Combined** ranks players by the sum of the three counts (equal weight).
  Own goals are not credited to a player.
- **Goalkeepers** and **Defenders**: this season's record of the keeper and defenders in each
  usual XI. "Conceded" counts the goals the team conceded while the player was on the pitch. A
  clean sheet needs 60 or more minutes with no goals conceded. Ranks compare all usual XI keepers
  (or defenders). Rank 1 is the fewest goals conceded per 90 minutes, or the most clean sheets.
  These numbers mostly measure the team's defence, not only the player.
- **Defenders on/off**: the team's goals conceded per 90 minutes with the defender on the pitch
  vs. off the pitch. Off includes the matches the defender missed. A positive difference means
  the team concedes less with the defender. The comparison needs 270 or more minutes off the
  pitch, so regular starters show "too few minutes off" until they miss about 3 matches.

`lineups` covers games in the next 36 hours whose XIs are both published; it lists the others as
"not published yet". Re-run it closer to kickoff.

**Team names.** ESPN ("Manchester United") is matched to football-data ("Man United") automatically
by pairing fixtures on the same date (`state/EPL/team_map.csv`). If a team is reported as unmatched,
add a row to `state/EPL/team_map_overrides.csv` with columns `espn_name,fd_name`. If ESPN starts
refusing requests (403), change `user_agent` under `[lineups]` in `src/football_predictions.toml`.

## Scoreline histogram

```bash
football-score-histogram            # Premier League, past two seasons + this season
```

Scores run left to right from the heaviest home defeat to the biggest home win (by goal
difference). Within one goal difference, defeats put the highest-scoring result first
(…, 1-2, 0-1) and wins put it last (1-0, 2-1, …); draws run 0-0, 1-1, 2-2, ….
Three files go to `reports/`:

- `EPL_score_histogram.html`: interactive chart. Tick any combination of the three seasons and
  "All seasons combined", and switch between % of matches and match counts. Open it in a browser.
- `EPL_score_histogram.xlsx`: the table, a per-season summary, a home-by-away goals grid, the
  match list, and a native Excel chart with all four series (use Excel's chart filter to hide some).
- `EPL_score_histogram.png`: all seasons combined as bars, each season as dots.

Seasons are compared as a share of matches by default, since the current season is only part-way
through. Options: `--seasons 5`, `--top 25` (keep only the 25 most common scores and pool the rest
as "Other"), `--league BUN`, `--no-refresh`.

## Website

```bash
football-predictions site           # after `update` (and `lineups-backtest` for the player tables)
git add site && git commit -m "Update site" && git push
```

`site` writes the website to `site/`. It reads the files in `state/EPL/` and the cached results, and
downloads nothing. A push that changes `site/` starts the GitHub Pages workflow
(`.github/workflows/publish.yml`), which publishes the site at
https://bernardfr.github.io/Football_Predictions/. The site has four tabs:

| Tab | Content |
|---|---|
| Predictions | The upcoming games (from `upcoming.csv`), then the last matchweek: the predictions logged before kickoff vs. the results, with the grade of each game and of the matchweek. Picks are shaded green (right) or red (wrong). |
| Season predictions | How good this season's predictions were: the season grade, the teams from most to least predictable, and the grade of each matchweek and each game (from the backtest). The same data as the Grades and Predictability sheets. |
| Impact players | The tables of the report's Impact Players sheet, one team at a time. A link such as `players.html#chelsea` opens one team. |
| Scoring statistics | The scoreline histogram, with only the current season ticked. |

The last matchweek is the latest matchweek of the base model's scored predictions (see **Matchweeks** under
**Grading predictions**). The pages come from the templates in `src/football_predictions/templates/`.
Edit their text there.

## Commands

```
football-predictions download|tune|backtest|predict|update|report|lineups-backtest|lineups|site|matchweek
    [--league EPL | --all] [--season 2025-26] [--source openfootball] [--today YYYY-MM-DD]
```

`--source openfootball` uses a GitHub-hosted results mirror (no odds, can lag by weeks).
`--today` replays the system as of a past date. `python -m pytest -q` runs the tests.



# RPS: Ranked Probability Score

The **Ranked Probability Score (RPS)** evaluates probabilistic forecasts for ordered categories (such as low, medium, and high) by measuring the squared error between cumulative forecast probabilities and cumulative observed outcomes. Lower scores indicate better forecasts, with `0` representing a perfect forecast. [[1](https://stats.stackexchange.com/questions/112250/understanding-the-rank-probability-score), [2](https://en.wikipedia.org/wiki/Scoring_rule), [3](https://dtcenter.org/book/export/html/2989), [4](https://charts.ecmwf.int/products/mofc_multi_verification_probability_family_rpssmap)]

## How It Works

- **Order sensitivity:** Unlike standard probability scores that treat every incorrect category equally, **RPS penalizes forecasts more when probability mass is placed far from the actual outcome**. For example, predicting probabilities `(0.5, 0.5, 0)` when class `1` occurs yields a better RPS (`0.25`) than predicting `(0.5, 0, 0.5)` (`0.50`).
- **Cumulative differences:** RPS compares cumulative probabilities across ordered bins rather than isolated point probabilities.
- **Relationship to Brier Score:** When there are only two ordered categories, RPS reduces to the Brier score. [[1](https://en.wikipedia.org/wiki/Scoring_rule), [2](https://journals.ametsoc.org/view/journals/mwre/98/12/1520-0493_1970_098_0917_trpsat_2_3_co_2.xml), [3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12818272/), [4](https://dtcenter.org/book/export/html/2989), [5](https://charts.ecmwf.int/products/mofc_multi_verification_probability_family_rpssmap)]

As noted in a comment by a single user on [Cross Validated](https://stats.stackexchange.com/questions/112250/understanding-the-rank-probability-score), RPS accounts for both the location and spread of a forecast distribution and is often more useful for comparing models than judging a single model in isolation. For formal definitions and comparisons, see [Wikipedia's scoring rule overview](https://en.wikipedia.org/wiki/Scoring_rule) and [Lokad's CRPS guide](https://www.lokad.com/continuous-ranked-probability-score/). [[1](https://stats.stackexchange.com/questions/112250/understanding-the-rank-probability-score), [2](https://en.wikipedia.org/wiki/Scoring_rule), [3](https://www.lokad.com/continuous-ranked-probability-score/)]

