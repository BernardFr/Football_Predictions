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

Run `update` after each round of games (or daily). It:

1. downloads the latest results,
2. scores every prediction you logged before kickoff (`state/EPL/ledger.csv`), which is your live track record,
3. refits on all matches played so far, with recent games weighted most, and re-runs the walk-forward backtest of the season,
4. predicts the fixtures in the next 10 days (from football-data's `fixtures.csv`) and logs them,
5. rebuilds the Excel report.

Re-tune once a season (`tune`), e.g. in July once the previous season is complete.

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

The **Backtest** sheet shows how much to trust these. It compares the model with closing
bookmaker odds and with a base rate, using ranked probability score (RPS; lower is better), log
loss and accuracy. It also shows the hit rate of each confidence band and how often results fell
inside the 80% interval.

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

The **Player impact** sheet has one column per team. It shows these tables:

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
(`.github/workflows/pages.yml`), which publishes the site at
https://bernardfr.github.io/Football_Predictions/. The site has four tabs:

| Tab | Content |
|---|---|
| Predictions | The upcoming games (from `upcoming.csv`), then the last round: the predictions logged before kickoff vs. the results. Picks are shaded green (right) or red (wrong). |
| Season predictions | Empty for now. |
| Impact players | The tables of the report's Impact Players sheet, one team at a time. A link such as `players.html#chelsea` opens one team. |
| Scoring statistics | The scoreline histogram, with only the current season ticked. |

The last round is the base model's scored games from the latest result back 4 days (`LAST_ROUND_DAYS` in
`website.py`). The pages come from the templates in `src/football_predictions/templates/`; edit their text
there.

## Commands

```
football-predictions download|tune|backtest|predict|update|report|lineups-backtest|lineups|site
    [--league EPL | --all] [--season 2025-26] [--source openfootball] [--today YYYY-MM-DD]
```

`--source openfootball` uses a GitHub-hosted results mirror (no odds, can lag by weeks).
`--today` replays the system as of a past date. `python -m pytest -q` runs the tests.
