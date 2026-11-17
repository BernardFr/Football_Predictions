# Usage

```python
cd ~/Documents/Code/Football_Predictions
football-predictions update
football-predictions lineups
football-predictions --help
football_predictions.py lineups-backtest
football_predictions.py lineups
football_predictions.py --help      # every command and option
```



# File Layout

```
Football_Predictions/
├── pyproject.toml              ← dependencies + the football-predictions command
├── requirements.txt            ← now just "-e .[test]", so the old install line still works
├── README.md
├── src/
│   ├── football_predictions.toml   ← your config, same place as before
│   └── football_predictions/       ← the package
│       ├── cli.py                  ← all the command-line code, with main()
│       ├── __main__.py             ← 2 lines, for python -m football_predictions
│       └── model.py, espn.py, lineups.py, …
└── tests/
```



# References

* API Football: https://dashboard.api-football.com/

# BF Q

can you build a model to forecast the results of the upcoming games of the English Premier League - for each game provide (a) Win-Draw-Lose prediction and (b) score differential ... include level of confidence. For data, use the history of the past 2 seasons. Run a backtest on this season's games so far. Build a feedback loop to update the model based on each game (including this season's game). Make this generic so that it can be applied to other soccer leagues. Before writing code, explain what model you will select and why - as well as how you plan to handle multiple leagues: e.g. 1 model per league, or a generic model with finetuning per league. Feel free to ask questions

------



Here's the plan for your review. I haven't written any code yet.

## Model: a time-weighted Dixon-Coles goals model

Each team gets an **attack** rating and a **defence** rating, and the league gets a **home advantage** term. Together these give each side's expected goals in a match. Goals are modelled as Poisson, with the Dixon-Coles correction for low scores (0-0, 1-0, 0-1, 1-1), which plain Poisson gets wrong. Older matches count for less through an exponential time decay, so recent form matters more than results from 18 months ago.

I chose it for these reasons:

- **One model answers both questions.** The model produces a full scoreline probability grid for every match. Summing that grid gives the win, draw and loss probabilities, and also the whole distribution of the goal difference. So "Arsenal by 1, with an 80% range of −1 to +3" comes from the same model as "Arsenal 52 / Draw 25 / Liverpool 23". There are no separately fitted pieces that could disagree with each other.
- **It suits the amount of data.** Two seasons is about 760 Premier League matches, which is enough for 20 teams × 2 ratings but too little for machine learning. XGBoost-style models on match features tend to overfit at this size and don't beat Dixon-Coles in the published comparisons.
- **The feedback loop is natural.** After each matchday the model refits on all data, with the new games carrying the most weight. That takes a second or two.
- **The ratings are interpretable.** You can see that a team's defence has slipped, which is good material for a post.

I rejected Elo ratings, because they give a strength gap but no scores, and a full Bayesian hierarchical model in PyMC, which is better on uncertainty but slow and fussy for a daily loop. I'd also skip the Chronos-type models from the Kalshi project. They forecast single time series, and a match is a contest between two teams.

**Confidence** shows up in three places:

- The W/D/L probabilities themselves.
- An 80% interval on the goal difference.
- A calibration check in the backtest, which tests whether things the model calls 60% actually happen about 60% of the time.

**Benchmark:** football-data.co.uk's free CSVs include bookmaker closing odds. The honest test is whether the model gets close to the bookmakers, measured by ranked probability score (RPS), the standard metric for W/D/L forecasts, plus log loss and accuracy. Like the Kalshi markets, the bookmakers will probably win overall. What matters is how close the model gets and where it falls short.

## Multiple leagues: same code, one fitted model per league

Team ratings only mean something inside their own league, because teams almost never play across leagues. A single model "fine-tuned" per league wouldn't share anything useful. So the setup is:

- **Generic code** driven by a TOML file with one block per league: data-source code, seasons and parameters.
- **League-level settings** (time decay, home advantage, low-score correction) start from defaults and are tuned for each league in the backtest. Home advantage really does differ between leagues.
- **Promoted teams** are the hard part. They have no Premier League history in your two seasons. I'd also load the second division (the Championship for England) and start promoted sides from their second-division rating minus a fitted "promotion penalty". Without that step, the model treats Sunderland like an average Premier League team.

## Backtest

This is walk-forward only. Each matchday of 2026-27 is predicted using only data available before it, then the model updates and moves on. That's the feedback loop running in replay. Because only about 110 matches have been played this season, the scores will be noisy. I'd also backtest all of 2025-26 (trained on 2023-24 and 2024-25) so the comparison with the bookmakers rests on a sample about four times bigger.

## Questions before I build

1. **Data source.** football-data.co.uk has results, shots, bookmaker odds and an upcoming-fixtures file for about 22 leagues, all free. Is that fine, or do you have a paid API you'd prefer?
2. **Second-division data for promoted teams.** I recommend loading it. Without it, predictions involving promoted sides will be poor for the first couple of months.
3. **The "pick".** Draws are almost never the single most likely outcome. If the pick is just the highest probability, the model will never predict a draw, even though about 25% of games end level. Should the pick be the most likely outcome, with the probabilities beside it, or should it call a draw when the two sides are close?
4. **Odds as an input.** I'd use bookmaker odds as a benchmark only. Feeding them in mostly reproduces the market. Agreed?
5. **Where it lives and what it outputs.** I can build it on your Mac next to forecast_tournament, with the same TOML-plus-Excel-report pattern. Would you also like a web page artifact showing the week's predictions?



------

# BF Answers

1/ football-data.co.uk is fine 
2/ Yes - load second division data
3/ Pick should be the most likely outcome
4/ Agree
5/ Create separate folder under ~/Documents/Code. Call it "Football_Predictions"

One more question, is there a convenient place to download the starting 11 for each game? If there is, could this be incorporated as an "enhanced prediction" once the starting 11 is known



------



I'll build the base model now in `~/Documents/Code/Football_Predictions` and look into lineup sources while I set up.

The cloud workspace and the shell on your Mac both block football-data.co.uk, so I'm checking what data I can reach for testing.

That goal-difference bookkeeping in `summary` is clumsy, so I'm rewriting it cleanly.

Fits take under 0.1 s and the ratings look sensible. Now tuning on 2024-25, then the walk-forward backtest on 2025-26.

Best ridge landed on the edge of the grid, so I'm widening it and re-running.

The 2025-26 backtest beats the base rate clearly (RPS 0.218 vs 0.234). The High band is a bit overconfident (it says 67%, hits 60%). Now tests and a full `update` cycle.

I've built the model and installed it in `~/Documents/Code/Football_Predictions`. I couldn't produce real predictions for this weekend: both my workspace and the shell on your Mac block football-data.co.uk. You'll need to run it once yourself in Terminal:

```bash
cd ~/Documents/Code/Football_Predictions
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m football_predictions download
python -m football_predictions tune      # about 30 seconds
python -m football_predictions update    # backtests this season, predicts the next round, writes reports/EPL_report.xlsx
```

After that, the feedback loop is just running `update` after each round of games. Each run:

- downloads the new results;
- scores every prediction logged before kickoff, which gives you a live track record separate from the backtest;
- refits the model on everything played so far;
- reruns this season's backtest;
- predicts the fixtures in the next 10 days.

**Test results.** To check it works, I ran it on an incomplete copy of the results hosted on GitHub. It has no betting odds, and it's missing all of this season after 20 September. I tuned the settings on 2024-25, then backtested all of 2025-26 match by match, using only games played before each one:

- **Ranked probability score** (lower is better): 0.218 for the model against 0.234 for always predicting the historical home/draw/away rates. Bookmakers usually score around 0.19–0.20, so expect them to win. Your first run will show by how much.
- **Accuracy of the pick:** 51%.
- **Confidence bands:** "Medium" and "Low" picks came true about as often as the model said they would. "High" picks were somewhat overconfident: the model said 67% on average and they came true 60% of the time.
- **Goal difference:** the 80% range contained the actual result 89% of the time, so it's a bit wide.
- **Promoted teams:** these start from their Championship form translated to Premier League level. This season's newcomers (Coventry, Ipswich, Hull) are already handled that way.
- **Draws:** because the pick is the single most likely outcome, a draw will almost never be picked, as we discussed. You can still see the draw probability for each game.

**Starting XIs.** No source is as convenient as football-data.co.uk. Teams announce lineups about an hour before kickoff, and the realistic option is API-Football:

- Its free plan allows 100 requests a day ([API-Football](https://www.api-football.com/news/post/how-to-optimize-api-sports-calls-and-quota-usage)). Live predictions would use about 10 a matchday.
- The catch is the backtest. Proving the lineup adjustment helps needs past lineups too, at one request per game. That's roughly 8 days of the free allowance for two seasons, or one month of a paid plan.
- Its lineups can also come in late, even after kickoff.
- Sportmonks sells "expected lineups" before the official announcement, but only on paid plans ([Sportmonks](https://www.sportmonks.com/blogs/premium-expected-lineups/)).

The model already has a slot for it. Each team's expected goals can be multiplied by a factor, and an automated version would rate each player and compare the announced XI with the team's usual one. You can use that slot by hand now through `state/EPL/adjustments.csv` (for example, cutting a team's attack by 10% when its striker is out). The README explains how. I'd keep the lineup version only if it improves the backtest; the gain is usually modest except when a key player is missing or a team rotates heavily.

If you sign up for an API-Football key, I can build the lineup version next.

I'll build the lineup-enhanced prediction on top of the existing model, starting by checking the API-Football details that matter for a free-tier budget.

The free API-Football plan may not cover the current season, which would block the lineup plan. Checking that and the alternatives before building.

I haven't built anything yet. Two separate developer reports from the past two months say the free API-Football plan can't access the current season. Before I write code around your key, you need to decide how we get lineups.

**What I found.** On the free plan, a request for season 2026 reportedly comes back empty with this error: *"Free plans do not have access to this season, try from 2022 to 2024"* ([report 1](https://github.com/MatrixLabSports/MATRIX-LAB/issues/3), [report 2](https://github.com/nimeshjm/fantasy-football/issues/58)). If that's right, your free key gives no lineups for this season or last, which are exactly the two the model needs. API-Football's [pricing page](https://www.api-football.com/pricing) says only that "all plans include all competitions and endpoints," so I can't confirm the restriction from their own docs. I also can't call their API from here. You can check it in Terminal with one request:

```bash
curl -s -H "x-apisports-key: $FOOTBALL_API_KEY" \
  "https://v3.football.api-sports.io/fixtures?league=39&season=2026&next=1" | head -c 400
```

If the response contains `"plan": "Free plans do not have access..."`, the restriction applies to you.

**The two options that work for this season:**

1. **ESPN's public data feed.** It's free, needs no key, and covers the current season for the EPL and the other leagues. Each match summary lists the starters with their positions and substitution minutes, plus goals, assists and shots per player. That's enough to measure how strong an announced XI is compared with the team's usual one. The open-source soccerdata library reads lineups from it ([soccerdata](https://soccerdata.readthedocs.io/en/latest/datasources/ESPN.html)), and developers are using it this season for exactly this reason. The catch is that the feed is undocumented, so ESPN could change or block it without notice.
2. **A paid API-Football plan.** Pro is 7,500 requests a day; I couldn't find a published price. It's documented and stable, and the backtest would fetch two seasons of lineups in about 45 requests. Your existing key works the moment you upgrade.

Either way, the lineup adjustment uses the same method. For each match, the model compares the announced XI with the team's usual starters. It looks at how much of the usual playing time is missing, how much of the team's goal output is missing, and whether there's a change in goal. The size of each effect is estimated from past seasons on top of the base model. It's then tested walk-forward the same way as the base model, and kept only if it improves the ranked probability score.

Which do you want? I'd go with ESPN: it's free, and if ESPN ever breaks the feed, switching to API-Football later only means replacing the download step.



​			>>> *API-Football does not work for me either - use ESPN*

I'll build the lineup layer on ESPN's feed. First, checking whether I can reach it from here so I can work against real responses.

Both machines block ESPN, so I'll pull sample responses through the web reader to get the exact JSON layout.

That `norm`/`eval` hack is ugly, so I'm rewriting the matching helpers properly.

Now the core lineup module: features, the adjustment model, and walk-forward evaluation.

Now config, ledger variants, and the CLI commands.

Tidying two awkward lines, then wiring the commands into the CLI.

Now offline tests that fake the ESPN cache. A synthetic league has a planted lineup effect (missing star striker, backup goalkeeper), and the model should recover it.

The planted effects are recovered with the right signs, but the adjusted forecasts score worse. Investigating before I trust it.

The base pipeline is unaffected. Now a CLI-level smoke test of `lineups-backtest`, using the synthetic league written as both football-data CSVs and an ESPN cache.

The ESPN sync and team mapping worked (all 12 names mapped). The backtest of the earlier season just needs one more empty history season in the fake cache.

`lineups-backtest` works end to end: features for 231 of 264 matches, effects recovered, report written. To smoke-test the live `lineups` command, I'm making "now" replayable with `--today`.

Smoke-testing the live `lineups` command: the last round of the synthetic season becomes "upcoming", with only starting XIs published.

The command correctly tried to refresh recent ESPN days. That's the right live behaviour, so the harness just needs to stub that refresh too.

You've renamed the package folder to `src` and added a notes file, so I'll fit the update to that layout. Reading your notes first.

I've added lineup-enhanced predictions using ESPN's free feed and installed the update in `~/Documents/Code/Football_Predictions`. ESPN is blocked from here and from the shell on your Mac, so it hasn't been run against real ESPN data yet. The first run in your Terminal will be that test.

You'd renamed the package folder to `src`, so I put the update there. All commands are now `python -m src …` instead of `python -m football_predictions …`, and your notes file is untouched.

**How the adjustment works.** For each team, the announced XI is compared with how the team has lined up over its last 19 games, looking at three things:

- **Missing regulars:** how much of the team's usual playing time is absent from the XI.
- **Attacking output:** the XI's goals plus assists per 90 minutes, compared with the team's usual starting lineups.
- **Goalkeeper:** whether the regular keeper is missing.

These scale both sides' expected goals from the base model. The size of each effect is estimated from past matches and re-estimated before every match date. Lineup predictions go into the track record separately, so the report scores them against the base model on live games as well as in the backtest.

**Checks I could run here:**

- **Synthetic league:** I built a fake league in ESPN's data format with planted effects (a star striker missing, a backup keeper starting). The model recovered both effects with the right sign and predicted goals better on games it hadn't seen.
- **Ranked probability score:** this didn't improve on about 75–125 test games. That's too few to show a small gain, so the comparison also reports the goal-level fit, which responds sooner.
- **Live command:** run against fake data with published XIs, it adjusted only the games with a real change. For example, when Wolves started their backup keeper, Man City's win chance went from 58% to 70%.
- **Tests:** all 13 pass.

**To run on your Mac:**

```bash
cd ~/Documents/Code/Football_Predictions && source .venv/bin/activate
python -m src download && python -m src tune && python -m src update   # base model, if not yet run
python -m src lineups-backtest   # first run: ~1,000 ESPN requests, about 10 minutes; cached afterwards
python -m src lineups            # about an hour before kickoff, once XIs are out
```

**What to send me if something breaks:**

- **A traceback.** ESPN's data format is undocumented, so a mismatch is the most likely first problem.
- **A 403 error.** ESPN refuses some user-agent strings. Changing the `user_agent` setting under `[lineups]` in `config.toml` usually fixes it.
- **An "unmatched team" warning.** That's a name the automatic ESPN-to-football-data matching couldn't pair. Add it to `state/EPL/team_map_overrides.csv`.

Two things only real data can tell us:

- **Whether it helps.** If the `lineups-backtest` comparison shows the adjustment isn't improving forecasts, stick with the base predictions.
- **How early ESPN publishes lineups.** If `lineups` reports "not published yet", run it again closer to kickoff.

The Premier League returns next weekend after the international break, so that's the first chance to try `lineups` live.

Sources:

- [API-Football free-plan season restriction (report 1)](https://github.com/MatrixLabSports/MATRIX-LAB/issues/3)
- [API-Football free-plan season restriction and ESPN as an alternative (report 2)](https://github.com/nimeshjm/fantasy-football/issues/58)
- [soccerdata: ESPN reader (endpoints and lineup fields)](https://soccerdata.readthedocs.io/en/latest/datasources/ESPN.html)
- [wc-data: ESPN JSON structures](https://docs.rs/wc-data/0.2.1/src/wc_data/backends/espn.rs.html)



# Added FA Cup, EFL Cup and Community Shield Results

I'll check how ESPN covers the three cups (league codes, extra time and penalties, neutral venues) before deciding how to fold them in.

Building it: cup results from ESPN, scored at 90 minutes, with neutral venues and a tunable weight.

`tune` needs to search the cup weight too, but adding it to the full grid would mean 120 combinations. Checking your real tuning results to see where to trim the grid.

Your real results show the best ridge (3.0) and feeder weight (0.5) both landed on the edge of the grid, so I'll widen those and trim the rest to keep `tune` to about 54 combinations.

Now the README, and tests for extra time, neutral venues, cup weighting and team filtering.

A mistake in my test: January 2025 belongs to the 2024-25 season, not 2025-26. Fixing that, and making `cups.load` safe when no logger is passed.

All 18 tests pass. Now a command-level check: `download` and a backtest on the synthetic league, with FA Cup ties in ESPN's format, including a neutral-venue tie decided on penalties.

Your copies are unchanged since my last update, so it's safe to overwrite them. Installing the 10 changed or new files:

I've added FA Cup, EFL Cup and Community Shield results to the model's training data, and installed the update on your Mac. All 20 files there match my tested versions, and the 18 tests pass. The cup games come from ESPN, because football-data.co.uk only has league games.

**How cup games are handled:**

- **Scored at 90 minutes.** A tie that went to extra time or penalties counts as a draw at its 90-minute score.
- **Neutral venues** get no home advantage. That covers the Wembley semi-finals and finals and the Community Shield.
- **Only Premier League and Championship teams.** A tie against a League One or non-league side is dropped, because the model has almost no data on those teams.
- **Kept separate from league games.** Cup games get their own scoring level and home advantage, so they don't skew the league figures. They also don't count as a team's division, and the lineup features still use league games only, since cup lineups are often rotated.
- **Weighted by the backtest.** A new setting, `cup_weight`, controls how much cup games count, and `tune` picks it (0, 0.5 or 1). If cups don't improve forecasts, `tune` will set the weight to 0, which switches them off.

The main benefit is Premier League v Championship ties, which measure the gap between the two divisions directly. That should sharpen the ratings of promoted teams.

**What to run:**

```bash
football-predictions download   # now also lists cup games found per season and competition
football-predictions tune       # re-tune: picks the cup weight (a few minutes)
football-predictions update
```

- **Wider tuning grid:** your real tuning results put the best shrinkage (3.0) and second-division weight (0.5) at the edge of the values tried, so the grid now extends past both. It's 54 combinations in all.
- **The EFL Cup and Community Shield codes are unconfirmed.** I'm using `eng.league_cup` and `eng.charity`; only `eng.fa` is confirmed by ESPN's own pages. If `download` shows a warning for either of the other two, the code is wrong. It's set under `cups` in `src/football_predictions.toml`.
- **First run:** fetching the cup data takes a few dozen extra requests to ESPN. After that it's cached.

**Your lineup backtest result:** on 279 games, adjusting for the announced XIs made no measurable difference. RPS was 0.2116 against 0.2115 for the base model; it was slightly better this season and slightly worse last season. The missing-regulars and attacking-output effects point the right way, but the goalkeeper effect came out backwards. That looks like noise at this sample size, so I'd stick with the base predictions for now and check again later in the season.

Sources:

- [ESPN: FA Cup (league code ENG.FA)](https://www.espn.com/soccer/teams/_/league/ENG.FA/fa-cup)
- [wc-data: ESPN status handling for extra time and penalties](https://docs.rs/wc-data/0.2.1/src/wc_data/backends/espn.rs.html)



# Score Histogram

*create a separate script that generates a histogram (both excel file and chart) of the scores of the past 2 seasons and this season - Premier League only*

I'll add a second command, `football-score-histogram`. It reads the Premier League results the project already downloads (2024-25, 2025-26 and this season so far) and writes an Excel workbook with a chart in it, plus a PNG of the chart.

Runs cleanly on the mirror data. Checking the PNG and the Excel chart visually:

The native Excel chart renders, but the column headers are cut off. Fixing the widths and wrapping, alongside the colours:

Both charts look right now. Checking your Mac copies, then installing:

I've added the histogram as a separate command, `football-score-histogram`, and installed it in your project. It covers Premier League games only, for 2024-25, 2025-26 and this season so far.

It needs a one-time reinstall to register the new command and add the charting library (matplotlib):

```bash
cd ~/Documents/Code/Football_Predictions && source .venv/bin/activate
pip install -e ".[test]"
football-score-histogram
```

**It writes two files to `reports/`:**

- **`EPL_score_histogram.xlsx`**
  - **Scorelines:** each final score (home-away), most common first, with match counts and the share of matches for each season and all three combined. A native Excel bar chart sits below the table.
  - **Summary:** per season, goals per match, home win / draw / away win rates, and the 0-0 rate.
  - **Home x away grid:** match counts by home goals and away goals, with 6 or more pooled.
  - **Matches:** the full list of games used.
- **`EPL_score_histogram.png`:** the same chart as a standalone image. Seasons run from light (oldest) to dark (current).

**Choices you might want to change:**

- **Shares, not counts:** seasons are compared as a share of matches, because this season is only about 120 games in.
- **Top 20 scorelines:** the 20 most common are shown, and the rest are grouped as "Other".
- **Options:** `--top 25` shows more scorelines, `--seasons 5` goes further back, and `--league BUN` runs it for the Bundesliga.
- **Data:** it re-downloads this season's results each run. `--no-refresh` uses the cached copy instead.

I tested it on the incomplete GitHub copy of the results, which is why I haven't quoted any numbers. That copy shows no 0-0 draws at all in 2025-26, which is almost certainly a gap in that data, so check the figures on your Mac's football-data run instead. All 19 tests pass, including a new one for the histogram tables.

*Modify the chart so that each of the 3 seasons can be individually selected, or in combination - add histogram for all 3 seasons combined - change the display order to rank, left to right, most severe defeat for home team to biggest victory for home team - for scores that have the same goal difference (e.g. 2-1 and 1-0) put the highest total score left for defeats and highest total score right for home victories: e.g 1-2, 0-1, ... , 1-0, 2-1, ...*

