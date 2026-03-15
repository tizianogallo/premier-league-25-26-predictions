# ⚽ Premier League 2025/26 Match Predictor

Predicting Premier League match outcomes and simulating the 2025/26 title race using 11 seasons of historical data.

---

## 🔍 Overview

This project builds two machine learning models to predict Premier League matches:

1. 🏆 **Win / Draw / Loss** — predict the result of any match
2. 🎯 **Over / Under 2.5 Goals** — predict whether a match will have more or fewer than 2.5 total goals

Both models are trained on 10 seasons of historical data (2014/15 → 2024/25) and tested on the current 2025/26 season. The project ends with a **Monte Carlo simulation** of the remaining 2025/26 fixtures to estimate each team's probability of winning the title, finishing top 4, or being relegated.

---

## 📊 Data

**Source:** [football-data.co.uk](https://www.football-data.co.uk/englandm.php)

- 📅 12 seasons of Premier League data (2014/15 → 2025/26)
- ⚽ 4,471 matches total
- 📋 Features used: goals, shots on target, corners, match result

> 💾 To run this notebook, download all season CSV files from the link above and place them in a `data/` folder inside the project directory. Name them `2014-15.csv`, `2015-16.csv`, ... `2025-26.csv`.

---

## ⚙️ Setup

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
```

---

## 📓 Notebook Structure

| Section | Description |
|---|---|
| 1️⃣ Load & Explore | Load all 12 seasons, clean and verify |
| 2️⃣ Feature Engineering | Build rolling form features for each team |
| 3️⃣ Train Models | Train XGBoost + Logistic Regression |
| 4️⃣ Evaluate Models | Holdout validation + TimeSeriesSplit CV |
| 5️⃣ 2025/26 Predictions | Compare predictions to real results |
| 6️⃣ Title Race Simulation | Monte Carlo simulation of remaining fixtures |
| 7️⃣ Visualizations | Charts and plots |

---

## 🛠️ Feature Engineering

For every match, the model only knows what happened **before** that match. For each team going into a game, we compute rolling averages over their last 5 matches:

| Feature | Description |
|---|---|
| 📈 `form_pts` | Average points per game |
| ⚽ `form_gf` | Average goals scored |
| 🥅 `form_ga` | Average goals conceded |
| 🎯 `form_sot_f` | Average shots on target for |
| 🛡️ `form_sot_a` | Average shots on target against |
| ↕️ `pts_diff` | Home team form_pts minus away team form_pts |
| ↕️ `gf_diff` | Goal scoring differential |
| ↕️ `ga_diff` | Goal conceding differential |
| ↕️ `sot_diff` | Shots on target differential |

> ✅ No data leakage — rolling features are computed strictly on matches played before the current one, ordered by date.

---

## 🤖 Models

### ⚡ XGBoost
Gradient boosting model — learns non-linear patterns and feature interactions. Industry standard for tabular prediction tasks.

```
n_estimators     = 400
max_depth        = 4
learning_rate    = 0.04
subsample        = 0.8
colsample_bytree = 0.7
min_child_weight = 5
```

### 📉 Logistic Regression
Linear model — predicts outcome probabilities directly from a weighted combination of features. Simpler than XGBoost but naturally outputs well-calibrated probabilities, which is important for the Monte Carlo simulation.

```
max_iter = 1000
scaler   = StandardScaler (features normalised before fitting)
```

---

## 📋 Evaluation

### 🔒 Method 1 — Holdout Validation
Train on 2014/15 → 2024/25, test on 2025/26 (291 matches the model never saw).

| Model | Win/Draw/Loss Accuracy | Over/Under Accuracy |
|---|---|---|
| ⚡ XGBoost | 43.3% | 58.1% |
| 📉 Logistic Regression | 46.7% | 56.4% |

### 🔄 Method 2 — TimeSeriesSplit Cross Validation (5 folds)
More robust estimate — trains on progressively more seasons and tests on the next one, rotating through the full dataset without breaking time order.

| Model | Win/Draw/Loss CV Mean | Over/Under CV Mean |
|---|---|---|
| ⚡ XGBoost | 47.8% (±2.2%) | 53.4% (±0.7%) |
| 📉 Logistic Regression | 51.7% (±2.6%) | 54.1% (±2.5%) |

![Model Comparison](outputs/model_comparison_full.png)

### 🔎 Analysis

**Win/Draw/Loss**

Both models beat the naive baseline of ~44% (always predict home win). Logistic Regression outperforms XGBoost on both holdout and CV mean, which suggests the relationship between rolling form features and match outcomes is largely linear. XGBoost's additional complexity does not add meaningful value here.

The draw class is consistently the hardest to predict — Logistic Regression scores near 0% recall on draws, XGBoost only slightly better at 3%. This is a well-known limitation of all football prediction models. Draws carry the least signal in pre-match features and are essentially noise from a statistical standpoint.

**Over/Under 2.5 Goals**

Both models comfortably beat the 50% baseline. XGBoost edges Logistic Regression on holdout (58.1% vs 56.4%) but Logistic Regression wins on CV mean (54.1% vs 53.4%). The extremely low standard deviation on XGBoost's CV (±0.7%) indicates very consistent performance across seasons.

**🏆 Why Logistic Regression was chosen for the simulation**

Despite XGBoost performing better on holdout Over/Under, Logistic Regression was used for the Monte Carlo simulation for two reasons. First, it outperforms XGBoost on Win/Draw/Loss across both validation methods. Second, Logistic Regression produces better-calibrated probabilities — meaning a 60% predicted probability genuinely reflects a 60% chance. In a simulation running 10,000 seasons, small miscalibrations compound across every remaining fixture, making calibration more important than raw accuracy.

---

## 📅 2025/26 Season

### 📊 Current Standings (Matchday 29)

![Standings Table](outputs/standings_table.png)

| Pos | Team | Played | Pts | GD |
|---|---|---|---|---|
| 🥇 | **Arsenal** | 30 | **67** | +37 |
| 🥈 | Man City | 29 | 60 | +32 |
| 🥉 | Man United | 29 | 51 | +11 |
| 4️⃣ | Aston Villa | 29 | 51 | +5 |
| 5️⃣ | Chelsea | 29 | 48 | +19 |
| 6️⃣ | Liverpool | 29 | 48 | +9 |

### 🔴 Arsenal Season Journey

![Arsenal Season](outputs/arsenal_season.png)

Arsenal have been the most consistent team in the league — dominant from October onwards with very few dropped points. The model predicted their results correctly 63.3% of the time, well above the overall accuracy of 46.7%. The main misses were draws and away defeats that the model expected them to win.

---

## 🎲 Title Race Simulation

**Method:** Monte Carlo simulation — 10,000 full season runs using Logistic Regression probabilities for every remaining fixture.

![Title Race](outputs/title_race.png)

| Team | 🏆 Title % | 🎯 Top 4 % | ⬇️ Relegated % |
|---|---|---|---|
| 🔴 **Arsenal** | **87.4%** | 100.0% | 0.0% |
| 🔵 Man City | 12.4% | 99.7% | 0.0% |
| 🔴 Man United | 0.1% | 66.6% | 0.0% |
| 🟣 Aston Villa | 0.0% | 44.9% | 0.0% |
| 🔵 Chelsea | 0.0% | 42.2% | 0.0% |
| 🔴 Liverpool | 0.0% | 37.1% | 0.0% |
| 🟠 Wolves | 0.0% | 0.0% | 99.7% |
| 🟣 Burnley | 0.0% | 0.0% | 97.4% |
| 🔵 West Ham | 0.0% | 0.0% | 32.0% |
| 🔴 Nott'm Forest | 0.0% | 0.0% | 30.8% |
| 🤍 Tottenham | 0.0% | 0.0% | 30.6% |

🏆 **Title race:** Arsenal are overwhelming favourites at 87.4%. With a 7-point lead and 8 games remaining, the simulation reflects the near-mathematical certainty of their title win. Man City at 12.4% are the only realistic challenger.

🎯 **Top 4:** Arsenal and Man City are both certain. The battle for 3rd and 4th is wide open — Man United (66.6%), Aston Villa (44.9%), Chelsea (42.2%), and Liverpool (37.1%) all still in contention.

### ⚠️ Relegation Battle

![Relegation](outputs/relegation.png)

Wolves (99.7%) and Burnley (97.4%) are effectively down. The third spot is a tight three-way battle between West Ham (32.0%), Nott'm Forest (30.8%), and Tottenham (30.6%).

---

## ⚠️ Limitations

- 🤝 **Draws are systematically underpredicted** — a known limitation of all football models
- 🏥 **No injury or suspension data** — squad availability is not modelled
- 🏠 **Home advantage** is captured implicitly through rolling form but not as an explicit feature
- 📈 **Newly promoted teams** have limited historical PL data which reduces feature quality
- 🔄 **Managerial changes** break rolling form continuity

---

*📊 Data: football-data.co.uk · 🤖 Models: XGBoost, Logistic Regression · 🎲 Simulation: Monte Carlo (10,000 runs)*
