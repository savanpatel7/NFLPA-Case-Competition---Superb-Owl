# AEGIS: Acute Endurance Guidance for Injury Safeguarding

A biomechanics-inspired predictive framework for NFL soft-tissue injury prevention, developed for the NFLPA Data Analytics Case Competition.

**Authors:** Lucca Ferraz, Savan Patel — Team SuperbOwl, Rice University

**Award: Runners-Up, NFLPA Data Analytics Case Competition**

---

## Contents

| File | Description |
|------|-------------|
| `AEGIS.pdf` | Full presentation with methodology, results, and CBA recommendations |
| `nflpafinalscript.R` | R source code for data processing, modeling, and causal analysis |

---

## Overview

Soft-tissue injuries account for 70% of all football-related physical therapy visits and are increasing year over year across the NFL. AEGIS uses publicly available play-by-play data to construct biomechanical stress proxies for each player each week, then predicts the probability of a soft-tissue injury the following week. The framework is designed to give players, agents, and the NFLPA actionable data to advocate for workload management and health protections.

---

## Data

Sourced via the `nflreadR` package in R:

- Play-by-play data from the 2016–2024 NFL seasons (435,483 plays)
- Weekly NFL injury reports
- Players included only if they participated in 20% or more of their team's offensive or defensive snaps

Injury scope was limited to soft-tissue diagnoses: hamstring, calf, groin, quadriceps, and non-contact knee, achilles, and ankle injuries. The target variable is a binary indicator for injury in the following week.

---

## Biomechanical Stress Categories

Three play-level biomechanical proxies are constructed. Players are only credited if directly involved in the play.

**Deceleration Events** — eccentric contractions are 1.5–3.0x more forceful on soft tissue than concentric contractions
- Tackles for loss, deep incompletions (>15 air yards), deep completions with low YAC (>15 air yards, <2 YAC), safeties

**Explosive Plays** — high-velocity movements elevate soft-tissue injury risk
- Runs >12 yards, receptions >16 yards, plays with EPA >1.5

**Collisions** — high-impact contact generates peak force and direct trauma
- Sacks, QB hits, solo and assisted tackles, forced fumbles, receivers and RBs upon tackle

These are aggregated into four weekly metrics per player:
- **Acute Load** — workload in the past 7 days
- **Chronic Load** — average weekly workload over the past 28 days
- **ACWR** (Acute-Chronic Workload Ratio) — acute ÷ chronic load; spikes indicate elevated risk
- **Stress Accumulation** — injuries over past 4 weeks × ACWR; captures vulnerability during recovery

---

## Methodology

**Predictive Modeling (XGBoost)**
- Forecasts probability of soft-tissue injury in the following week using biomechanical, contextual, and historical features
- Class imbalance addressed via `scale_pos_weight` parameter (injuries comprise <6% of observations)
- Evaluated using AUC-PR, which is more informative than AUC-ROC for highly imbalanced datasets
- All performance metrics calculated out of sample

**Causal Inference (Propensity Score Matching)**
- Isolates the independent effect of high deceleration exposure from confounders including position, snap volume, and injury history
- Mimics a randomized experiment by matching high-exposure players to similar low-exposure counterparts

---

## Results

- AEGIS improves AUC-PR by **44% over baseline** (assigning the league-average 5.9% injury rate to all observations)
- Correctly identified **57.6% of all next-week injuries** out of sample
- Top predictors: Stress Accumulation, Total Plays, Full Acute Workload, ACWR Position Z-Score, Chronic Explosive Plays
- Propensity score matching found high deceleration exposure associated with a **0.72% absolute increase** in injury probability (95% CI: −0.27% to +1.71%), corresponding to a **12% higher relative risk**
- For every 139 player-weeks of high deceleration exposure, one additional injury is expected

---

## CBA Recommendations

- Position-specific biomechanical safety standards mandated by the CBA, with options for reduced practice workloads when thresholds are exceeded
- Biomechanical Data Resumes for free agency, giving players and agents documented histories of load, resilience, and past safety threshold violations
- Salary protection for players injured during weeks of significant threshold exceedance
- League-wide automated load management alerts to players and the NFLPA

---

## Projected Impact

- 10–15% reduction in soft-tissue injuries based on the 12% elevated relative risk from deceleration exposure
- Potential career extension of 1–2 seasons for players who avoid catastrophic and cumulative injuries
- Contract negotiation leverage for players and agents to demonstrate injury causation from overuse

---

## Future Work

- Integrate Next Gen Stats tracking data for true biomechanical measurements (actual deceleration, velocity, collision force)
- Develop an athlete-facing digital dashboard for real-time load and injury data access
- Research physiological risk factors including limb length, height/weight, bone density, strength, and speed

---

## Data Source

NFL play-by-play and injury report data via the `nflreadR` R package
