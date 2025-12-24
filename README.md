# Tipping Paper Code — README (Terminal Quickstart)

This repository runs a simple, but powerful, agent-based simulation of green consumption tipping with:
- two firms (different “greenness” shares),
- heterogeneous households,
- habit formation + peer influence,
- a carbon tax **τ (tau)** and green subsidy **σ (sigma)** that can be switched on/off mid-run.

---

## 1) Requirements

-  See environment.yml
---

## 2) Project structure

- `main.py` 
- `economy.py`
- `firm.py`
- `household.py`
- `environment.yml`

---

python3 main.py --cycles 100 --households 500 --seed 42

---

## 3 )Core CLI arguments

### Common parameters:

- cycles : number of time steps (e.g., 100)

- households : number of households (e.g., 500)

- seed : RNG seed for reproducibility (e.g., 42)

- tau : carbon tax level before the switch (e.g., 0.10)

- sigma : green subsidy level before the switch (e.g., 0.05)

### Policy switching (counterfactual / hysteresis tests):

- switch_cycle : cycle where the policy changes (e.g., 30)

- tau_after : carbon tax after the switch (e.g., 0.00)

- sigma_after : green subsidy after the switch (e.g., 0.00)

## Usage examples

### A) Baseline run (no policy)
python3 main.py --cycles 100 --households 500 --seed 42 --tau 0.0 --sigma 0.0

### B) Policy-on for the whole run
python3 main.py --cycles 100 --households 500 --seed 42 --tau 0.10 --sigma 0.05

### C) Policy-on then policy-off (our counterfactual “switch-off” test)

python3 main.py --cycles 100 --households 500 --seed 42 --tau 0.10 --sigma 0.05 --switch_cycle 60 --tau_after 0.0 --sigma_after 0.0

### D) Policy-off then policy-on (reverse pathway / hysteresis direction test)
python3 main.py --cycles 100 --households 500 --seed 42 --tau 0.0 --sigma 0.0 --switch_cycle 30 --tau_after 0.10 --sigma_after 0.05

### E) Reduce noise (bigger population)
python3 main.py --cycles 100 --households 2000 --seed 42 --tau 0.10 --sigma 0.05 --switch_cycle 30 --tau_after 0.0 --sigma_after 0.0



python3 main.py --cycles 100 --households 500 --seed 42 \
  --tau 0.10 --sigma 0.05 \
  --tau_after 0.0 --sigma_after 0.0 \
  --switch_cycles 30 80 \
  --save_csv


python main.py --mc --mc_runs 20 --seed_start 1 \
  --households 500 --cycles 100 \
  --tau 0.10 --sigma 0.05 \
  --tau_after 0.0 --sigma_after 0.0 \
  --rho 0.1 --gamma 0.5 --delta 0.5 \
  --switch_cycles 90 --save_csv \
  --results_dir sens_rho_01