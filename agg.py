import pandas as pd
from pathlib import Path


RESULTS_DIR = Path.home() / "Desktop" / "Tipping-point" / "Results" / "sens_delta_07"
SUMMARY_FILE = RESULTS_DIR / "runs_summary.csv"
TIMESERIES_FILE = RESULTS_DIR / "timeseries.csv"

OUT_DIR = RESULTS_DIR / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(SUMMARY_FILE)

num_cols = [
    "seed", "households", "cycles",
    "tau", "sigma", "tau_after", "sigma_af ter",
    "switch_cycle", "post_window",
    "s_base_mean", "s_post_mean", "H_delta",
    "rho", "gamma", "delta",
]

for c in num_cols:
    if c in summary.columns:
        summary[c] = pd.to_numeric(summary[c], errors="coerce")


POTENTIAL_KEYS = [
    "tau", "sigma",
    "tau_after", "sigma_after",
    "switch_cycle",
    "rho", "gamma", "delta",
]

SCENARIO_KEYS = [k for k in POTENTIAL_KEYS if k in summary.columns]



agg = (
    summary
    .groupby(SCENARIO_KEYS, dropna=False)
    .agg(
        n_runs=("seed", "count"),
        H_mean=("H_delta", "mean"),
        H_median=("H_delta", "median"),
        H_std=("H_delta", "std"),
        H_min=("H_delta", "min"),
        H_max=("H_delta", "max"),
        share_positive=("H_delta", lambda x: (x > 0).mean()),
        s_base_mean=("s_base_mean", "mean"),
        s_post_mean=("s_post_mean", "mean"),
    )
    .reset_index()
)

# Save aggregated table
agg_path = OUT_DIR / "scenario_level_summary.csv"
agg.to_csv(agg_path, index=False)

print(f"[OK] Scenario-level summary saved to: {agg_path}")


if TIMESERIES_FILE.exists():
    ts = pd.read_csv(TIMESERIES_FILE)

    # numeric coercion
    for c in ["cycle", "s1", "s2", "tau_t", "sigma_t"]:
        if c in ts.columns:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

    ts_agg = (
        ts
        .groupby(SCENARIO_KEYS + ["cycle"], dropna=False)
        .agg(
            s1_mean=("s1", "mean"),
            s1_std=("s1", "std"),
            s2_mean=("s2", "mean"),
            s2_std=("s2", "std"),
        )
        .reset_index()
    )

    ts_path = OUT_DIR / "scenario_level_timeseries.csv"
    ts_agg.to_csv(ts_path, index=False)

    print(f"[OK] Scenario-level time series saved to: {ts_path}")
else:
    print("[INFO] No timeseries.csv found — skipping time series aggregation.")


print("\n=== Scenario-level overview ===")
print(
    agg[
        [
            "tau", "sigma", "switch_cycle",
            "H_mean", "H_std", "share_positive"
        ]
    ]
    .sort_values(["tau", "sigma", "switch_cycle"])
)
