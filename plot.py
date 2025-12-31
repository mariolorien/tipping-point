import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration

BASE_DIR = Path.home() / "Desktop" /"Tipping-point" / "Results"/ "sens_delta_07" / "aggregated"

SUMMARY_FILE = BASE_DIR / "scenario_level_summary.csv"
TIMESERIES_FILE = BASE_DIR / "scenario_level_timeseries.csv"

OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


summary = pd.read_csv(SUMMARY_FILE)
ts = pd.read_csv(TIMESERIES_FILE)

# Ensure numeric
for c in ["cycle", "s1_mean", "s1_std"]:
    if c in ts.columns:
        ts[c] = pd.to_numeric(ts[c], errors="coerce")

#Plot 1: Average adoption time-series with std shading

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(
    ts["cycle"],
    ts["s1_mean"],
    label="Mean adoption (Firm 1)",
    linewidth=2,
)

ax.fill_between(
    ts["cycle"],
    ts["s1_mean"] - ts["s1_std"],
    ts["s1_mean"] + ts["s1_std"],
    alpha=0.25,
    label="±1 std (across MC runs)",
)

ax.set_xlabel("Cycle")
ax.set_ylabel("Market share")
ax.set_title("Average adoption trajectory (Monte Carlo mean)")

ax.legend(frameon=False)
ax.grid(alpha=0.3)

plt.tight_layout()

out_path = OUT_DIR / "average_adoption_timeseries.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"[OK] Time-series plot saved to: {out_path}")


print("\n=== Scenario summary ===")
print(
    summary[
        [
            "tau", "sigma", "switch_cycle",
            "H_mean", "H_std", "share_positive"
        ]
    ]
)
