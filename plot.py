import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ---------------- Configuration ----------------
BASE_DIR = Path.home() / "Desktop" / "Tipping-point" / "Results" / "New_Late_Policy_Withdraw_Scenario" / "aggregated"

SUMMARY_FILE = BASE_DIR / "scenario_level_summary.csv"
TIMESERIES_FILE = BASE_DIR / "scenario_level_timeseries.csv"

OUT_DIR = BASE_DIR / "plots_II"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("SUMMARY_FILE:", SUMMARY_FILE, "| exists:", SUMMARY_FILE.exists())
print("TIMESERIES_FILE:", TIMESERIES_FILE, "| exists:", TIMESERIES_FILE.exists())
print("OUT_DIR:", OUT_DIR, "| exists:", OUT_DIR.exists())

# ---------------- Load data ----------------
summary = pd.read_csv(SUMMARY_FILE)
ts = pd.read_csv(TIMESERIES_FILE)

print("\n=== scenario_level_summary.csv (head) ===")
print(summary.head(3))

print("\n=== scenario_level_timeseries.csv columns ===")
print(ts.columns.tolist())


# Ensure numeric
for c in ["cycle", "s1_mean", "s1_std", "s2_mean", "s2_std", "switch_cycle"]:
    if c in ts.columns:
        ts[c] = pd.to_numeric(ts[c], errors="coerce")

# ---------------- Sanity checks ----------------
if "s1_mean" in ts.columns and "s2_mean" in ts.columns:
    print("\nMean(s1_mean + s2_mean):", (ts["s1_mean"] + ts["s2_mean"]).mean())
    print("Last 5 sums:\n",
          ts[["cycle", "s1_mean", "s2_mean"]]
          .tail(5)
          .assign(sum=lambda d: d.s1_mean + d.s2_mean)
    )

print("\nTS last cycles:\n", ts[["cycle", "s1_mean", "s1_std"]].tail(8))

# Optional: compute post-switch mean ONLY if switch_cycle is available (non-NaN)
if "switch_cycle" in ts.columns and ts["switch_cycle"].notna().any():
    sc = int(ts["switch_cycle"].dropna().iloc[0])
    post_ts_mean = ts.loc[ts["cycle"] >= sc, "s1_mean"].mean()
    print(f"\nPost-switch mean from plotted series (cycles >= {sc}):", post_ts_mean)
else:
    print("\n[INFO] No switch_cycle in this time series. Skipping post-switch mean.")

# If present, show summary's s_post_mean (tail mean) as a single-number reference
if "s_post_mean" in summary.columns and len(summary) > 0:
    try:
        print("Tail mean from summary (s_post_mean):", float(summary.loc[0, "s_post_mean"]))
    except Exception as e:
        print("[INFO] Could not print s_post_mean:", e)

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(ts["cycle"], ts["s1_mean"], label="Mean market share (Firm 1)", linewidth=2)

ax.fill_between(
    ts["cycle"],
    ts["s1_mean"] - ts["s1_std"],
    ts["s1_mean"] + ts["s1_std"],
    alpha=0.25,
    label="±1 std (across MC runs)",
)

# --- Tail mean line (last post_window cycles) ---
post_window = 20  
tail_start = ts["cycle"].max() - (post_window - 1)   # e.g. 99-19 = 80
tail_mean = ts.loc[ts["cycle"] >= tail_start, "s1_mean"].mean()

ax.hlines(
    y=tail_mean,
    xmin=tail_start, xmax=ts["cycle"].max(),
    linestyles="--", linewidth=2,
    label=f"Tail mean (last {post_window} cycles): {tail_mean:.3f}"
)

# refresh legend so the new line appears
ax.legend(frameon=False)


ax.set_xlabel("Cycle")
ax.set_ylabel("Firm 1 market share")
ax.set_title("Firm 1 market share trajectory (Monte Carlo mean)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

# ---------------- Zoom inset (end of series) ----------------
zoom_last_n = 4  # adjust: 4, 8, 15...
x_max = ts["cycle"].max()
x_min = max(ts["cycle"].min(), x_max - zoom_last_n)

zoom_mask = (ts["cycle"] >= x_min) & (ts["cycle"] <= x_max)
y_low = (ts.loc[zoom_mask, "s1_mean"] - ts.loc[zoom_mask, "s1_std"]).min()
y_high = (ts.loc[zoom_mask, "s1_mean"] + ts.loc[zoom_mask, "s1_std"]).max()

pad = 0.01
y_min = max(0.0, y_low - pad)
y_max = min(1.0, y_high + pad)

axins = inset_axes(
    ax,
    width="27%", height="27%",            # inset size
    loc="center left",
    bbox_to_anchor=(1.02, 0.20, 1, 1),    # inset position (x, y, _, _)
    bbox_transform=ax.transAxes,
    borderpad=0.0
)

axins.plot(ts["cycle"], ts["s1_mean"], linewidth=2)
axins.fill_between(
    ts["cycle"],
    ts["s1_mean"] - ts["s1_std"],
    ts["s1_mean"] + ts["s1_std"],
    alpha=0.25
)

axins.set_xlim(x_min, x_max)
axins.set_ylim(y_min, y_max)
axins.grid(alpha=0.25)
axins.tick_params(labelsize=8)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# Leave space on the right so the inset isn't clipped
fig.subplots_adjust(right=0.72)

# ---------------- Save ----------------
out_path = OUT_DIR / "average_adoption_timeseries.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n[OK] Time-series plot saved to: {out_path}")

print("\n=== Scenario summary (selected columns) ===")
cols = [c for c in ["tau", "sigma", "switch_cycle", "H_mean", "H_std", "share_positive", "s_base_mean", "s_post_mean"] if c in summary.columns]
print(summary[cols])
