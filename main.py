import argparse
import os
import csv
from datetime import datetime

from economy import Economy


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_csv(path: str, fieldnames: list[str], row: dict) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def run_simulation(
    econ,
    n_cycles,
    tau,
    sigma,
    switch_cycle=None,
    tau_after=None,
    sigma_after=None,
    *,
    verbose: bool = True,
    run_id: str | None = None,
    timeseries_csv: str | None = None,
):
    """
    Runs the simulation for n_cycles.
    If switch_cycle is provided, policy switches to (tau_after, sigma_after) starting at that cycle.
    Optionally appends per-cycle results to a CSV (timeseries_csv) when run_id is provided.
    """
    if verbose:
        print("\n--- Simulation start ---")
        print(f"Households: {len(econ.households)} | Cycles: {n_cycles}")
        print(f"Policy (start): tau={tau} | sigma={sigma}")

        if switch_cycle is not None:
            print(f"Policy switch at cycle {switch_cycle}: tau={tau_after} | sigma={sigma_after}")

        print("\nCycle | Share firm1 | Share firm2 | PeerSignal")
        print("----- | ---------- | ---------- | ----------")

    for t in range(n_cycles):
        # policy schedule
        if switch_cycle is not None and t >= switch_cycle:
            tau_t = tau_after
            sigma_t = sigma_after
        else:
            tau_t = tau
            sigma_t = sigma

        # run one cycle (IMPORTANT: your Economy.run_cycle must use tau_t, sigma_t)
        shares = econ.run_cycle(tau=tau_t, sigma=sigma_t)

        peer = econ.peer_signal if hasattr(econ, "peer_signal") else None
        s1 = shares.get(1, 0.0)
        s2 = shares.get(2, 0.0)

        # --- optional: append cycle record to timeseries.csv ---
        if timeseries_csv is not None and run_id is not None:
            _append_csv(
                timeseries_csv,
                fieldnames=[
                    "run_id", "cycle",
                    "tau_t", "sigma_t",
                    "switch_cycle", "tau", "sigma", "tau_after", "sigma_after",
                    "s1", "s2", "peer_signal",
                ],
                row={
                    "run_id": run_id,
                    "cycle": t,
                    "tau_t": tau_t,
                    "sigma_t": sigma_t,
                    "switch_cycle": switch_cycle if switch_cycle is not None else "",
                    "tau": tau,
                    "sigma": sigma,
                    "tau_after": tau_after if tau_after is not None else "",
                    "sigma_after": sigma_after if sigma_after is not None else "",
                    "s1": s1,
                    "s2": s2,
                    "peer_signal": peer if peer is not None else "",
                },
            )

        if verbose:
            print(f"{t:>5} | {s1:>10.4f} | {s2:>10.4f} | {peer if peer is not None else '-':>10}")

    if verbose:
        print("\n--- Simulation end ---")

        # tipping summary
        if getattr(econ, "tipping_point", None) is None:
            print("Tipping point: NOT detected.")
        else:
            print(f"Tipping point: detected at cycle {econ.tipping_point}")

    return econ


def main():
    parser = argparse.ArgumentParser(description="Run sustainability tipping ABM (terminal).")

    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--households", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)

    # baseline policy
    parser.add_argument("--tau", type=float, default=0.10, help="Carbon tax on brown component")
    parser.add_argument("--sigma", type=float, default=0.05, help="Subsidy on green component")

    # counterfactual irreversibility test (policy switch)
    parser.add_argument(
        "--switch_cycle",
        type=int,
        default=None,
        help="Cycle at which to switch policy (counterfactual test).",
    )
    parser.add_argument(
        "--tau_after",
        type=float,
        default=0.0,
        help="Tau after switch_cycle (e.g., 0.0 to remove tax).",
    )
    parser.add_argument(
        "--sigma_after",
        type=float,
        default=0.0,
        help="Sigma after switch_cycle (e.g., 0.0 to remove subsidy).",
    )

    # run multiple switch cycles in one command (e.g. 30 and 80)
    parser.add_argument(
        "--switch_cycles",
        type=int,
        nargs="*",
        default=None,
        help="Run multiple switch cycles in one command, e.g. --switch_cycles 30 80",
    )

    # saving and hysteresis settings
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Append results to results/timeseries.csv and results/runs_summary.csv",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Folder for CSV outputs (default: results)",
    )
    parser.add_argument(
        "--post_window",
        type=int,
        default=20,
        help="Number of final cycles used for post-policy mean (default: 20)",
    )

    args = parser.parse_args()

    # Decide which switch cycles to run
    if args.switch_cycles is not None:
        switch_list = args.switch_cycles
    else:
        switch_list = [args.switch_cycle]  # may be [None]

    # Prepare CSV paths
    if args.save_csv:
        _ensure_dir(args.results_dir)
        timeseries_csv = os.path.join(args.results_dir, "timeseries.csv")
        summary_csv = os.path.join(args.results_dir, "runs_summary.csv")
    else:
        timeseries_csv = None
        summary_csv = None

    # --- baseline run for hysteresis (same seed, tau=sigma=0, no switch) ---
    econ_base = Economy(n_households=args.households, seed=args.seed)
    econ_base = run_simulation(
        econ=econ_base,
        n_cycles=args.cycles,
        tau=0.0,
        sigma=0.0,
        switch_cycle=None,
        tau_after=None,
        sigma_after=None,
        verbose=False,      # keep baseline quiet
        run_id=None,        # baseline not written to CSV by default
        timeseries_csv=None,
    )

    # baseline mean over last M cycles
    M = args.post_window
    base_tail = econ_base.market_shares[-M:]
    s_base_mean = sum(d.get(1, 0.0) for d in base_tail) / max(len(base_tail), 1)

    # --- policy runs (one per switch cycle) ---
    for sc in switch_list:
        econ = Economy(n_households=args.households, seed=args.seed)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = (
            f"{stamp}_seed{args.seed}_N{args.households}_T{args.cycles}"
            f"_tau{args.tau}_sig{args.sigma}_sc{sc}_tauA{args.tau_after}_sigA{args.sigma_after}"
        )

        econ = run_simulation(
            econ=econ,
            n_cycles=args.cycles,
            tau=args.tau,
            sigma=args.sigma,
            switch_cycle=sc,
            tau_after=args.tau_after,
            sigma_after=args.sigma_after,
            verbose=True,
            run_id=run_id,
            timeseries_csv=timeseries_csv,
        )

        # post mean over last M cycles
        post_tail = econ.market_shares[-M:]
        s_post_mean = sum(d.get(1, 0.0) for d in post_tail) / max(len(post_tail), 1)

        # hysteresis residual uplift
        H_delta = s_post_mean - s_base_mean

        # append one-row summary
        if summary_csv is not None:
            _append_csv(
                summary_csv,
                fieldnames=[
                    "run_id", "seed", "households", "cycles",
                    "tau", "sigma", "switch_cycle", "tau_after", "sigma_after",
                    "post_window",
                    "tipping_point",
                    "s_base_mean", "s_post_mean", "H_delta",
                ],
                row={
                    "run_id": run_id,
                    "seed": args.seed if args.seed is not None else "",
                    "households": args.households,
                    "cycles": args.cycles,
                    "tau": args.tau,
                    "sigma": args.sigma,
                    "switch_cycle": sc if sc is not None else "",
                    "tau_after": args.tau_after,
                    "sigma_after": args.sigma_after,
                    "post_window": M,
                    "tipping_point": econ.tipping_point if getattr(econ, "tipping_point", None) is not None else "",
                    "s_base_mean": s_base_mean,
                    "s_post_mean": s_post_mean,
                    "H_delta": H_delta,
                },
            )

        # quick console check
        print(
            f"\nHysteresis (post-window={M}): "
            f"s_base_mean={s_base_mean:.4f} | s_post_mean={s_post_mean:.4f} | H_delta={H_delta:.4f}\n"
        )


if __name__ == "__main__":
    main()
