import argparse
import os
import csv
from datetime import datetime

from economy import Economy


# =====================================================
# Utilities
# =====================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_csv(path: str, fieldnames: list[str], row: dict) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# =====================================================
# Simulation function
# =====================================================

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

        # run one cycle
        shares = econ.run_cycle(tau=tau_t, sigma=sigma_t)

        # collect firm-side variables
        firm_data = {}
        for firm in econ.firms:
            k = firm.firm_id

            firm_data[f"price_{k}"] = firm.price
            firm_data[f"markup_{k}"] = firm.markup
            firm_data[f"unit_cost_{k}"] = firm.unit_cost(tau_t, sigma_t)
            firm_data[f"quantity_{k}"] = (
                firm.quantity_history[-1] if firm.quantity_history else None
            )
            firm_data[f"profit_{k}"] = (
                firm.profit_history[-1] if firm.profit_history else None
            )
            firm_data[f"elasticity_{k}"] = (
                firm.last_elasticity
                if hasattr(firm, "last_elasticity")
                else firm.estimate_elasticity_last_two()
            )

        peer = econ.peer_signal if hasattr(econ, "peer_signal") else None
        s1 = shares.get(1, 0.0)
        s2 = shares.get(2, 0.0)

        # append cycle record
        if timeseries_csv is not None and run_id is not None:
            _append_csv(
                timeseries_csv,
                fieldnames=[
                    "run_id", "cycle",
                    "tau_t", "sigma_t",
                    "switch_cycle", "tau", "sigma", "tau_after", "sigma_after",
                    "s1", "s2", "peer_signal",
                    "price_1", "markup_1", "unit_cost_1", "quantity_1", "profit_1", "elasticity_1",
                    "price_2", "markup_2", "unit_cost_2", "quantity_2", "profit_2", "elasticity_2",
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
                    **firm_data,
                },
            )

        if verbose:
            print(f"{t:>5} | {s1:>10.4f} | {s2:>10.4f} | {peer if peer is not None else '-':>10}")

    if verbose:
        print("\n--- Simulation end ---")
        if getattr(econ, "tipping_point", None) is None:
            print("Tipping point: NOT detected.")
        else:
            print(f"Tipping point: detected at cycle {econ.tipping_point}")

    return econ


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Run sustainability tipping ABM (terminal).")

    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--households", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)

    # policy
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--tau_after", type=float, default=0.0)
    parser.add_argument("--sigma_after", type=float, default=0.0)

    parser.add_argument("--switch_cycles", nargs="*", type=int, default=None)

    # output
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--post_window", type=int, default=20)

    # ------------------------------
    # Monte Carlo (NEW)
    # ------------------------------
    parser.add_argument("--mc", action="store_true")
    parser.add_argument("--mc_runs", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=1)

    args = parser.parse_args()

    # switch cycles
    switch_list = args.switch_cycles if args.switch_cycles is not None else [None]

    # CSV paths
    if args.save_csv:
        _ensure_dir(args.results_dir)
        timeseries_csv = os.path.join(args.results_dir, "timeseries.csv")
        summary_csv = os.path.join(args.results_dir, "runs_summary.csv")
    else:
        timeseries_csv = None
        summary_csv = None

    # seeds
    if args.mc:
        seeds = list(range(args.seed_start, args.seed_start + args.mc_runs))
        print(f"Running Monte Carlo with {len(seeds)} seeds")
    else:
        seeds = [args.seed]

    # =================================================
    # OUTER LOOP OVER SEEDS (Monte Carlo)
    # =================================================
    for run_idx, seed in enumerate(seeds, start=1):

        if args.mc:
            print(f"[MC {run_idx}/{len(seeds)}] seed={seed}")

        # --- baseline (tau=sigma=0) ---
        econ_base = Economy(n_households=args.households, seed=seed)
        econ_base = run_simulation(
            econ_base,
            n_cycles=args.cycles,
            tau=0.0,
            sigma=0.0,
            switch_cycle=None,
            tau_after=None,
            sigma_after=None,
            verbose=False,
            run_id=None,
            timeseries_csv=None,
        )

        M = args.post_window
        base_tail = econ_base.market_shares[-M:]
        s_base_mean = sum(d.get(1, 0.0) for d in base_tail) / max(len(base_tail), 1)

        # --- policy runs ---
        for sc in switch_list:
            econ = Economy(n_households=args.households, seed=seed)

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = (
                f"{stamp}_seed{seed}_N{args.households}_T{args.cycles}"
                f"_tau{args.tau}_sig{args.sigma}_sc{sc}_tauA{args.tau_after}_sigA{args.sigma_after}"
            )

            econ = run_simulation(
                econ,
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

            post_tail = econ.market_shares[-M:]
            s_post_mean = sum(d.get(1, 0.0) for d in post_tail) / max(len(post_tail), 1)
            H_delta = s_post_mean - s_base_mean

            if summary_csv is not None:
                _append_csv(
                    summary_csv,
                    fieldnames=[
                        "run_id", "seed", "households", "cycles",
                        "tau", "sigma", "switch_cycle", "tau_after", "sigma_after",
                        "post_window", "tipping_point",
                        "s_base_mean", "s_post_mean", "H_delta",
                    ],
                    row={
                        "run_id": run_id,
                        "seed": seed,
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

            print(
                f"\nHysteresis (post-window={M}): "
                f"s_base_mean={s_base_mean:.4f} | "
                f"s_post_mean={s_post_mean:.4f} | "
                f"H_delta={H_delta:.4f}\n"
            )


if __name__ == "__main__":
    main()
