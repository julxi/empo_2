"""Train AlphaZero on a small Empo instance and compare with the exact solver.

Run with::

    .venv/bin/python -m experiments.compare_alphazero

This script
1. builds a deterministic grid-world instance,
2. solves it exactly with `BackwardInductionSolver`,
3. trains `AlphaZeroSolver` against the same instance,
4. reports the per-iteration history and the final comparison.
"""

from __future__ import annotations

import argparse
import time

from grid_world import GridWorldState, empo_eval, solvino

from experiments._common import (
    Instance,
    add_common_az_args,
    build_env,
    config_from_args,
    train_alphazero,
)


def default_instances() -> list[Instance]:
    return [
        Instance(
            name="fair-5x5-corner",
            size=5,
            max_steps=10,
            start=GridWorldState(agent=(0, 0), target=(0, 0), box=(1, 0), step=0),
        ),
        Instance(
            name="fair-5x5-mid",
            size=5,
            max_steps=10,
            start=GridWorldState(agent=(2, 2), target=(0, 0), box=(2, 3), step=0),
        ),
        Instance(
            name="fair-7x7-corner",
            size=7,
            max_steps=14,
            start=GridWorldState(agent=(0, 0), target=(0, 0), box=(1, 0), step=0),
        ),
    ]


def exact_solution(instance: Instance, params: solvino.EmpoParameter):
    e = build_env(instance)
    s = solvino.BackwardInductionSolver(e, params)
    t0 = time.perf_counter()
    s.solve(instance.start)
    elapsed = time.perf_counter() - t0
    return e, s, elapsed


def report(
    instance: Instance,
    params: solvino.EmpoParameter,
    cfg,
    eval_simulations: int = 256,
    seed: int = 0,
) -> dict:
    e_exact, exact, t_exact = exact_solution(instance, params)
    e_az, az, t_az = train_alphazero(instance, params, cfg, seed=seed)
    optimal_V = exact.V_r[instance.start]
    optimal_action = int(exact.robot_policy[instance.start])

    greedy = az.greedy_policy(num_simulations=eval_simulations)
    az_action = greedy(instance.start)
    az_traj = empo_eval.evaluate_policy(e_az, params, greedy, instance.start)
    az_V = az_traj.V_r[0]
    az_terminal_box = az_traj.states[-1].box
    exact_states, _ = empo_eval.rollout(
        e_exact, lambda s: exact.robot_policy[s], instance.start
    )
    exact_terminal_box = exact_states[-1].box

    print(f"\n=== {instance.name} (size={instance.size}, T={instance.max_steps}) ===")
    print(f"exact:    V_r={optimal_V:.4f}  best_action={optimal_action}  "
          f"terminal_box={exact_terminal_box}  time={t_exact:.3f}s")
    print(
        f"alphaz:   V_r={az_V:.4f}  best_action={az_action}  "
        f"terminal_box={az_terminal_box}  time={t_az:.3f}s "
        f"({cfg.iterations} iters x {cfg.episodes_per_iter} eps, "
        f"{cfg.mcts.num_simulations} sims/move; eval={eval_simulations} sims)"
    )
    print(f"V_r gap:  {az_V - optimal_V:+.4f}  (closer to 0 is better; 0 == optimal)")
    print("training history (last 5):")
    for h in az.history[-5:]:
        print(
            f"  it={h['iteration']:>3}  policy_loss={h['policy_loss']:.3f}"
            f"  value_loss={h['value_loss']:.3f}  best_V={h['best_V_r']:.3f}"
            f"  mean_V={h['mean_V_r']:.3f}"
        )

    return {
        "instance": instance.name,
        "exact_V": optimal_V,
        "exact_action": optimal_action,
        "exact_time": t_exact,
        "az_V": az_V,
        "az_action": az_action,
        "az_time": t_az,
        "az_terminal_box": az_terminal_box,
        "exact_terminal_box": exact_terminal_box,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_az_args(parser)
    args = parser.parse_args()

    params = solvino.EmpoParameter(
        gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1
    )
    cfg = config_from_args(args)

    instances = default_instances()
    if args.instance:
        instances = [x for x in instances if x.name == args.instance]
    rows = []
    for inst in instances:
        rows.append(
            report(
                inst,
                params,
                cfg,
                eval_simulations=args.eval_sims,
                seed=args.seed,
            )
        )

    print("\n=== summary ===")
    print(f"{'instance':<25} {'exact V':>10} {'az V':>10} {'gap':>10} "
          f"{'exact t':>10} {'az t':>10} {'V_r match':>10}")
    for r in rows:
        gap = r["az_V"] - r["exact_V"]
        v_match = "yes" if abs(gap) < 1e-4 else "no"
        print(
            f"{r['instance']:<25} {r['exact_V']:>10.4f} {r['az_V']:>10.4f} "
            f"{gap:>+10.4f} {r['exact_time']:>9.2f}s {r['az_time']:>9.2f}s "
            f"{v_match:>10}"
        )


if __name__ == "__main__":
    main()
