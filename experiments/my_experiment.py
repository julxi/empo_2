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
from dataclasses import dataclass

import numpy as np

from grid_world import alphazero, empo_eval, env, solvino


def fair_box_population(size: int):
    """Each human cares about one inequality on the box's first coordinate."""
    human_1 = [(lambda x, i=i: float(x.box[0] <= i)) for i in range(size)]
    human_2 = [(lambda x, i=i: float(x.box[0] >= i)) for i in range(size)]
    return [human_1, human_2]


@dataclass
class Instance:
    name: str
    size: int
    max_steps: int
    start: env.GridWorldState


def default_instances() -> list[Instance]:
    return [
        Instance(
            name="biggy",
            size=5,
            max_steps=2 * 5,
            start=env.GridWorldState(agent=(0, 0), target=(0, 0), box=(1, 0), step=0),
        ),
    ]


def alphazero_solution(
    instance: Instance,
    params: solvino.EmpoParameter,
    cfg: alphazero.AlphaZeroConfig,
    seed: int = 0,
):
    e = env.GridWorldFuncEnv(
        instance.size, fair_box_population(instance.size), max_steps=instance.max_steps
    )
    az = alphazero.AlphaZeroSolver(e, params, cfg, seed=seed)
    t0 = time.perf_counter()
    az.fit(instance.start)
    elapsed = time.perf_counter() - t0
    return e, az, elapsed


def _terminal_box(env_, start, policy):
    cur = start
    while not env_.terminal(cur):
        cur = env_.transition(cur, policy(cur))
    return cur.box


def report(
    instance: Instance,
    params: solvino.EmpoParameter,
    cfg: alphazero.AlphaZeroConfig,
    eval_simulations: int = 256,
    seed: int = 0,
) -> dict:
    e_az, az, t_az = alphazero_solution(instance, params, cfg, seed=seed)

    greedy = az.greedy_policy(num_simulations=eval_simulations)
    az_action = greedy(instance.start)
    az_traj = empo_eval.evaluate_policy(e_az, params, greedy, instance.start)
    az_V = az_traj.V_r[0]
    az_terminal_box = _terminal_box(e_az, instance.start, greedy)

    print(f"\n=== {instance.name} (size={instance.size}, T={instance.max_steps}) ===")
    print(
        f"alphaz:   V_r={az_V:.4f}  best_action={az_action}  "
        f"terminal_box={az_terminal_box}  time={t_az:.3f}s "
        f"({cfg.iterations} iters x {cfg.episodes_per_iter} eps, "
        f"{cfg.mcts.num_simulations} sims/move; eval={eval_simulations} sims)"
    )
    print("training history (last 5):")
    for h in az.history[-5:]:
        print(
            f"  it={h['iteration']:>3}  policy_loss={h['policy_loss']:.3f}"
            f"  value_loss={h['value_loss']:.3f}  best_V={h['best_V_r']:.3f}"
            f"  mean_V={h['mean_V_r']:.3f}"
        )

    return {
        "instance": instance.name,
        "az_V": az_V,
        "az_action": az_action,
        "az_time": t_az,
        "az_terminal_box": az_terminal_box,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--episodes-per-iter", type=int, default=8)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--eval-sims", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder", choices=["flat", "rich"], default="flat")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument(
        "--temp-drop",
        type=int,
        default=8,
        help="step after which self-play becomes greedy",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="restrict to a single instance by name",
    )
    args = parser.parse_args()

    params = solvino.EmpoParameter(gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1)
    cfg = alphazero.AlphaZeroConfig(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        mcts=alphazero.MCTSConfig(num_simulations=args.sims),
        temperature_drop_step=args.temp_drop,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.hidden_layers,
    )

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


if __name__ == "__main__":
    main()
