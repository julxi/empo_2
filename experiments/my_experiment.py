"""Train AlphaZero on a small Empo instance without comparing to the exact solver.

Run with::

    .venv/bin/python -m experiments.my_experiment
"""

from __future__ import annotations

import argparse

from grid_world import GridWorldState, empo_eval, solvino

from _common import (
    Instance,
    add_common_az_args,
    config_from_args,
    train_alphazero,
)


def default_instances() -> list[Instance]:
    size = 10
    return [
        Instance(
            name="biggy",
            size=size,
            max_steps=2 * size,
            start=GridWorldState(agent=(0, 0), target=(0, 0), box=(1, 0), step=0),
        ),
    ]


def report(
    instance: Instance,
    params: solvino.EmpoParameter,
    cfg,
    eval_simulations: int = 256,
    seed: int = 0,
) -> dict:
    e_az, az, t_az = train_alphazero(instance, params, cfg, seed=seed)

    greedy = az.greedy_policy(num_simulations=eval_simulations)
    az_action = greedy(instance.start)
    az_traj = empo_eval.evaluate_policy(e_az, params, greedy, instance.start)
    az_V = az_traj.V_r[0]
    az_terminal_box = az_traj.states[-1].box

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
    add_common_az_args(parser)
    args = parser.parse_args()

    params = solvino.EmpoParameter(gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1)
    cfg = config_from_args(args)

    instances = default_instances()
    if args.instance:
        instances = [x for x in instances if x.name == args.instance]
    for inst in instances:
        report(inst, params, cfg, eval_simulations=args.eval_sims, seed=args.seed)


if __name__ == "__main__":
    main()
