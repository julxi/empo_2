from __future__ import annotations

import argparse

from rich import print

from empo.solvers.params import EmpoParameter
from empo.envs.grid.base import GridConfig, GridState
import empo.envs.grid.moving_box as moving_box

from _common import Instance, solve_and_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    params = EmpoParameter()

    instance = Instance(
        name="box_to_middle",
        config=GridConfig(
            width=args.size,
            height=args.size,
            max_steps=args.steps,
            walls=frozenset(),
            buttons=(),
        ),
        population=moving_box.fair_box_population(args.size),
        start=GridState(robot=(0, 0), objects=((1, 0),)),
    )

    env = moving_box.Env(instance.config, instance.population)
    states, _actions = solve_and_rollout(env, instance.start, params)
    print(states[-1])


if __name__ == "__main__":
    main()
