from __future__ import annotations

import argparse

from rich import print

from grid_world import (
    EmpoParameter,
    GridWorldLayout,
    GridWorldState,
    MovingBoxEnv,
)

from _common import Instance, fair_box_population, solve_and_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    params = EmpoParameter()

    instance = Instance(
        name="box_to_middle",
        layout=GridWorldLayout(
            width=args.size,
            height=args.size,
            max_steps=args.steps,
            walls=frozenset(),
            buttons=(),
        ),
        population=fair_box_population(args.size),
        start=GridWorldState(robot=(0, 0), object=(1, 0)),
    )

    env = MovingBoxEnv(instance.layout, instance.population)
    states, _actions = solve_and_rollout(env, instance.start, params)
    print(states[-1])


if __name__ == "__main__":
    main()
