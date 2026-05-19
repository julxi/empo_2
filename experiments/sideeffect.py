"""Demonstrates a small walled grid-world (no goals).

Layout (top row = ``y = size - 1``):

    +---+---+---+---+
    |   | A | W | W |
    +---+---+---+---+
    |   | X |   |   |
    +---+---+---+---+
    | W |   |   |   |
    +---+---+---+---+
    |   | W |   | G |
    +---+---+---+---+

`A` is the agent, `X` the box, `W` a wall, `G` the target marker. The env
is built with an empty population (no goal functions), so no rewards are
emitted; the script just constructs the env, prints the layout, walks a
short scripted trajectory through walls and a box push, and shows the
channel-stacked tensor encoding so you can verify the wall plane.

Run with::

    .venv/bin/python -m experiments.walled_world
"""

from __future__ import annotations

import numpy as np

from grid_world import GridWorldFuncEnv, GridWorldState, render_grid
from grid_world.alphazero import CHANNEL_NAMES, encode_obs
from grid_world.env import Action
from grid_world.solvino import BackwardInductionSolver
from grid_world.empo_eval import rollout

SIZE = 4
MAX_STEPS = 10

# Coordinate convention: pos = (x, y) with RIGHT increasing x and UP increasing
# y. The diagram is drawn with y = size - 1 on top, so the visual top-left
# cell is (0, size - 1).
WALLS: frozenset[tuple[int, int]] = frozenset(
    {
        (2, 3),
        (3, 3),  # right end of the top row
        (0, 1),  # row y=1, left
        (1, 0),  # bottom row, column 1
        (0, 0),
    }
)
START = GridWorldState(
    agent=(1, 3),
    target=(3, 0),
    box=(1, 2),
    step=0,
)


def make_env() -> GridWorldFuncEnv:
    """Build the walled 4x4 environment with no goals."""

    human_1 = [
        lambda o: o.agent == o.target,
        lambda o: o.box == (0, 3),
        lambda o: o.box == (1, 3),
        lambda o: o.box in {(0, 2), (1, 2), (2, 2)},
        lambda o: o.box in {(1, 2), (2, 2)},
        lambda o: o.box in {(1, 2), (2, 2)},
        lambda o: o.box in {(1, 2), (2, 2), (3, 2)},
        lambda o: o.box in {(1, 1)},
    ]

    return GridWorldFuncEnv(
        size=SIZE,
        population=[human_1],
        max_steps=MAX_STEPS,
        walls=WALLS,
    )


def robot_policy_wrapper(robot_dict):
    def policy_function(state):
        return robot_dict[state]

    return policy_function


def main() -> None:
    env = make_env()
    start = START
    start_obs = env.observation(start)

    solver = BackwardInductionSolver(env)
    solver.solve(start)

    policy = robot_policy_wrapper(solver.robot_policy)
    trajectory = rollout(env, policy, start)

    for i, (state, action) in enumerate(zip(*trajectory)):
        print(f"===step: {i}, action: {action}===")
        obs = env.observation(state)
        print(render_grid(obs))


if __name__ == "__main__":
    main()
