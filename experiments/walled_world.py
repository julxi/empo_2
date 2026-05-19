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


SIZE = 4
MAX_STEPS = 16

# Coordinate convention: pos = (x, y) with RIGHT increasing x and UP increasing
# y. The diagram is drawn with y = size - 1 on top, so the visual top-left
# cell is (0, size - 1).
WALLS: frozenset[tuple[int, int]] = frozenset(
    {
        (2, 3), (3, 3),  # right end of the top row
        (0, 1),          # row y=1, left
        (1, 0),          # bottom row, column 1
    }
)
START = GridWorldState(agent=(1, 3), target=(3, 0), box=(1, 2), step=0)


def make_env() -> GridWorldFuncEnv:
    """Build the walled 4x4 environment with no goals."""
    return GridWorldFuncEnv(
        size=SIZE,
        population=[],
        max_steps=MAX_STEPS,
        walls=WALLS,
    )


def main() -> None:
    env = make_env()
    state = START

    print("Initial state:")
    print(render_grid(env.observation(state)))

    # Walk a short scripted trajectory that exercises walls and box pushing.
    scripted: list[Action] = [
        Action.LEFT,   # agent: (1,3) -> (0,3)
        Action.DOWN,   # (0,3) -> (0,2)
        Action.DOWN,   # (0,2) -> blocked by wall at (0,1)
        Action.RIGHT,  # (0,2) -> (1,2): push box (1,2) -> (2,2)
        Action.DOWN,   # agent (1,2) -> (1,1)
    ]
    for action in scripted:
        state = env.transition(state, int(action))
        print(f"\nafter {action.name}  step={state.step}:")
        print(render_grid(env.observation(state)))

    # Show the channel-stacked encoding of the current state.
    enc = encode_obs(env.observation(state))
    print(f"\nencoded tensor shape: {enc.shape}  dtype: {enc.dtype}")
    for c, name in enumerate(CHANNEL_NAMES):
        plane = enc[c]
        if name == "step":
            print(f"channel {c} ({name}): constant = {plane[0, 0]:.3f}")
        else:
            # Print the plane with y = size - 1 on top to match the diagram.
            flipped = np.flipud(plane.T)
            print(f"channel {c} ({name}):")
            print(flipped.astype(int))


if __name__ == "__main__":
    main()
