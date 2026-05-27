"""Demonstrates a small walled grid-world side-effect scenario.

Layout (top row = ``y = height - 1``):

    +---+---+---+---+
    |   | A | W | W |
    +---+---+---+---+
    |   | X |   |   |
    +---+---+---+---+
    | W |   |   |   |
    +---+---+---+---+
    | W | W |   | G |
    +---+---+---+---+

``A`` is the robot, ``X`` the box, ``W`` a wall, ``G`` a target position the
human cares about (encoded in the goal list, not stored in state).
"""

from __future__ import annotations

from rich import print

from grid_world import (
    GridWorldLayout,
    GridWorldState,
    MovingBoxEnv,
)

from _common import Instance, at_terminal, solve_and_rollout


SIZE = 4
MAX_STEPS = 10
TARGET = (3, 0)

# Coordinate convention: pos = (x, y) with RIGHT increasing x and UP increasing
# y. The diagram is drawn with y = size - 1 on top.
WALLS: frozenset[tuple[int, int]] = frozenset(
    {
        (2, 3),
        (3, 3),
        (0, 1),
        (1, 0),
        (0, 0),
    }
)

START = GridWorldState(robot=(1, 3), object=(1, 2), step=0)


def make_population():
    human_1 = [
        lambda o: float(at_terminal(o) and o.state.robot == TARGET),
        lambda o: float(at_terminal(o) and o.state.object == (0, 3)),
        lambda o: float(at_terminal(o) and o.state.object == (1, 3)),
        lambda o: float(at_terminal(o) and o.state.object in {(0, 2), (1, 2), (2, 2)}),
        lambda o: float(at_terminal(o) and o.state.object in {(1, 2), (2, 2)}),
        lambda o: float(at_terminal(o) and o.state.object in {(1, 2), (2, 2)}),
        lambda o: float(at_terminal(o) and o.state.object in {(1, 2), (2, 2), (3, 2)}),
        lambda o: float(at_terminal(o) and o.state.object == (1, 1)),
    ]
    return [human_1]


def main() -> None:
    instance = Instance(
        name="sideeffect",
        layout=GridWorldLayout(
            width=SIZE, height=SIZE, max_steps=MAX_STEPS, walls=WALLS
        ),
        population=make_population(),
        start=START,
    )
    env = MovingBoxEnv(instance.layout, instance.population)
    states, actions = solve_and_rollout(env, instance.start)

    for i, (state, action) in enumerate(zip(states, actions)):
        print(f"===step: {i}, action: {action}===")
        print(state)
    print(f"===final===")
    print(states[-1])


if __name__ == "__main__":
    main()
