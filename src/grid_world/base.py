from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv


class Action(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


DELTAS: dict[Action, tuple[int, int]] = {
    Action.RIGHT: (1, 0),
    Action.UP: (0, 1),
    Action.LEFT: (-1, 0),
    Action.DOWN: (0, -1),
}


type Info = dict[str, float]

type pos = tuple[int, int]


@dataclass(frozen=True)
class GridWorldLayout:
    width: int = 5
    height: int = 5
    max_steps: int = 10
    walls: frozenset[pos] = frozenset()
    buttons: tuple[pos, ...] = tuple()


@dataclass(frozen=True)
class GridWorldState:
    """Only dynamic attributes"""

    robot: tuple[int, int] = (-1, -1)
    step: int = 0
    button_states: tuple[bool, ...] = tuple()
    object: tuple[int, int] = (-1, -1)


@dataclass(frozen=True)
class GridWorldObs:
    layout: GridWorldLayout
    state: GridWorldState


type Goal = Callable[[GridWorldObs], float]
type Human = list[Goal]
type Population = list[Human]
type Rewards = list[list[float]]


def out_of_bounds(pos: tuple[int, int], layout: GridWorldLayout) -> bool:
    return not (0 <= pos[0] < layout.width and 0 <= pos[1] < layout.height)


def invalid_pos(pos: tuple[int, int], layout: GridWorldLayout) -> bool:
    return out_of_bounds(pos, layout) or pos in layout.walls
