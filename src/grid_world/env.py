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


_DELTAS: dict[Action, tuple[int, int]] = {
    Action.RIGHT: (1, 0),
    Action.UP: (0, 1),
    Action.LEFT: (-1, 0),
    Action.DOWN: (0, -1),
}


type Info = dict[str, float]


@dataclass(frozen=True)
class GridWorldState:
    "Minimal Markov state. Only dynamic objects"

    agent: tuple[int, int]
    target: tuple[int, int]
    box: tuple[int, int]
    step: int


@dataclass(frozen=True)
class GridWorldObs:
    """Complete state description: Dynamic and static objects"""

    agent: tuple[int, int]
    target: tuple[int, int]
    box: tuple[int, int]
    step: int
    walls: frozenset[tuple[int, int]]
    size: int
    max_steps: int


type Goal = Callable[[GridWorldObs], float]
type Human = list[Goal]
type Population = list[Human]
type Rewards = list[list[float]]


def _in_bounds(pos: tuple[int, int], size: int) -> bool:
    return 0 <= pos[0] < size and 0 <= pos[1] < size


def render_grid(obs: GridWorldObs) -> str:
    """ASCII render of an observation with ``y = size - 1`` drawn on top.

    Glyphs: ``A`` agent, ``X`` box, ``G`` target, ``W`` wall.
    """
    size = obs.size
    rows: list[str] = []
    bar = "+" + "+".join(["---"] * size) + "+"
    for y in range(size - 1, -1, -1):
        rows.append(bar)
        cells: list[str] = []
        for x in range(size):
            pos = (x, y)
            if pos == obs.agent:
                cells.append(" A ")
            elif pos == obs.box:
                cells.append(" X ")
            elif pos == obs.target:
                cells.append(" G ")
            elif pos in obs.walls:
                cells.append(" W ")
            else:
                cells.append("   ")
        rows.append("|" + "|".join(cells) + "|")
    rows.append(bar)
    return "\n".join(rows)


class GridWorldFuncEnv(
    FuncEnv[GridWorldState, GridWorldObs, int, Rewards, bool, None, None]
):
    size: int

    def __init__(
        self,
        size: int,
        population: Population,
        max_steps: int,
        walls: "frozenset[tuple[int, int]] | None" = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.population = population
        self.max_steps = max_steps
        self.walls: frozenset[tuple[int, int]] = frozenset(walls or ())
        coord = spaces.Box(0, size - 1, shape=(2,), dtype=np.int64)
        # observation_space is kept for gym compatibility but nothing in this
        # codebase actually consumes it.
        self.observation_space = spaces.Dict(
            {"agent": coord, "target": coord, "box": coord}
        )
        self.action_space = spaces.Discrete(4)

    def initial(self, rng: np.random.Generator, params: Any = None) -> GridWorldState:
        free = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.walls
        ]
        if len(free) < 3:
            raise ValueError("not enough free cells for agent, target and box")
        idx = rng.choice(len(free), size=3, replace=False)
        return GridWorldState(
            agent=free[int(idx[0])],
            target=free[int(idx[1])],
            box=free[int(idx[2])],
            step=0,
        )

    def transition(
        self,
        state: GridWorldState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> GridWorldState:
        dx, dy = _DELTAS[Action(action)]
        next_step = state.step + 1
        walls = self.walls
        new_agent = (state.agent[0] + dx, state.agent[1] + dy)
        if not _in_bounds(new_agent, self.size) or new_agent in walls:
            return GridWorldState(
                agent=state.agent,
                target=state.target,
                box=state.box,
                step=next_step,
            )
        if new_agent == state.box:
            new_box = (state.box[0] + dx, state.box[1] + dy)
            if not _in_bounds(new_box, self.size) or new_box in walls:
                return GridWorldState(
                    agent=state.agent,
                    target=state.target,
                    box=state.box,
                    step=next_step,
                )
            return GridWorldState(
                agent=new_agent,
                target=state.target,
                box=new_box,
                step=next_step,
            )
        return GridWorldState(
            agent=new_agent,
            target=state.target,
            box=state.box,
            step=next_step,
        )

    def observation(
        self, state: GridWorldState, rng: Any = None, params: Any = None
    ) -> GridWorldObs:
        return GridWorldObs(
            agent=state.agent,
            target=state.target,
            box=state.box,
            step=state.step,
            walls=self.walls,
            size=self.size,
            max_steps=self.max_steps,
        )

    def reward(
        self,
        state: GridWorldState,
        action: int,
        next_state: GridWorldState,
        rng: Any = None,
        params: Any = None,
    ) -> Rewards:
        if not self.terminal(next_state, rng, params):
            return [[0.0] * len(human_goals) for human_goals in self.population]
        obs = self.observation(next_state)
        return [[goal(obs) for goal in human_goals] for human_goals in self.population]

    def terminal(
        self, state: GridWorldState, rng: Any = None, params: Any = None
    ) -> bool:
        return state.step == self.max_steps

    def state_info(self, state: GridWorldState, params: Any = None) -> Info:
        dx = abs(state.agent[0] - state.target[0])
        dy = abs(state.agent[1] - state.target[1])
        return {"distance": float(dx + dy)}

    def transition_info(
        self,
        state: GridWorldState,
        action: int,
        next_state: GridWorldState,
        params: Any = None,
    ) -> Info:
        return self.state_info(next_state)
