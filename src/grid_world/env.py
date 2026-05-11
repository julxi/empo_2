from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv
from numpy.typing import NDArray


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


type Obs = dict[str, NDArray[np.int64]]
type Info = dict[str, float]


@dataclass(frozen=True)
class GridWorldState:
    agent: tuple[int, int]
    target: tuple[int, int]
    box: tuple[int, int]


class GridWorldFuncEnv(FuncEnv[GridWorldState, Obs, int, float, bool, Any, Any]):
    size: int

    def __init__(self, size: int = 5) -> None:
        super().__init__()
        self.size = size
        coord = spaces.Box(0, size - 1, shape=(2,), dtype=np.int64)
        self.observation_space = spaces.Dict(
            {"agent": coord, "target": coord, "box": coord}
        )
        self.action_space = spaces.Discrete(4)

    def initial(self, rng: np.random.Generator, params: Any = None) -> GridWorldState:
        flat = rng.choice(self.size * self.size, size=3, replace=False)
        rows, cols = np.unravel_index(flat, (self.size, self.size))
        return GridWorldState(
            agent=(int(rows[0]), int(cols[0])),
            target=(int(rows[1]), int(cols[1])),
            box=(int(rows[2]), int(cols[2])),
        )

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def transition(
        self,
        state: GridWorldState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> GridWorldState:
        dx, dy = _DELTAS[Action(action)]
        new_agent = (state.agent[0] + dx, state.agent[1] + dy)
        if not self._in_bounds(new_agent):
            return state
        if new_agent == state.box:
            new_box = (state.box[0] + dx, state.box[1] + dy)
            if not self._in_bounds(new_box):
                return state
            return GridWorldState(agent=new_agent, target=state.target, box=new_box)
        return GridWorldState(agent=new_agent, target=state.target, box=state.box)

    def observation(
        self, state: GridWorldState, rng: Any = None, params: Any = None
    ) -> Obs:
        return {
            "agent": np.array(state.agent, dtype=np.int64),
            "target": np.array(state.target, dtype=np.int64),
            "box": np.array(state.box, dtype=np.int64),
        }

    def reward(
        self,
        state: GridWorldState,
        action: int,
        next_state: GridWorldState,
        rng: Any = None,
        params: Any = None,
    ) -> float:
        return 1.0 if next_state.agent == next_state.target else 0.0

    def terminal(
        self, state: GridWorldState, rng: Any = None, params: Any = None
    ) -> bool:
        return state.agent == state.target

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
