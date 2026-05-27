"""Robot pushes a box around a 2D grid (deterministic)."""

from dataclasses import replace
from typing import Any

from ..base import DELTAS, Action, GridWorldState, invalid_pos
from ..env_base import DeterministicGridWorldEnv


class MovingBoxEnv(DeterministicGridWorldEnv):
    def transition(
        self,
        state: GridWorldState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> GridWorldState:
        next_step = state.step + 1

        dx, dy = DELTAS[Action(action)]

        new_robot = (state.robot[0] + dx, state.robot[1] + dy)
        if invalid_pos(new_robot, self.layout):
            return replace(state, step=next_step)

        new_object = (state.object[0], state.object[1])
        if new_robot == state.object:
            new_object = (state.object[0] + dx, state.object[1] + dy)

        if invalid_pos(new_object, self.layout):
            return replace(state, step=next_step)

        return replace(state, step=next_step, robot=new_robot, object=new_object)
