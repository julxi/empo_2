"""Trolley problem: an object falls every step; a switch deflects it."""

from dataclasses import replace
from typing import Any

from ..base import (
    DELTAS,
    Action,
    GridWorldLayout,
    GridWorldState,
    Population,
    invalid_pos,
)
from ..env_base import DeterministicGridWorldEnv


class RunawayTrainEnv(DeterministicGridWorldEnv):
    BUTTON_SWITCH = 0

    def __init__(self, layout: GridWorldLayout, population: Population) -> None:
        super().__init__(layout, population)
        self.pause_chance = 0.5

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
            new_robot = state.robot

        new_object = (state.object[0], state.object[1] - 1)  # moving down
        if invalid_pos(new_object, self.layout):
            new_object = state.object

        new_buttons_state = list(state.button_states)

        if (
            new_buttons_state[self.BUTTON_SWITCH]
            and new_robot == self.layout.buttons[self.BUTTON_SWITCH]
        ):
            new_buttons_state[self.BUTTON_SWITCH] = False
            new_object = (new_object[0] - 2, new_object[1])

        for button_idx in range(1, len(new_buttons_state)):
            if new_object == self.layout.buttons[button_idx]:
                new_buttons_state[button_idx] = False

        return replace(
            state,
            step=next_step,
            robot=new_robot,
            object=new_object,
            button_states=tuple(new_buttons_state),
        )
