"""Runaway train (grid version): an object falls every step; a switch deflects it."""

from dataclasses import replace
from typing import Any

from ...core import DeterministicEnv, at_terminal
from .base import DELTAS, Action, GridConfig, GridObs, GridState, invalid_pos

# Grid envs share the grid flavours of config/state/obs; re-export under the
# uniform per-env names so callers can use ``runaway_train.EnvConfig`` etc.
EnvConfig = GridConfig
State = GridState
Obs = GridObs


class Env(DeterministicEnv[GridConfig, GridState]):
    BUTTON_SWITCH = 0
    num_actions = 4

    def transition(
        self,
        state: GridState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> GridState:

        next_step = state.step + 1

        dx, dy = DELTAS[Action(action)]

        new_robot = (state.robot[0] + dx, state.robot[1] + dy)
        if invalid_pos(new_robot, self.config):
            new_robot = state.robot

        obj = state.objects[0]
        new_object = (obj[0], obj[1] - 1)  # moving down
        if invalid_pos(new_object, self.config):
            new_object = obj

        new_buttons_state = list(state.button_states)

        if (
            new_buttons_state[self.BUTTON_SWITCH]
            and new_robot == self.config.buttons[self.BUTTON_SWITCH]
        ):
            new_buttons_state[self.BUTTON_SWITCH] = False
            new_object = (new_object[0] - 2, new_object[1])

        for button_idx in range(1, len(new_buttons_state)):
            if new_object == self.config.buttons[button_idx]:
                new_buttons_state[button_idx] = False

        return replace(
            state,
            step=next_step,
            robot=new_robot,
            objects=(new_object,),
            button_states=tuple(new_buttons_state),
        )


def survival_goal(button_idx: int):
    """1 iff at the terminal state the human's button is still pressed (alive)."""

    def g(obs: GridObs) -> float:
        return float(at_terminal(obs) and obs.state.button_states[button_idx])

    return g


def switch_unpressed_goal(switch_idx: int = 0):
    """1 iff at the terminal state the switch button has not been pressed."""

    def g(obs: GridObs) -> float:
        return float(at_terminal(obs) and obs.state.button_states[switch_idx])

    return g
