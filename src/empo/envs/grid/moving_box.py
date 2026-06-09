"""Robot pushes a box around a 2D grid (deterministic)."""

from dataclasses import replace
from typing import Any

from ...core import DeterministicEnv, Population, at_terminal
from .base import DELTAS, Action, GridConfig, GridObs, GridState, invalid_pos

# Grid envs share the grid flavours of config/state/obs; re-export under the
# uniform per-env names so callers can use ``moving_box.EnvConfig`` etc.
EnvConfig = GridConfig
State = GridState
Obs = GridObs


class Env(DeterministicEnv[GridConfig, GridState]):
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

        box = state.objects[0]
        new_robot = (state.robot[0] + dx, state.robot[1] + dy)
        if invalid_pos(new_robot, self.config):
            return replace(state, step=next_step)

        new_box = box
        if new_robot == box:
            new_box = (box[0] + dx, box[1] + dy)

        if invalid_pos(new_box, self.config):
            return replace(state, step=next_step)

        return replace(state, step=next_step, robot=new_robot, objects=(new_box,))


def above_below_goal(row: int, above: bool):
    """1 iff at the terminal state the box's x-coordinate is on the requested side of ``row``."""
    if above:

        def g(obs: GridObs) -> float:
            return float(at_terminal(obs) and obs.state.objects[0][0] <= row)

    else:

        def g(obs: GridObs) -> float:
            return float(at_terminal(obs) and obs.state.objects[0][0] >= row)

    return g


def fair_box_population(rows: int) -> Population:
    """Two humans with mirrored above/below preferences over rows 0..rows-1."""
    human_1 = [above_below_goal(i, True) for i in range(rows)]
    human_2 = [above_below_goal(i, False) for i in range(rows)]
    return [human_1, human_2]
