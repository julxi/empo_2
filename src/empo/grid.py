"""Grid-specific environment primitives.

Everything that assumes a 2D grid lives here: the 4-directional action enum,
movement deltas, bounds/wall checks, the grid flavours of
:class:`~empo.core.EnvConfig` / :class:`~empo.core.State`, and the
spatial (channel-stacked) observation encoder used by the AlphaZero solver.

Abstract (non-grid) environments such as the trolley problem import none of
this; they build on :mod:`empo.core` directly.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .core import EnvConfig, Obs, State


class Action(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


type pos = tuple[int, int]


DELTAS: dict[Action, pos] = {
    Action.RIGHT: (1, 0),
    Action.UP: (0, 1),
    Action.LEFT: (-1, 0),
    Action.DOWN: (0, -1),
}


@dataclass(frozen=True)
class GridConfig(EnvConfig):
    width: int = 0
    height: int = 0
    walls: frozenset[pos] = frozenset()
    buttons: tuple[pos, ...] = tuple()


@dataclass(frozen=True)
class GridState(State):
    """Shared dynamic state for grid environments.

    ``objects`` holds movable entities (boxes, falling objects); ``button_states``
    tracks per-button booleans (indexed in parallel with ``GridConfig.buttons``).
    """

    robot: pos = (-1, -1)
    objects: tuple[pos, ...] = ()
    button_states: tuple[bool, ...] = ()


type GridObs = Obs[GridConfig, GridState]


def out_of_bounds(p: pos, config: GridConfig) -> bool:
    return not (0 <= p[0] < config.width and 0 <= p[1] < config.height)


def invalid_pos(p: pos, config: GridConfig) -> bool:
    return out_of_bounds(p, config) or p in config.walls


CHANNEL_NAMES: tuple[str, ...] = ("robot", "object", "walls", "step")


def encode_obs(obs: Obs) -> np.ndarray:
    """Channel-stacked spatial encoding with shape ``(C, width, height)``.

    Channels (see :data:`CHANNEL_NAMES`):
      0. robot indicator
      1. object indicator (first entry of ``objects``)
      2. walls indicator (1 at every wall cell)
      3. normalised step, broadcast across the grid (``step / max_steps``)

    Grid-only: ``obs.config`` must be a :class:`GridConfig` and ``obs.state`` a
    :class:`GridState`.
    """
    state = obs.state
    config = obs.config
    assert isinstance(state, GridState)
    assert isinstance(config, GridConfig)
    channels = np.zeros(
        (len(CHANNEL_NAMES), config.width, config.height), dtype=np.float32
    )
    channels[0, state.robot[0], state.robot[1]] = 1.0
    if state.objects:
        ox, oy = state.objects[0]
        channels[1, ox, oy] = 1.0
    for wx, wy in config.walls:
        channels[2, wx, wy] = 1.0
    channels[3, :, :] = state.step / config.max_steps
    return channels
