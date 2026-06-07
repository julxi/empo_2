"""General environment primitives and base classes.

These types know nothing about grids. An environment is described by a
:class:`State` (whose only guaranteed field is the step clock), an
:class:`EnvConfig` (carrying ``max_steps``), and observations are the pair
:class:`Obs` of ``(config, state)`` handed to each goal. The :class:`Env`
base classes define the functional environment API on top of these. Grid-specific
primitives (positions, walls, movement) live in :mod:`empo.grid`.
"""

from dataclasses import dataclass
from typing import Any, Callable

from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv


@dataclass(frozen=True)
class State:
    """Dynamic environment state. Subclasses add domain-specific fields.

    Frozen (and therefore hashable) so solvers can use states as dict keys.
    Every field must have a default so subclasses can add their own defaulted
    fields without dataclass field-ordering errors.
    """

    step: int = 0


@dataclass(frozen=True)
class EnvConfig:
    """Static environment description. Subclasses add domain-specific fields."""

    max_steps: int = 0


@dataclass(frozen=True)
class Obs[ConfigT: EnvConfig, StateT: State]:
    config: ConfigT
    state: StateT


type Goal[ObsT] = Callable[[ObsT], float]
type Human[ObsT] = list[Goal[ObsT]]
type Population[ObsT] = list[Human[ObsT]]
type Rewards = list[list[float]]


def at_terminal(obs: Obs) -> bool:
    return obs.state.step >= obs.config.max_steps


class Env[ConfigT: EnvConfig, StateT: State](
    FuncEnv[StateT, Obs, int, Rewards, bool, None, None]
):
    num_actions: int  # set by each concrete env

    def __init__(
        self, config: ConfigT, population: Population[Obs[ConfigT, StateT]]
    ) -> None:
        super().__init__()
        self.config: ConfigT = config
        self.population = population
        self.action_space = spaces.Discrete(self.num_actions)

    def initial(self, rng: Any, params: Any = None) -> StateT:
        raise NotImplementedError

    def transition(
        self,
        state: StateT,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> StateT:
        raise NotImplementedError

    def observation(
        self, state: StateT, rng: Any = None, params: Any = None
    ) -> Obs[ConfigT, StateT]:
        return Obs(config=self.config, state=state)

    def goal_values(self, state: StateT) -> Rewards:
        obs = self.observation(state)
        return [[goal(obs) for goal in human_goals] for human_goals in self.population]

    def terminal(self, state: StateT, rng: Any = None, params: Any = None) -> bool:
        return at_terminal(self.observation(state))

    def state_info(self, state: StateT, params: Any = None):
        return {}

    def transition_info(
        self,
        state: StateT,
        action: int,
        next_state: StateT,
        params: Any = None,
    ):
        return {}


class DeterministicEnv[ConfigT: EnvConfig, StateT: State](Env[ConfigT, StateT]):
    """``transition`` is a pure function of ``(state, action)``."""


class StochasticEnv[ConfigT: EnvConfig, StateT: State](Env[ConfigT, StateT]):
    """``transition`` samples via ``rng``; ``distribution`` exposes the dynamics."""

    def distribution(
        self, state: StateT, action: int
    ) -> tuple[list[StateT], list[float]]:
        raise NotImplementedError
