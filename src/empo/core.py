"""General environment primitives and base classes."""

from dataclasses import dataclass
from typing import Any, Callable

from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv


@dataclass(frozen=True)
class EnvConfig:
    """Static environment description."""

    max_steps: int = 0


@dataclass(frozen=True)
class State:
    """Dynamic environment state."""

    step: int = 0


@dataclass(frozen=True)
class Obs[ConfigT: EnvConfig, StateT: State]:
    config: ConfigT
    state: StateT


type Goal[ObsT] = Callable[[ObsT], float]
type Human[ObsT] = list[Goal[ObsT]]
type Population[ObsT] = list[Human[ObsT]]
type GoalValues = list[list[float]]


def at_terminal(obs: Obs) -> bool:
    return obs.state.step >= obs.config.max_steps


class Env[ConfigT: EnvConfig, StateT: State](
    FuncEnv[StateT, Obs, int, GoalValues, bool, None, None]
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

    def goal_values(self, state: StateT) -> GoalValues:
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
