"""Abstract base classes for gridworld environments.

Shared boilerplate (observation wrapping, terminal-only goal evaluation, default
``state_info`` / ``transition_info``) lives on :class:`GridWorldEnv`. Concrete
envs subclass either :class:`DeterministicGridWorldEnv` (a pure-function
``transition``) or :class:`StochasticGridWorldEnv` (``transition`` samples via
``rng`` and the law of the next state is exposed via ``distribution``). The
distinction is encoded as separate types so solvers that only work on
deterministic dynamics (backward induction, AlphaZero MCTS as currently
written) can declare that in their signatures.
"""

from typing import Any

from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv

from .base import (
    GridWorldLayout,
    GridWorldObs,
    GridWorldState,
    Population,
    Rewards,
)


class GridWorldEnv(
    FuncEnv[GridWorldState, GridWorldObs, int, Rewards, bool, None, None]
):
    def __init__(self, layout: GridWorldLayout, population: Population) -> None:
        super().__init__()
        self.layout = layout
        self.population = population
        self.observation_space = spaces.Dict()  # not used for now
        self.action_space = spaces.Discrete(4)

    @property
    def width(self) -> int:
        return self.layout.width

    @property
    def height(self) -> int:
        return self.layout.height

    def initial(self, rng: Any, params: Any = None) -> GridWorldState:
        raise NotImplementedError

    def transition(
        self,
        state: GridWorldState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> GridWorldState:
        raise NotImplementedError

    def observation(
        self, state: GridWorldState, rng: Any = None, params: Any = None
    ) -> GridWorldObs:
        return GridWorldObs(layout=self.layout, state=state)

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
        return state.step == self.layout.max_steps

    def state_info(self, state: GridWorldState, params: Any = None):
        return {}

    def transition_info(
        self,
        state: GridWorldState,
        action: int,
        next_state: GridWorldState,
        params: Any = None,
    ):
        return {}


class DeterministicGridWorldEnv(GridWorldEnv):
    """``transition`` is a pure function of ``(state, action)``.

    Theory behind these environments: see 2_deterministic.typ

    Solvers that rely on this (backward induction over the state graph, MCTS
    with a single child per action) should type their env argument with this
    class.
    """


class StochasticGridWorldEnv(GridWorldEnv):
    """``transition`` samples via ``rng``; ``distribution`` exposes the law.

    The full categorical distribution over next states is required by exact
    solvers (value iteration, stochastic backward induction).
    """

    def distribution(
        self, state: GridWorldState, action: int
    ) -> tuple[list[GridWorldState], list[float]]:
        raise NotImplementedError
