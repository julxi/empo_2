"""Trolley problem: a single decision over named flags (no grid).

The abstract showcase for the general environment tier. There are no positions,
walls or movement — just one timestep in which the robot either flips the switch
or not, deciding which human survives. State is captured by named boolean flags
rather than grid coordinates.
"""

from dataclasses import dataclass, replace
from typing import Any

from ... import core
from ...core import Population, at_terminal


@dataclass(frozen=True)
class EnvConfig(core.EnvConfig):
    max_steps: int = 1


@dataclass(frozen=True)
class State(core.State):
    button: bool = True  # switch not yet thrown
    survivors: tuple[bool, bool] = (True, True)


class Env(core.DeterministicEnv[EnvConfig, State]):
    num_actions = 2

    def __init__(self, population: Population) -> None:
        super().__init__(EnvConfig(), population)

    def initial(self, rng: Any, params: Any = None) -> State:
        return State()

    def transition(
        self,
        state: State,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> State:

        if action == 0:
            return replace(state, step=1, survivors=(False, state.survivors[1]))
        return replace(
            state, step=1, button=False, survivors=(state.survivors[0], False)
        )


Obs = core.Obs[EnvConfig, State]


def survival_goal(h_idx: int):
    # h_idx ∈ {0,1}
    def g(obs: Obs) -> float:
        return float(at_terminal(obs) and obs.state.survivors[h_idx])

    return g


def agent_was_passive_goal():
    def g(obs: Obs) -> float:
        return float(at_terminal(obs) and obs.state.button)

    return g


def always_true_goal():
    def g(obs: Obs) -> float:
        return 1

    return g


@dataclass(frozen=True)
class PopConfig:
    n_killed_by_passive: int = 0  # humans killed if the robot stays passive (action 0)
    n_killed_by_pressing: int = 0  # humans killed if the robot presses (action 1)
    m_survival_goals: int = 1
    m_passivity_goals: int = 0


def make_population(pop_config: PopConfig) -> Population:
    population: Population = []
    groups = (
        (0, pop_config.n_killed_by_passive),
        (1, pop_config.n_killed_by_pressing),
    )
    for group, count in groups:
        human = (
            [always_true_goal()]
            + [survival_goal(group)] * pop_config.m_survival_goals
            + [agent_was_passive_goal()] * pop_config.m_passivity_goals
        )
        population.extend([human] * count)
    return population


def make_env(pop_config: PopConfig) -> Env:
    return Env(make_population(pop_config))
