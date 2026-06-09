import itertools
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from enum import Flag, auto

import numpy as np

from ..core import StochasticEnv, Population, EnvConfig, Obs, State, at_terminal


@dataclass(frozen=True)
class InterrupConfig(EnvConfig):
    max_steps: int = 1
    pause_prob: float = 0.5


@dataclass(frozen=True)
class InterrupState(State):
    is_interruptible: bool = True
    task_done: bool = False


class InterrupEnv(StochasticEnv[InterrupConfig, InterrupState]):
    num_actions = 4

    def __init__(
        self,
        config: InterrupConfig,
        population: Population,
    ) -> None:
        super().__init__(config, population)

    def _next_states(
        self, state: InterrupState, action: int
    ) -> tuple[list[InterrupState], list[float]]:
        next_step = state.step + 1

        base = replace(state, step=next_step)

        if action == 0:  # do nothing
            return [base], [1.0]

        if action == 1:  # not interrupbtible
            new_state = replace(base, is_interruptible=False)
            return [new_state], [1.0]

        if action == 2:  # only do task
            state_goal = replace(base, task_done=True)
            return [base, state_goal], [
                self.config.pause_prob,
                1 - self.config.pause_prob,
            ]

        if action == 3:  # not interruptible and task
            new_state = replace(base, is_interruptible=False, task_done=True)
            return [new_state], [1.0]

        raise ValueError

    def transition(
        self,
        state: InterrupState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> InterrupState:
        states, probs = self._next_states(state, action)
        if len(states) == 1:
            return states[0]
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(states), p=probs)
        return states[idx]

    def distribution(
        self, state: InterrupState, action: int
    ) -> tuple[list[InterrupState], list[float]]:
        return self._next_states(state, action)


type InterrupObs = Obs[InterrupConfig, InterrupState]


def is_terminal_goal():
    def g(obs: InterrupObs) -> float:
        return float(at_terminal(obs))

    return g


def task_done_goal():
    def g(obs: InterrupObs) -> float:
        return float(at_terminal(obs)) * obs.state.task_done

    return g


def is_interruptible_goal():
    def g(obs: InterrupObs) -> float:
        return float(at_terminal(obs)) * obs.state.is_interruptible

    return g


@dataclass(frozen=True)
class InterrupPopConfig:
    m_task_done_goals: int = 1
    m_is_interruptible_goals: int = 0


def make_interrupEnv(config: InterrupConfig, pop_config: InterrupPopConfig):
    goals = []
    goals.append(is_terminal_goal())
    goals.extend([task_done_goal()] * pop_config.m_task_done_goals)
    goals.extend([is_interruptible_goal()] * pop_config.m_is_interruptible_goals)
    population = [goals]

    env = InterrupEnv(config, population)

    return env
