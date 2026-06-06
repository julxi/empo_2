"""Trolley problem: a single decision over named flags (no grid).

The abstract showcase for the general environment tier. There are no positions,
walls or movement — just one timestep in which the robot either flips the switch
or not, deciding which human survives. State is captured by named boolean flags
rather than grid coordinates.
"""

from dataclasses import dataclass, replace
from typing import Any

from ..core import DeterministicEnv, EnvConfig, Obs, Population, State, at_terminal


@dataclass(frozen=True)
class TrolleyState(State):
    button: bool = True  # switch not yet thrown
    survivors: tuple[bool, bool] = (True, True)


type TrolleyObs = Obs[EnvConfig, TrolleyState]


class TrolleyEnv(DeterministicEnv[EnvConfig, TrolleyState]):
    num_actions = 2

    def __init__(self, population: Population) -> None:
        super().__init__(EnvConfig(max_steps=1), population)

    def initial(self, rng: Any, params: Any = None) -> TrolleyState:
        return TrolleyState()

    def transition(
        self,
        state: TrolleyState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> TrolleyState:

        if action == 0:
            return replace(state, step=1, survivors=(False, state.survivors[1]))
        return replace(
            state, step=1, button=False, survivors=(state.survivors[0], False)
        )


def survival_goal(h_idx: int):
    # h_idx ∈ {0,1}
    def g(obs: TrolleyObs) -> float:
        return float(at_terminal(obs) and obs.state.survivors[h_idx])

    return g


def agent_was_passive_goal():
    def g(obs: TrolleyObs) -> float:
        return float(at_terminal(obs) and obs.state.button)

    return g


def always_true_goal():
    def g(obs: TrolleyObs) -> float:
        return 1

    return g
