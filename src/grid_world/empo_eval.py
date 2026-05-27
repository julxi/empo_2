"""Trajectory-based evaluation of the simplified Empo equations.

Given a deterministic policy (or just a trajectory of actions), compute
V_h, X_h, U_r and V_r at every state along the trajectory. Mirrors
``BackwardInductionSolver``

Useful for two purposes:
- A reference implementation that any policy (not just the optimal one) can be
  scored with, including learnt AlphaZero policies.
- A simple, vectorised target generator for AlphaZero training.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .base import GridWorldState
from .empo import EmpoParameter
from .env_base import DeterministicGridWorldEnv

type Policy = Callable[[GridWorldState], int]


@dataclass(frozen=True)
class TrajectoryEvaluation:
    states: list[GridWorldState]
    actions: list[int]
    V_h: list[list[list[float]]]  # [t][h][g_h]
    X_h: list[list[float]]  # [t][h]
    U_r: list[float]  # [t]
    V_r: list[float]  # [t]


def wrap_dict(d: dict[GridWorldState, int]) -> Policy:
    def policy(state: GridWorldState):
        return d[state]

    return policy


def rollout(
    env: DeterministicGridWorldEnv,
    policy: Policy,
    start: GridWorldState,
) -> tuple[list[GridWorldState], list[int]]:
    states = [start]
    actions: list[int] = []
    current = start
    while not env.terminal(current):
        a = policy(current)
        actions.append(a)
        current = env.transition(current, a)
        states.append(current)
    return states, actions


def evaluate_trajectory(
    env: DeterministicGridWorldEnv,
    params: EmpoParameter,
    states: list[GridWorldState],
    actions: list[int],
) -> TrajectoryEvaluation:
    assert env.terminal(states[-1]), "trajectory must end in a terminal state"
    assert len(actions) == len(states) - 1

    N = len(states) - 1
    pop = env.population

    V_h: list[list[list[float]]] = [
        [[0.0] * len(goals) for goals in pop] for _ in range(N + 1)
    ]
    V_h[N] = env.goal_values(states[N])
    for t in range(N - 1, -1, -1):
        gv = env.goal_values(states[t])
        for h, goals in enumerate(pop):
            for j in range(len(goals)):
                g_here = gv[h][j]
                V_h[t][h][j] = (
                    g_here if g_here > 0 else params.gamma_h * V_h[t + 1][h][j]
                )

    X_h: list[list[float]] = [
        [sum(v**params.zeta for v in V_h[t][h]) for h in range(len(pop))]
        for t in range(N + 1)
    ]

    U_r: list[float] = [0.0] * (N + 1)
    V_r: list[float] = [0.0] * (N + 1)
    for t in range(N - 1, -1, -1):
        fair_power = sum(x ** (-params.xi) for x in X_h[t])
        U_r[t] = -(fair_power**params.eta)
        V_r[t] = U_r[t] + params.gamma_r * V_r[t + 1]

    return TrajectoryEvaluation(
        states=states, actions=actions, V_h=V_h, X_h=X_h, U_r=U_r, V_r=V_r
    )


def evaluate_policy(
    env: DeterministicGridWorldEnv,
    params: EmpoParameter,
    policy: Policy,
    start: GridWorldState,
) -> TrajectoryEvaluation:
    states, actions = rollout(env, policy, start)
    return evaluate_trajectory(env, params, states, actions)
