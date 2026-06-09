"""Analytic integration tests for the backward-induction solvers.

Each solver is run on an environment with a known closed-form solution and
checked against it:

- ``BackwardInductionSolver`` on the trolley problem — terminal ``U_r`` matches
  the analytic ``eq-trolley-Uu`` / ``eq-trolley-Up`` from ``math/2_deterministic.typ``.
- ``StochasticBackwardInductionSolver`` on the interruptibility env — ``Q_r`` at
  the start matches an expectation over the analytic terminal utilities.

A third test asserts the solver outputs satisfy the recursive equations
(``eq-Vr`` / ``eq-pir`` and ``V_r = U_r`` at terminals) on every visited state —
a regression guard for the terminal-value bug class these tests were written for.
"""

import numpy as np
import pytest

from empo.solvers.params import EmpoParameter
from empo.envs.grid.base import GridConfig, GridState
import empo.envs.grid.moving_box as moving_box
import empo.envs.symbolic.trolley_problem as trolley
import empo.envs.symbolic.interruptibility as interruptibility
from empo.solvers import backward_induction as solvino
from empo.solvers import stochastic_backward_induction as stoch_solvino

# All analytic formulas below assume these Empo parameters.
PARAMS = EmpoParameter(gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1)


# --- Trolley problem vs BackwardInductionSolver ----------------------------


def build_trolley(n0: int, n1: int, ms: int, m0: int):
    """Population with ``n0`` humans killed by action 0 (survivors[0]) and
    ``n1`` killed by action 1 (survivors[1]); each carries ``ms`` survival
    goals and ``m0`` passivity goals on top of a constant baseline."""
    return trolley.make_env(
        trolley.PopConfig(
            n_killed_by_passive=n0,
            n_killed_by_pressing=n1,
            m_survival_goals=ms,
            m_passivity_goals=m0,
        )
    )


def trolley_utilities(n0: int, n1: int, ms: int, m0: int) -> tuple[float, float]:
    """Analytic terminal U_r for action 0 (passive) and action 1 (press)."""
    u0 = -(n0 / (1 + m0) + n1 / (1 + ms + m0))  # eq-trolley-Uu
    u1 = -(n0 / (1 + ms) + n1)  # eq-trolley-Up
    return u0, u1


@pytest.mark.parametrize(
    "n0, n1, ms, m0, expected_action",
    [
        (3, 1, 1, 0, 1),  # doc version 1: U0=-3.5, U1=-2.5 -> press
        (3, 1, 1, 1, 0),  # doc version 2: U0=-11/6, U1=-2.5 -> passive
        (1, 1, 1, 0, 0),  # symmetric humans, no preference -> passive (tie, argmax->0)
        (2, 5, 2, 0, 0),  # U0=-11/3, U1=-17/3 -> passive (pressing kills more)
        (4, 1, 3, 2, 0),
    ],
)
def test_trolley_matches_analytic(n0, n1, ms, m0, expected_action) -> None:
    func_env = build_trolley(n0, n1, ms, m0)
    start = trolley.State()

    solver = solvino.BackwardInductionSolver(func_env, PARAMS)
    solver.solve(start)

    u0, u1 = trolley_utilities(n0, n1, ms, m0)
    term0 = func_env.transition(start, 0)
    term1 = func_env.transition(start, 1)

    # solver U_r at each terminal matches the closed form
    assert solver.U_r[term0] == pytest.approx(u0)
    assert solver.U_r[term1] == pytest.approx(u1)

    # robot picks the higher-utility action
    assert solver.robot_policy[start] == expected_action
    assert expected_action == int(np.argmax([u0, u1]))

    # eq-Vr-unrolled for N=1: V_r(s0) = 2 * U_r(s_N) (gamma_r = gamma_h = 1)
    chosen_terminal = func_env.transition(start, expected_action)
    assert solver.V_r[start] == pytest.approx(2 * solver.U_r[chosen_terminal])


# --- Interruptibility vs StochasticBackwardInductionSolver -----------------


@pytest.mark.parametrize(
    "m_task, m_int, pause_prob",
    [
        (1, 0, 0.5),
        (1, 0, 0.2),
        (2, 1, 0.5),
        (0, 1, 0.7),
        (3, 2, 0.5),
    ],
)
def test_interruptibility_matches_analytic(m_task, m_int, pause_prob) -> None:
    config = interruptibility.EnvConfig(pause_prob=pause_prob)
    pop_config = interruptibility.PopConfig(
        m_task_done_goals=m_task, m_is_interruptible_goals=m_int
    )
    func_env = interruptibility.make_env(config, pop_config)
    start = interruptibility.State()

    solver = stoch_solvino.StochasticBackwardInductionSolver(func_env, PARAMS)
    solver.solve(start)

    # single human; every action leads straight to a terminal state
    def u_terminal(state: interruptibility.State) -> float:
        x = 1 + m_task * state.task_done + m_int * state.is_interruptible
        return -1.0 / x

    def q_analytic(action: int) -> float:
        states, probs = func_env.distribution(start, action)
        return PARAMS.gamma_r * sum(p * u_terminal(s) for s, p in zip(states, probs))

    for action in range(func_env.num_actions):
        assert solver.Q_r[start][action] == pytest.approx(q_analytic(action))

    expected = int(np.argmax([q_analytic(a) for a in range(func_env.num_actions)]))
    assert solver.robot_policy[start] == expected


# --- Equation invariants on every visited state ----------------------------


def _check_invariants(solver, func_env) -> None:
    for state in solver.V_r:
        if func_env.terminal(state):
            # terminal: no future actions, V_r collapses to U_r
            assert solver.Q_r[state] == {}
            assert solver.V_r[state] == pytest.approx(solver.U_r[state])
        else:
            best = solver.robot_policy[state]
            q = solver.Q_r[state]
            # eq-pir: policy is greedy in Q_r
            assert best == max(q, key=q.get)
            # eq-Vr: V_r = U_r + Q_r(s, pi(s))
            assert solver.V_r[state] == pytest.approx(solver.U_r[state] + q[best])


def test_deterministic_solver_invariants_trolley() -> None:
    func_env = build_trolley(n0=3, n1=1, ms=1, m0=1)
    solver = solvino.BackwardInductionSolver(func_env, PARAMS)
    solver.solve(trolley.State())
    _check_invariants(solver, func_env)


@pytest.mark.parametrize("size", [5, 7])
def test_deterministic_solver_invariants_moving_box(size: int) -> None:
    func_env = moving_box.Env(
        GridConfig(
            width=size, height=size, max_steps=2 * size, walls=frozenset()
        ),
        moving_box.fair_box_population(size),
    )
    start = GridState(robot=(0, 0), objects=((1, 0),), step=0)
    solver = solvino.BackwardInductionSolver(func_env, PARAMS)
    solver.solve(start)
    _check_invariants(solver, func_env)


def test_stochastic_solver_invariants_interruptibility() -> None:
    func_env = interruptibility.make_env(
        interruptibility.EnvConfig(pause_prob=0.5),
        interruptibility.PopConfig(m_task_done_goals=2, m_is_interruptible_goals=1),
    )
    solver = stoch_solvino.StochasticBackwardInductionSolver(func_env, PARAMS)
    solver.solve(interruptibility.State())
    _check_invariants(solver, func_env)
