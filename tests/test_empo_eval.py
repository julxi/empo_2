import math

import pytest

import grid_world as env
import grid_world.empo_eval as empo_eval
from grid_world.solvers import backward_induction as solvino


def _fair_population(size: int):
    human_1 = [(lambda x, i=i: float(x.state.object[0] <= i)) for i in range(size)]
    human_2 = [(lambda x, i=i: float(x.state.object[0] >= i)) for i in range(size)]
    return [human_1, human_2]


@pytest.mark.parametrize("size", [5, 7])
def test_trajectory_matches_backward_induction(size: int) -> None:
    max_steps = 2 * size
    func_env = env.MovingBoxEnv(
        env.GridWorldLayout(width=size, height=size, max_steps=max_steps, walls=frozenset()),
        _fair_population(size),
    )
    start = env.GridWorldState(robot=(0, 0), object=(1, 0), step=0)

    params = env.EmpoParameter(
        gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1,
    )
    solver = solvino.BackwardInductionSolver(func_env, params)
    solver.solve(start)

    traj = empo_eval.evaluate_policy(
        func_env, params, lambda s: solver.robot_policy[s], start
    )

    assert traj.V_r[0] == pytest.approx(solver.V_r[start])
    for t in range(len(traj.states) - 1):
        s = traj.states[t]
        if s in solver.V_r:
            assert traj.V_r[t] == pytest.approx(solver.V_r[s])


def test_discounted_trajectory_agrees_with_solver() -> None:
    size = 5
    max_steps = 8
    func_env = env.MovingBoxEnv(
        env.GridWorldLayout(width=size, height=size, max_steps=max_steps, walls=frozenset()),
        _fair_population(size),
    )
    start = env.GridWorldState(robot=(2, 2), object=(2, 3), step=0)

    params = env.EmpoParameter(
        gamma_r=0.9, beta_r=1, gamma_h=0.95, zeta=2, xi=1, eta=1,
    )
    solver = solvino.BackwardInductionSolver(func_env, params)
    solver.solve(start)

    traj = empo_eval.evaluate_policy(
        func_env, params, lambda s: solver.robot_policy[s], start
    )

    assert traj.V_r[0] == pytest.approx(solver.V_r[start])
