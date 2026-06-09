import pytest

import empo.solvers.trajectory as trajectory
from empo.solvers.params import EmpoParameter
from empo.envs.grid.base import GridConfig, GridState
import empo.envs.grid.moving_box as moving_box
from empo.solvers import backward_induction as solvino


@pytest.mark.parametrize("size", [5, 7])
def test_trajectory_matches_backward_induction(size: int) -> None:
    max_steps = 2 * size
    func_env = moving_box.Env(
        GridConfig(width=size, height=size, max_steps=max_steps, walls=frozenset()),
        moving_box.fair_box_population(size),
    )
    start = GridState(robot=(0, 0), objects=((1, 0),), step=0)

    params = EmpoParameter(
        gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1,
    )
    solver = solvino.BackwardInductionSolver(func_env, params)
    solver.solve(start)

    traj = trajectory.evaluate_policy(
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
    func_env = moving_box.Env(
        GridConfig(width=size, height=size, max_steps=max_steps, walls=frozenset()),
        moving_box.fair_box_population(size),
    )
    start = GridState(robot=(2, 2), objects=((2, 3),), step=0)

    params = EmpoParameter(
        gamma_r=0.9, beta_r=1, gamma_h=0.95, zeta=2, xi=1, eta=1,
    )
    solver = solvino.BackwardInductionSolver(func_env, params)
    solver.solve(start)

    traj = trajectory.evaluate_policy(
        func_env, params, lambda s: solver.robot_policy[s], start
    )

    assert traj.V_r[0] == pytest.approx(solver.V_r[start])
