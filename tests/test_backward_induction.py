import pytest

from empo.solvers.params import EmpoParameter
from empo.envs.grid.base import GridConfig, GridState
import empo.envs.grid.moving_box as moving_box
from empo.solvers import backward_induction as solvino


@pytest.mark.parametrize("size", [5, 7, 9])
def test_robot_picks_fair_middle_box_column(size: int) -> None:
    max_steps = 2 * size

    population = moving_box.fair_box_population(size)

    func_env = moving_box.Env(
        GridConfig(width=size, height=size, max_steps=max_steps, walls=frozenset()),
        population,
    )
    start_state = GridState(robot=(0, 0), objects=((1, 0),), step=0)

    params = EmpoParameter(
        gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1,
    )
    solver = solvino.BackwardInductionSolver(func_env, params)
    solver.solve(start_state)

    assert start_state in solver.V_r

    current_state = start_state
    while not func_env.terminal(current_state):
        action = solver.robot_policy[current_state]
        current_state = func_env.transition(current_state, action)

    assert current_state.objects[0][0] == (size - 1) // 2
