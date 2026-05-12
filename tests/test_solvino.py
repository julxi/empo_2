import pytest

import grid_world.env as env
import grid_world.solvino as solvino


@pytest.mark.parametrize("size", [5, 7, 9])
def test_robot_picks_fair_middle_box_column(size: int) -> None:
    max_steps = 2 * size

    human_1 = [(lambda x, i=i: float(x.box[0] <= i)) for i in range(size)]
    human_2 = [(lambda x, i=i: float(x.box[0] >= i)) for i in range(size)]
    population = [human_1, human_2]

    func_env = env.GridWorldFuncEnv(size, population, max_steps=max_steps)
    start_state = env.GridWorldState(
        agent=(0, 0), target=(0, 0), box=(1, 0), step=0
    )

    params = solvino.EmpoParameter(
        gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1,
    )
    solver = solvino.BackwardInductionSolver(func_env, params)
    solver.solve(start_state)

    assert start_state in solver.V_r

    current_state = start_state
    while not func_env.terminal(current_state):
        action = solver.robot_policy[current_state]
        current_state = func_env.transition(current_state, action)

    assert current_state.box[0] == (size - 1) // 2
