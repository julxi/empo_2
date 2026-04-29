from grid_world_core import GridWorldCore, Actions, Pos, GridWorldState
from grid_world_env import GridWorldEnv
from solver import EmpoParameter, BackwardInductionSolver

# params

width = 3
height = 1
robot_pos = Pos(1, 0)
humans_pos = (Pos(0, 0),)
goals = ((Pos(1, 0),),)

max_steps = 3

# environment and solver setup
gw_c = GridWorldCore(
    width=width, height=height, walls=frozenset(), max_steps=max_steps, goals=goals
)
gw_env = GridWorldEnv(gw_c)
gw_start_state = GridWorldState(
    crates=(),
    robots=(robot_pos,),
    humans=humans_pos,
    time_step=0,
)


empo_params = EmpoParameter(
    gamma_r=1,
    beta_r=1,
    gamma_h=1,
    zeta=1,
    xi=1,
    eta=1,
)
solver = BackwardInductionSolver(gw_env, empo_params)

# solving

robot_policy = solver.compute_robot_policy(gw_start_state)

print(robot_policy)
