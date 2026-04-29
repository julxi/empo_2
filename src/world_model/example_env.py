import grid_world
import world_model

gw = grid_world.GridWorld(width=3, height=1, walls=frozenset())
env = world_model.GWEnvironment(
    gw,
    max_steps=3,
    goals=[[grid_world.Pos(1, 0)]],
)


empo_params = world_model.EmpoParameter(
    gamma_r=1,
    beta_r=1,
    gamma_h=1,
    zeta=1,
    xi=1,
    eta=1,
)

solver = world_model.BackwardInductionSolver(env, empo_params)

# solving

gw_start_state = grid_world.GridWorldState(
    crates=(),
    robots=(grid_world.Pos(1, 0),),
    humans=(grid_world.Pos(0, 0),),
)

env_state = world_model.GWEnvState(
    grid_world_state=gw_start_state,
    time_step=0,
)

robot_policy = solver.compute_robot_policy(env_state)

all_Directions = [
    grid_world.Direction.STAY,
    grid_world.Direction.UP,
    grid_world.Direction.DOWN,
    grid_world.Direction.LEFT,
    grid_world.Direction.RIGHT,
]
for action in all_Directions:
    print(action, robot_policy[env_state, action])
