import src.grid_world.env as env
import src.grid_world.solvino as solvino
import numpy as np

size = 7
max_steps = 10
rng = np.random.default_rng(0)

# pop
human_1 = []
for i in range(size):
    human_1.append(lambda x, i=i: float(x.box[0] <= i))
human_2 = []
for i in range(size):
    human_2.append(lambda x, i=i: float(x.box[0]) >= i)
population = [human_1, human_2]

# env
func_env = env.GridWorldFuncEnv(size, population, max_steps=max_steps)
start_state = env.GridWorldState(
    agent=(0, 0),
    target=(0, 0),
    box=(1, 0),
    step=0,
)

# solve
params = solvino.EmpoParameter(
    gamma_r=1,
    beta_r=1,
    gamma_h=1,
    zeta=2,
    xi=1,
    eta=1,
)
solver = solvino.BackwardInductionSolver(func_env, params)
solver.solve(start_state)

print("V_r at start:", solver.V_r[start_state])

print("Theoretical prediction of box column:", (size - 1) / 2)
current_state = start_state
while not func_env.terminal(current_state):
    next_action = solver.robot_policy[current_state]
    current_state = func_env.transition(current_state, next_action)
print("Actual box column:", current_state.box[0])
