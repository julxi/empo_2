import src.grid_world.env as env
import src.grid_world.solvino as solvino
import numpy as np

size = 5
# 0 1 2 3 4
rng = np.random.default_rng(0)

# pop
human = []
for i in range(size):
    human.append(lambda x, i=i: float(x.box[0] <= i))
population = [human]

# env
func_env = env.GridWorldFuncEnv(size, population, max_steps=2 * size)
start_state = env.GridWorldState(
    agent=(4, 4),
    target=(3, 4),
    box=(2, 0),
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
print("actual optimum:", -((len(human) ** (-params.xi)) ** params.eta))

box_in_best_state = env.GridWorldState(
    agent=(4, 4),
    target=(3, 4),
    box=(0, 0),
    step=0,
)
print("V_r also found by solver:", solver.V_r[box_in_best_state])
