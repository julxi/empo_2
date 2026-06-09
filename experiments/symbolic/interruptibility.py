import empo.envs.symbolic.interruptibility as e
import empo.solvers.stochastic_backward_induction as s
from empo.solvers.params import EmpoParameter
import numpy.random as rnd
from rich import print

# env
config = e.EnvConfig(pause_prob=0.1)
pop_config = e.PopConfig(m_task_done_goals=5, m_is_interruptible_goals=5)

env = e.make_env(config, pop_config)

start_state = e.State()

# solv
params = EmpoParameter()
solver = s.StochasticBackwardInductionSolver(env, params)

## ! do solve

solver.solve(start_state)


print("Q_r:", solver.Q_r[start_state])
