import empo.envs.interruptibility as e
import empo.solvers.stochastic_backward_induction as s
from empo import EmpoParameter
import numpy.random as rnd
from rich import print

# env
config = e.InterrupConfig()
pop_config = e.InterrupPopConfig(m_task_done_goals=1, m_is_interruptible_goals=0)

env = e.make_interrupEnv(config, pop_config)

start_state = e.InterrupState()

# solv
params = EmpoParameter()
solver = s.StochasticBackwardInductionSolver(env, params)

## ! do solve

solver.solve(start_state)


print("Q_r:", solver.Q_r)

action_rewards = {}
for action in range(env.num_actions):
    states, probs = env.distribution(start_state, action)

    goal_probs = []
    for state, prob in zip(states, probs):
        goal_probs.append((env.goal_values(state), prob))

    action_rewards[action] = goal_probs

print("action -> (rewards, probs):", action_rewards)
