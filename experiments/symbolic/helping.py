import empo.envs.symbolic.race as r
from empo.solvers.params import EmpoParameter
from empo.solvers.stochastic_backward_induction import (
    StochasticBackwardInductionSolver,
)
import random
from rich import print

n_racers = 2
len_track = 6
max_steps = 2 * len_track

config = r.EnvConfig(n_racers=n_racers, len_track=len_track, max_steps=max_steps)
mode = r.Mode.PULL | r.Mode.PUSH
rng = random.Random()


env = r.make_env(config, r.PopConfig(), mode, trip_prob=0.5)
start = r.State(progress=(0, 0), race_result=(), step=0)
action = env.noop_action
params = EmpoParameter()
solver = StochasticBackwardInductionSolver(env, params)

solver.solve(start)


for i in range(config.len_track - 1):
    state = r.State(progress=(i, 0), race_result=(), step=i)
    robot_action = solver.robot_policy[state]

    print(
        f"In state {state} robot chooses {robot_action}:{env.translate_action(robot_action)}"
    )
    print(f"{solver.Q_r[state]}")
