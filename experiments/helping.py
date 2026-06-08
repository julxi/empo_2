import empo.envs.race as r
from empo import (
    Population,
    StochasticBackwardInductionSolver,
    EmpoParameter,
    at_terminal,
)
import random

n_racers = 2
len_track = 6

config = r.RaceConfig(n_racers=n_racers, len_track=len_track, max_steps=2 * len_track)
mode = r.Mode.PUSH
rng = random.Random()


def winner_population(n_racers: int) -> Population:
    population = []

    for racer in range(n_racers):
        human = []
        for pos in range(n_racers):
            human.append(r.position_goal(racer, pos))
        # terminal goal
        human.append(lambda o: float(at_terminal(o)))
        population.append(human)

    return population


population = winner_population(n_racers)


# pulling into negative

env = r.RaceEnv(config, population, mode, trip_prob=0.5)
start = r.RaceState(progress=(0, 0), race_result=(), step=0)
action = env.noop_action
params = EmpoParameter()
solver = StochasticBackwardInductionSolver(env, params)

solver.solve(start)


state = r.RaceState(progress=(0, 0), race_result=(), step=5)
print(solver.robot_policy[state])
