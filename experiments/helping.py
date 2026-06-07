import empo.envs.race as r
from empo import Population
import random

n_racers = 3
len_track = 5

config = r.RaceConfig(n_racers=n_racers, len_track=len_track)
mode = r.Mode.PUSH | r.Mode.PULL
rng = random.Random()


def winner_population(n_racers: int) -> Population:
    population = []

    for racer in range(n_racers):
        human = []
        for pos in range(n_racers):
            human.append(r.position_goal(racer, pos))
        population.append(human)

    return population


population = winner_population(n_racers)


# pulling into negative

env = r.RaceEnv(config, population, mode, trip_prob=1)
state = r.RaceState(progress=(0,) * n_racers, race_result=())
action = 0  # pull racer0
new_state = env.transition(state, action, rng)
print("Racer0 stays non-zero", new_state)

# pushing over finish

env = r.RaceEnv(config, population, mode, trip_prob=0)
state = r.RaceState(progress=(len_track - 2,) * n_racers, race_result=())
action = n_racers  # push racer0
new_state = env.transition(state, action, rng)
print("Racer0 stops at finish line", new_state)

# goals seem to work

env = r.RaceEnv(config, population, mode, trip_prob=1)
state = r.RaceState(
    progress=tuple(
        [
            len_track - 1,
        ]
        * (n_racers - 1)
        + [0]
    ),
    race_result=tuple(range(n_racers - 1)),
)

print("Just finished the race gets goals", env.goal_values(state))

new_state = env.transition(state, 0, rng)
print(new_state)

print("Just finished the race gets goals", env.goal_values(new_state))


env = r.RaceEnv(config, population, mode, trip_prob=1)
state = r.RaceState(
    progress=tuple(
        [
            len_track,
        ]
        * (n_racers - 1)
        + [0]
    ),
    race_result=tuple(range(n_racers - 1)),
)

print("Already finished the race no goals", env.goal_values(state))
