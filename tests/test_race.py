"""Tests for the race environment.

Each test pins down behaviour of a bug that was fixed in
:mod:`empo.envs.symbolic.race` — negative progress, recording a push over the finish
line, the step clock, the just-finished assertion, duplicate finishes, and the
``position_goal`` indexing — plus the push/pull movement semantics and the
stochastic ``distribution``.

The transition tests use ``trip_prob`` of 0 or 1 so the outcome is certain and
no rng is needed; the ``distribution`` tests cover the genuinely stochastic case.
"""

import pytest

import empo.envs.symbolic.race as r

N_RACERS = 3
LEN_TRACK = 5
FINISH_LINE = LEN_TRACK - 1


def make_env(mode: r.Mode, trip_prob: float) -> r.Env:
    config = r.EnvConfig(n_racers=N_RACERS, len_track=LEN_TRACK)
    population: r.Population = []  # population is irrelevant to the dynamics
    return r.Env(config, population, mode, trip_prob=trip_prob)


def obs_for(progress: tuple[int, ...], race_result: tuple[int, ...]) -> r.Obs:
    config = r.EnvConfig(n_racers=N_RACERS, len_track=LEN_TRACK)
    state = r.State(progress=progress, race_result=race_result)
    return r.Obs(config=config, state=state)


# action encoding: with PULL|PUSH, actions [0, n) pull and [n, 2n) push.
def pull(racer: int) -> int:
    return racer


def push(racer: int) -> int:
    return N_RACERS + racer


# --- Bug 2: pulling never produces negative progress -----------------------


def test_pull_does_not_go_negative() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=1)  # nobody steps
    state = r.State(progress=(0,) * N_RACERS, race_result=())

    new_state = env.transition(state, pull(0))

    assert new_state.progress[0] == 0
    assert all(p >= 0 for p in new_state.progress)


# --- Bug 1: a push over the finish line is recorded in race_result ---------


def test_push_onto_finish_is_recorded() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0)  # everyone steps
    state = r.State(progress=(LEN_TRACK - 2,) * N_RACERS, race_result=())

    new_state = env.transition(state, push(0))

    # pushed racer stops exactly on the line and is recorded
    assert new_state.progress[0] == FINISH_LINE
    assert 0 in new_state.race_result


def test_finisher_stops_on_the_line_not_beyond() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0)
    state = r.State(progress=(LEN_TRACK - 2,) * N_RACERS, race_result=())

    new_state = env.transition(state, push(0))

    # +1 push and +1 step would overshoot, but a finisher is clamped to the line
    assert all(p == FINISH_LINE for p in new_state.progress)
    assert set(new_state.race_result) == {0, 1, 2}


# --- Bug 3: the step clock advances ----------------------------------------


def test_step_is_incremented() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=1)
    state = r.State(step=3, progress=(0,) * N_RACERS, race_result=())

    new_state = env.transition(state, push(0))

    assert new_state.step == 4


# --- Bug 6: the state produced the moment a racer finishes is valid input --


def test_just_finished_state_is_accepted_as_input() -> None:
    """A finisher sits on ``len_track - 1`` while in ``race_result``; feeding
    that state back into ``transition`` must not trip the assertion."""
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=1)
    just_finished = r.State(
        progress=(FINISH_LINE, 1, 1), race_result=(0,)
    )

    new_state = env.transition(just_finished, push(1))

    # the already-finished racer is moved past the line on the next step
    assert new_state.progress[0] == LEN_TRACK


# --- Bug 5: a finished racer is not recorded twice -------------------------


def test_no_duplicate_finish_when_pulled_back() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0)
    just_finished = r.State(
        progress=(FINISH_LINE, 1, 1), race_result=(0,)
    )

    new_state = env.transition(just_finished, pull(0))

    assert list(new_state.race_result).count(0) == 1


# --- push / pull movement semantics ----------------------------------------


def test_push_advances_by_two_without_trip() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0)  # always steps
    state = r.State(progress=(0,) * N_RACERS, race_result=())

    new_state = env.transition(state, push(0))

    assert new_state.progress[0] == 2  # +1 push, +1 step
    assert new_state.progress[1] == 1  # unaffected racer just steps


def test_push_advances_by_one_with_trip() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=1)  # never steps
    state = r.State(progress=(0,) * N_RACERS, race_result=())

    new_state = env.transition(state, push(0))

    assert new_state.progress[0] == 1  # +1 push, no step
    assert new_state.progress[1] == 0  # unaffected racer stays put


def test_pull_nets_zero_without_trip() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0)
    state = r.State(progress=(2,) * N_RACERS, race_result=())

    new_state = env.transition(state, pull(0))

    assert new_state.progress[0] == 2  # -1 pull, +1 step


# --- Bug 4: position_goal does not index past race_result ------------------


def test_position_goal_no_index_error_mid_race() -> None:
    # only the leader has finished; ask for a 3rd-place (pos=2) goal
    obs = obs_for(progress=(FINISH_LINE, 2, 1), race_result=(0,))

    assert r.position_goal(0, 2)(obs) == 1.0  # leader is within the first 3
    assert r.position_goal(1, 2)(obs) == 0.0  # racer 1 has not finished


@pytest.mark.parametrize(
    "pos, expected",
    [(0, 1.0), (1, 1.0), (2, 1.0)],
)
def test_position_goal_leader_satisfies_every_place(pos: int, expected: float) -> None:
    obs = obs_for(progress=(FINISH_LINE,) * N_RACERS, race_result=(0, 1, 2))
    assert r.position_goal(0, pos)(obs) == expected


def test_position_goal_requires_a_good_enough_place() -> None:
    obs = obs_for(progress=(FINISH_LINE,) * N_RACERS, race_result=(0, 1, 2))
    # racer 2 finished last, so only the pos=2 goal is satisfied
    assert r.position_goal(2, 0)(obs) == 0.0
    assert r.position_goal(2, 1)(obs) == 0.0
    assert r.position_goal(2, 2)(obs) == 1.0


def test_position_goal_zero_once_past_the_line() -> None:
    obs = obs_for(progress=(LEN_TRACK,) * N_RACERS, race_result=(0, 1, 2))
    assert r.position_goal(0, 0)(obs) == 0.0  # already past the finish line


def test_position_goal_zero_before_finishing() -> None:
    obs = obs_for(progress=(2, 1, 0), race_result=())
    assert r.position_goal(0, 0)(obs) == 0.0


# --- StochasticEnv: distribution -------------------------------------------


def test_distribution_sums_to_one() -> None:
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0.5)
    state = r.State(progress=(0, 1, 2), race_result=())

    states, probs = env.distribution(state, push(0))

    assert len(states) == len(probs)
    assert sum(probs) == pytest.approx(1.0)
    assert len(states) == len(set(states))  # outcomes are merged, not duplicated


def test_distribution_two_outcomes_for_a_single_racer() -> None:
    # racers 1 and 2 have finished; only racer 0 is still racing -> trip or step
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0.3)
    state = r.State(progress=(1, FINISH_LINE, FINISH_LINE), race_result=(1, 2))

    states, probs = env.distribution(state, pull(0))

    assert len(states) == 2
    assert sorted(probs) == pytest.approx([0.3, 0.7])
    # both outcomes leave the finished racers parked past the line
    assert all(s.progress[1] == LEN_TRACK and s.progress[2] == LEN_TRACK for s in states)


def test_distribution_collapses_when_outcome_is_certain() -> None:
    # racer 0 is pushed onto the line; tripping or stepping yields the same state
    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0.5)
    state = r.State(progress=(FINISH_LINE - 1, FINISH_LINE, FINISH_LINE), race_result=(1, 2))

    states, probs = env.distribution(state, push(0))

    assert len(states) == 1
    assert probs[0] == pytest.approx(1.0)
    assert states[0].progress[0] == FINISH_LINE
    assert 0 in states[0].race_result


def test_transition_samples_from_distribution() -> None:
    import numpy as np

    env = make_env(r.Mode.PUSH | r.Mode.PULL, trip_prob=0.5)
    state = r.State(progress=(0, 1, 2), race_result=())

    valid = set(env.distribution(state, push(0))[0])
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert env.transition(state, push(0), rng) in valid
