import itertools
from dataclasses import dataclass
from typing import Any
from enum import Flag, auto

import numpy as np

from ... import core
from ...core import Population, at_terminal


@dataclass(frozen=True)
class EnvConfig(core.EnvConfig):
    n_racers: int = 0
    len_track: int = 0


@dataclass(frozen=True)
class State(core.State):
    progress: tuple[int, ...] = ()
    race_result: tuple[int, ...] = ()  # order of finish so far


@dataclass(frozen=True)
class RaceAction:
    offset: int  # +1 push, -1 pull
    racer: int


class Mode(Flag):
    PUSH = auto()
    PULL = auto()


class Env(core.StochasticEnv[EnvConfig, State]):
    def __init__(
        self,
        config: EnvConfig,
        population: Population,
        mode: Mode,
        trip_prob: float = 0.5,
    ) -> None:
        n_modes = bin(mode.value).count("1")
        self.noop_action = n_modes * config.n_racers
        self.num_actions = self.noop_action + 1

        super().__init__(config, population)
        self.trip_prob = trip_prob
        self.mode = mode

    def translate_action(self, action: int) -> RaceAction | None:
        if action == self.noop_action:
            return None
        n_racers = self.config.n_racers
        racer_affected = action % n_racers
        mode = action // n_racers
        pulling = mode == 0 and Mode.PULL in self.mode
        return RaceAction(offset=-1 if pulling else +1, racer=racer_affected)

    def _next_states(
        self, state: State, action: int
    ) -> tuple[list[State], list[float]]:
        n_racers = self.config.n_racers
        finish_line = self.config.len_track - 1
        next_step = state.step + 1

        assert len(state.progress) == n_racers
        assert len(state.race_result) <= n_racers
        for racer in state.race_result:
            assert state.progress[racer] >= finish_line

        # The robot pushes/pulls one racer (noop touches no one). This nudge stacks
        # with whatever step that racer then takes on its own below.
        base_progress = list(state.progress)
        act = self.translate_action(action)
        if act is not None:
            base_progress[act.racer] += act.offset

        # Racers past the line are frozen; the rest each trip independently, so we
        # enumerate every trip/no-trip combination and weight it by its probability.
        racing = [r for r in range(n_racers) if r not in state.race_result]

        outcomes: dict[State, float] = {}
        for tripped in itertools.product((False, True), repeat=len(racing)):
            prob = 1.0
            progress = list(base_progress)
            race_result = list(state.race_result)

            for racer in state.race_result:  # finished earlier: park beyond the line
                progress[racer] = self.config.len_track

            for racer, trip in zip(racing, tripped):
                prob *= self.trip_prob if trip else 1.0 - self.trip_prob
                if not trip:  # a trip forfeits the normal step
                    progress[racer] += 1
                progress[racer] = max(0, progress[racer])  # no negative progress
                if progress[racer] >= finish_line:  # reached the line: stop and record
                    progress[racer] = finish_line
                    race_result.append(racer)

            if prob == 0.0:  # impossible combination when trip_prob is 0 or 1
                continue
            result = State(
                step=next_step,
                progress=tuple(progress),
                race_result=tuple(race_result),
            )
            outcomes[result] = outcomes.get(result, 0.0) + prob

        return list(outcomes.keys()), list(outcomes.values())

    def transition(
        self,
        state: State,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> State:
        states, probs = self._next_states(state, action)
        if len(states) == 1:
            return states[0]
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(states), p=probs)
        return states[idx]

    def distribution(
        self, state: State, action: int
    ) -> tuple[list[State], list[float]]:
        return self._next_states(state, action)


Obs = core.Obs[EnvConfig, State]


def position_goal(h_idx: int, pos: int):
    """1 iff human is at least pos."""

    def g(obs: Obs) -> float:
        if obs.state.progress[h_idx] == obs.config.len_track - 1:  # just finished
            return 1.0 if h_idx in obs.state.race_result[: pos + 1] else 0.0
        return 0.0  # not finished yet, or already past the line

    return g


def position_goal_at_step(h_idx: int, pos: int, step: int):
    """1 iff human is at least pos."""

    def g(obs: Obs) -> float:
        if obs.state.progress[h_idx] == obs.config.len_track - 1:  # just finished
            return 1.0 if h_idx in obs.state.race_result[: pos + 1] else 0.0
        return 0.0  # not finished yet, or already past the line

    return g


def terminal_goal():
    """1 iff at a terminal state (used as a baseline 'finish the race' goal)."""

    def g(obs: Obs) -> float:
        return float(at_terminal(obs))

    return g


@dataclass(frozen=True)
class PopConfig:
    m_position_goals: int = 1  # copies of each per-place goal a racer holds
    include_terminal_goal: bool = True


def make_population(config: EnvConfig, pop_config: PopConfig) -> Population:
    """One human per racer, each wanting to finish in every place (and, by
    default, for the race to terminate)."""
    population: Population = []
    for racer in range(config.n_racers):
        human = []
        for pos in range(config.n_racers):
            human.extend([position_goal(racer, pos)] * pop_config.m_position_goals)
        if pop_config.include_terminal_goal:
            human.append(terminal_goal())
        population.append(human)
    return population


def make_env(
    config: EnvConfig,
    pop_config: PopConfig,
    mode: Mode,
    trip_prob: float = 0.5,
) -> Env:
    population = make_population(config, pop_config)
    return Env(config, population, mode, trip_prob=trip_prob)
