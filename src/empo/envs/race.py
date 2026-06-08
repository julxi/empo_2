import itertools
from dataclasses import dataclass
from typing import Any
from enum import Flag, auto

import numpy as np

from ..core import StochasticEnv, Population, EnvConfig, Obs, State


@dataclass(frozen=True)
class RaceConfig(EnvConfig):
    n_racers: int = 0
    len_track: int = 0


@dataclass(frozen=True)
class RaceState(State):
    progress: tuple[int, ...] = ()
    race_result: tuple[int, ...] = ()  # order of finish so far


class Mode(Flag):
    PUSH = auto()
    PULL = auto()


class RaceEnv(StochasticEnv[RaceConfig, RaceState]):
    def __init__(
        self,
        config: RaceConfig,
        population: Population,
        mode: Mode,
        trip_prob: float = 0.5,
    ) -> None:
        n_modes = bin(mode.value).count("1")
        # Push/pull actions index every (mode, racer) pair; the final action is a
        # mode-independent noop where the robot leaves every racer alone.
        self.noop_action = n_modes * config.n_racers
        self.num_actions = self.noop_action + 1

        super().__init__(config, population)
        self.trip_prob = trip_prob
        self.mode = mode

    def _next_states(
        self, state: RaceState, action: int
    ) -> tuple[list[RaceState], list[float]]:
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
        if action != self.noop_action:
            racer_affected = action % n_racers
            mode = action // n_racers
            pulling = mode == 0 and Mode.PULL in self.mode
            base_progress[racer_affected] += -1 if pulling else +1

        # Racers past the line are frozen; the rest each trip independently, so we
        # enumerate every trip/no-trip combination and weight it by its probability.
        racing = [r for r in range(n_racers) if r not in state.race_result]

        outcomes: dict[RaceState, float] = {}
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
            result = RaceState(
                step=next_step,
                progress=tuple(progress),
                race_result=tuple(race_result),
            )
            outcomes[result] = outcomes.get(result, 0.0) + prob

        return list(outcomes.keys()), list(outcomes.values())

    def transition(
        self,
        state: RaceState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> RaceState:
        states, probs = self._next_states(state, action)
        if len(states) == 1:
            return states[0]
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(states), p=probs)
        return states[idx]

    def distribution(
        self, state: RaceState, action: int
    ) -> tuple[list[RaceState], list[float]]:
        return self._next_states(state, action)


type RaceObs = Obs[RaceConfig, RaceState]


def position_goal(h_idx: int, pos: int):
    """1 iff human is at least pos."""

    def g(obs: RaceObs) -> float:
        if obs.state.progress[h_idx] == obs.config.len_track - 1:  # just finished
            return 1.0 if h_idx in obs.state.race_result[: pos + 1] else 0.0
        return 0.0  # not finished yet, or already past the line

    return g
