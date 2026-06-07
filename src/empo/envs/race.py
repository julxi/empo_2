from dataclasses import replace, dataclass
from typing import Any
from enum import Flag, auto

from ..core import DeterministicEnv, at_terminal, Population, EnvConfig, Obs, State


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


class RaceEnv(DeterministicEnv[RaceConfig, RaceState]):
    def __init__(
        self,
        config: RaceConfig,
        population: Population,
        mode: Mode,
        trip_prob: float = 0.5,
    ) -> None:
        n_modes = bin(mode.value).count("1")
        self.num_actions = n_modes * config.n_racers

        super().__init__(config, population)
        self.trip_prob = trip_prob
        self.mode = mode

    def transition(
        self,
        state: RaceState,
        action: int,
        rng: Any = None,
        params: Any = None,
    ) -> RaceState:
        new_step = state.step + 1

        assert len(state.progress) == self.config.n_racers
        assert len(state.race_result) <= self.config.n_racers

        for racer in state.race_result:
            assert state.progress[racer] >= self.config.len_track - 1

        new_progress = list(state.progress)
        new_race_result = list(state.race_result)

        # robot push/pull racer
        racer_affected = action % self.config.n_racers
        mode = action // self.config.n_racers
        effect = -1 if mode == 0 and Mode.PULL in self.mode else +1
        new_progress[racer_affected] += effect

        finish_line = self.config.len_track - 1
        for racer in range(self.config.n_racers):
            if racer in state.race_result:  # already past the finish line
                new_progress[racer] = self.config.len_track
                continue

            if rng.random() > self.trip_prob:  # normal step unless tripped
                new_progress[racer] += 1

            new_progress[racer] = max(0, new_progress[racer])  # no negative progress

            if new_progress[racer] >= finish_line:  # stop on the line, record once
                new_progress[racer] = finish_line
                new_race_result.append(racer)

        return RaceState(
            step=new_step,
            progress=tuple(new_progress),
            race_result=tuple(new_race_result),
        )


type RaceObs = Obs[RaceConfig, RaceState]


def position_goal(h_idx: int, pos: int):
    """1 iff human is at least pos."""

    def g(obs: RaceObs) -> float:
        if obs.state.progress[h_idx] == obs.config.len_track - 1:  # just finished
            return 1.0 if h_idx in obs.state.race_result[: pos + 1] else 0.0
        return 0.0  # not finished yet, or already past the line

    return g
