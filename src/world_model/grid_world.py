from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class Direction(Enum):
    STAY = 0
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4


_DELTAS = {
    Direction.STAY: (0, 0),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
    Direction.UP: (0, 1),
    Direction.DOWN: (0, -1),
}


@dataclass(frozen=True)
class Pos:
    x: int
    y: int

    def move(self, direction: Direction) -> Pos:
        dx, dy = _DELTAS[direction]
        return Pos(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class GridWorldState:
    crates: tuple[Pos, ...]
    robots: tuple[Pos, ...]
    humans: tuple[Pos, ...]


@dataclass(frozen=True)
class GridWorld:
    width: int
    height: int
    walls: frozenset[Pos]

    @staticmethod
    def create(
        width: int,
        height: int,
        walls: list[Pos],
    ) -> GridWorld:
        return GridWorld(width=width, height=height, walls=frozenset(walls))

    @staticmethod
    def make_state(
        crates: list[Pos], robots: list[Pos], humans: list[Pos]
    ) -> GridWorldState:
        return GridWorldState(
            crates=tuple(crates), robots=tuple(robots), humans=tuple(humans)
        )

    def _in_bounds(self, p: Pos) -> bool:
        return 0 <= p.x < self.width and 0 <= p.y < self.height

    def _is_free(
        self,
        p: Pos,
        crates: tuple[Pos, ...],
        robots: tuple[Pos, ...],
        humans: tuple[Pos, ...],
    ) -> bool:
        return (
            self._in_bounds(p)
            and p not in self.walls
            and p not in crates
            and p not in robots
            and p not in humans
        )

    @staticmethod
    def _crate_at(p: Pos, crates: tuple[Pos, ...]) -> int | None:
        for i, c in enumerate(crates):
            if c == p:
                return i
        return None

    def _try_move_robot(
        self,
        idx: int,
        direction: Direction,
        crates: tuple[Pos, ...],
        robots: tuple[Pos, ...],
        humans: tuple[Pos, ...],
    ) -> tuple[tuple[Pos, ...], tuple[Pos, ...]]:
        if direction == Direction.STAY:
            return crates, robots
        dst = robots[idx].move(direction)
        crate_idx = self._crate_at(dst, crates)
        if crate_idx is not None:
            beyond = dst.move(direction)
            if self._is_free(beyond, crates, robots, humans):
                new_crates = crates[:crate_idx] + (beyond,) + crates[crate_idx + 1 :]
                new_robots = robots[:idx] + (dst,) + robots[idx + 1 :]
                return new_crates, new_robots
        elif self._is_free(dst, crates, robots, humans):
            new_robots = robots[:idx] + (dst,) + robots[idx + 1 :]
            return crates, new_robots
        return crates, robots

    def _try_move_human(
        self,
        idx: int,
        direction: Direction,
        crates: tuple[Pos, ...],
        robots: tuple[Pos, ...],
        humans: tuple[Pos, ...],
    ) -> tuple[Pos, ...]:
        if direction == Direction.STAY:
            return humans
        dst = humans[idx].move(direction)
        if self._is_free(dst, crates, robots, humans):
            return humans[:idx] + (dst,) + humans[idx + 1 :]
        return humans

    def step(
        self,
        state: GridWorldState,
        robot_actions: tuple[Direction, ...],
        human_actions: tuple[Direction, ...],
    ) -> GridWorldState:
        crates, robots, humans = state.crates, state.robots, state.humans
        assert len(robot_actions) == len(robots)
        assert len(human_actions) == len(humans)

        for i in range(len(robots)):
            crates, robots = self._try_move_robot(
                i, robot_actions[i], crates, robots, humans
            )
        for i in range(len(humans)):
            humans = self._try_move_human(i, human_actions[i], crates, robots, humans)

        return GridWorldState(crates=crates, robots=robots, humans=humans)
