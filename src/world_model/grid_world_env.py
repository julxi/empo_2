from itertools import product

from .grid_world_core import Actions, GridWorldCore, GridWorldState, Pos

_ALL_DIRECTIONS: tuple[Actions, ...] = (
    Actions.STAY,
    Actions.UP,
    Actions.DOWN,
    Actions.LEFT,
    Actions.RIGHT,
)


class GridWorldEnv:
    """Single-robot RL distribution environment over a GridWorldCore.

    Humans are hardcoded as uniform random over Direction. step() returns the
    full distribution over next states given a single robot action.
    """

    def __init__(self, core: GridWorldCore):
        self.core = core

    def actions(self) -> tuple[Actions, ...]:
        return _ALL_DIRECTIONS

    def is_terminal(self, state: GridWorldState) -> bool:
        return self.core.is_terminal(state)

    def num_humans(self, state: GridWorldState) -> int:
        return len(state.humans)

    def goals(self, h_idx: int) -> tuple[Pos, ...]:
        return self.core.goals[h_idx]

    def human_at_goal(self, state: GridWorldState, h_idx: int, goal: Pos) -> bool:
        return state.humans[h_idx] == goal

    def step(
        self, state: GridWorldState, action: Actions
    ) -> tuple[list[GridWorldState], list[float]]:
        assert len(state.robots) == 1, "GridWorldEnv requires exactly one robot"

        n_humans = len(state.humans)
        human_action_tuples = list(product(_ALL_DIRECTIONS, repeat=n_humans))
        prob = 1.0 / len(human_action_tuples)

        dist: dict[GridWorldState, float] = {}
        for human_actions in human_action_tuples:
            new_state = self.core.step(state, (action,), human_actions)
            dist[new_state] = dist.get(new_state, 0.0) + prob
        return list(dist.keys()), list(dist.values())
