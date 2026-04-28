from grid_world import GridWorld, Direction, Pos, GridWorldState
from itertools import product
from dataclasses import dataclass

import random

all_Directions = [
    Direction.STAY,
    Direction.UP,
    Direction.DOWN,
    Direction.LEFT,
    Direction.RIGHT,
]


@dataclass(frozen=True)
class GWEnvState:
    grid_world_state: GridWorldState
    time_step: int


class GWEnvironment:
    def __init__(self, grid_world: GridWorld, max_steps: int, goals: list[list[Pos]]):

        self.gw = grid_world
        self.max_steps = max_steps
        # goals is human_idx -> list[Pos]
        self.goals = goals

    def step(
        self, state: GWEnvState, action_r: Direction, actions_h: tuple[Direction, ...]
    ):
        if self.is_terminal(state):
            raise ValueError("Can't step a terminal state")

        new_gw_state = self.gw.step(state.grid_world_state, (action_r,), actions_h)
        new_state = GWEnvState(new_gw_state, state.time_step + 1)
        return new_state

    def is_terminal(self, state: GWEnvState):
        return state.time_step == self.max_steps

    def goal_reward(self, state: GWEnvState, h_idx: int, goal_pos: Pos):
        return int(state.grid_world_state.humans[h_idx] == goal_pos)


@dataclass(frozen=True)
class EmpoParameter:
    gamma_r: float  # γ
    beta_r: float  # β
    gamma_h: float  # γ
    zeta: float  # ζ
    xi: float  # ξ
    eta: float  # η


class BackwardInductionSolver:
    def __init__(self, env: GWEnvironment, empo_params: EmpoParameter):
        self.env = env
        self.empo_params = empo_params

        self.robot_policy = {}
        self.V_h = {}
        self.X_h = {}
        self.U_r = {}
        self.V_r = {}
        self.Q_r = {}

        self.all_Directions = [
            Direction.STAY,
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
        ]

    def compute_robot_policy(self, init_state: GWEnvState):
        self._rec_compute_policy(init_state)
        return self.robot_policy

    def _rec_compute_policy(self, state: GWEnvState):
        if state in self.V_r:
            return

        gw_state = state.grid_world_state
        time_step = state.time_step

        if self.env.is_terminal(state):
            for h_idx in range(len(gw_state.humans)):
                for pos in self.env.goals[h_idx]:
                    self.V_h[h_idx, state, pos] = 1  # being alive goal? :)
            self.V_r[state] = -1e-3  # small negative number, but why?
            return

        # ----------------------------------
        # real recursion
        # ----------------------------------
        num_humans = len(gw_state.humans)
        all_human_actions = list(product(self.all_Directions, repeat=num_humans))
        num_human_actions = len(all_human_actions)

        # recursive compute all successor states
        for action in self.all_Directions:
            for human_actions in all_human_actions:
                new_state = self.env.step(state, action, human_actions)
                self._rec_compute_policy(new_state)

        # compute Q_r
        for action in self.all_Directions:
            self.Q_r[state, action] = 0
            for human_actions in all_human_actions:
                new_state = self.env.step(state, action, human_actions)
                self.Q_r[state, action] += (
                    self.V_r[new_state] * self.empo_params.gamma_r / num_human_actions
                )

        # compute robot policy
        for action in self.all_Directions:
            self.robot_policy[state, action] = (-self.Q_r[state, action]) ** (
                -self.empo_params.beta_r
            )
        total = 0
        for action in self.all_Directions:
            total += self.robot_policy[state, action]
        for action in self.all_Directions:
            self.robot_policy[state, action] /= total

        # compute V_h
        for h_idx in range(num_humans):
            for goal in self.env.goals[h_idx]:
                self.V_h[h_idx, state, goal] = 0
                for action in self.all_Directions:
                    for human_actions in all_human_actions:
                        new_state = self.env.step(state, action, human_actions)
                        self.V_h[h_idx, state, goal] += (
                            10 * self.env.goal_reward(new_state, h_idx, goal)
                            + self.empo_params.gamma_h
                            * self.V_h[h_idx, new_state, goal]
                        )

        # compute X_h
        for h_idx in range(num_humans):
            self.X_h[h_idx, state] = 0
            for goal in self.env.goals[h_idx]:
                self.X_h[h_idx, state] += (
                    self.V_h[h_idx, state, goal] ** self.empo_params.zeta
                )

        # compute U_r
        self.U_r[state] = 0
        for h_idx in range(num_humans):
            self.U_r[state] += self.X_h[h_idx, state] ** (-self.empo_params.xi)
        self.U_r[state] = -(self.U_r[state] ** self.empo_params.eta)

        # compute V_r
        self.V_r[state] = self.U_r[state]
        for action in self.all_Directions:
            self.V_r[state] += (
                self.robot_policy[state, action] * self.Q_r[state, action]
            )
