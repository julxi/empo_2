from dataclasses import dataclass
import numpy as np

from .env import Action, GridWorldFuncEnv


@dataclass(frozen=True)
class EmpoParameter:
    gamma_r: float
    beta_r: float
    gamma_h: float
    zeta: float
    xi: float
    eta: float


# Special case that solves on a spanning tree
class BackwardInductionSolver:
    def __init__(self, func_env: GridWorldFuncEnv, params: EmpoParameter):
        self.env = func_env
        self.params = params

        self.in_progress: set = set()
        self.Q_r: dict = {}
        self.robot_policy: dict = {}
        self.V_h: dict = {}
        self.X_h: dict = {}
        self.U_r: dict = {}
        self.V_r: dict = {}

    def solve(self, state):
        if state in self.V_r:
            return
        if state in self.in_progress:
            return
        if self.env.terminal(state):
            self.V_h[state] = [
                [0.0] * len(human_goals) for human_goals in self.env.population
            ]
            self.V_r[state] = 0.0
            return

        self.in_progress.add(state)

        actions = list(Action)
        valid_actions = []
        for action in actions:
            next_state = self.env.transition(state, action)
            if next_state in self.in_progress:
                continue
            self.solve(next_state)
            if next_state in self.V_r:
                valid_actions.append(action)

        if not valid_actions:
            # All paths from here close a cycle; leave V_r unset so the
            # caller drops the action that led here. A different ancestor
            # may still reach `state` via a non-cycling route later.
            return

        # Q_r
        q_values = [
            self.params.gamma_r * self.V_r[self.env.transition(state, a)]
            for a in valid_actions
        ]
        self.Q_r[state] = dict(zip(valid_actions, q_values))

        # (5) robot policy
        best_action = valid_actions[int(np.argmax(q_values))]
        self.robot_policy[state] = best_action

        next_state = self.env.transition(state, best_action)

        # V_h
        rewards = self.env.reward(state, best_action, next_state)
        self.V_h[state] = [
            [
                r + self.params.gamma_h * v_next
                for r, v_next in zip(human_rewards, human_v_next)
            ]
            for human_rewards, human_v_next in zip(rewards, self.V_h[next_state])
        ]

        # X_h
        self.X_h[state] = [
            sum(v ** self.params.zeta for v in human_v) for human_v in self.V_h[state]
        ]

        # U_r
        fair_power = sum(x ** (-self.params.xi) for x in self.X_h[state])
        self.U_r[state] = -(fair_power**self.params.eta)

        # V_r
        self.V_r[state] = self.U_r[state] + self.Q_r[state][best_action]

        self.in_progress.remove(state)
