from dataclasses import dataclass
import numpy as np

from .env import GridWorldFuncEnv


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
        self.U_r: dict = {}
        self.V_r: dict = {}

    def solve(self, state):
        if state in self.V_r:
            return
        if state in self.in_progress:
            return
        if self.env.terminal(state):
            self.V_h[state] = 0
            self.V_r[state] = 0
            return

        self.in_progress.add(state)

        actions = list(range(self.env.action_space.n))
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
        U = self.env.reward(state, best_action, next_state)
        self.V_h[state] = U + self.params.gamma_h * self.V_h[next_state]

        # U_r
        self.U_r[state] = -(self.V_h[state] ** self.params.zeta)

        # V_r
        self.V_r[state] = self.U_r[state] + self.Q_r[state][best_action]

        self.in_progress.remove(state)
