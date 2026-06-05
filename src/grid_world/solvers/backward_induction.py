import numpy as np

from ..base import Action
from ..empo import EmpoParameter
from ..env_base import DeterministicGridWorldEnv


class BackwardInductionSolver:
    """
    Backward induction solver for the acyclic, determinstic environments.
    (see `math/2_deterministic.typ` for theory).
    """

    def __init__(
        self,
        func_env: DeterministicGridWorldEnv,
        params: EmpoParameter = EmpoParameter(),
    ):
        self.env = func_env
        self.params = params

        self.Q_r: dict = {}
        self.robot_policy: dict = {}
        self.V_h: dict = {}
        self.X_h: dict = {}
        self.U_r: dict = {}
        self.V_r: dict = {}

    def solve(self, state):
        if state in self.V_r:
            return

        if self.env.terminal(state):
            self.V_h[state] = self.env.goal_values(state)
            self.V_r[state] = 0.0
            return

        # recursively compute successor states
        actions = list(Action)
        for action in actions:
            next_state = self.env.transition(state, action)
            self.solve(next_state)

        # Q_r
        q_values = [
            self.params.gamma_r * self.V_r[self.env.transition(state, a)]
            for a in actions
        ]
        self.Q_r[state] = dict(zip(actions, q_values))

        # robot policy
        best_action = actions[int(np.argmax(q_values))]
        self.robot_policy[state] = best_action

        next_state = self.env.transition(state, best_action)

        # V_h
        gv = self.env.goal_values(state)
        self.V_h[state] = [
            [
                g_here if g_here > 0 else self.params.gamma_h * v_next
                for g_here, v_next in zip(human_gv, human_v_next)
            ]
            for human_gv, human_v_next in zip(gv, self.V_h[next_state])
        ]

        # X_h
        self.X_h[state] = [
            sum(v**self.params.zeta for v in human_v) for human_v in self.V_h[state]
        ]

        # U_r
        fair_power = sum(x ** (-self.params.xi) for x in self.X_h[state])
        self.U_r[state] = -(fair_power**self.params.eta)

        # V_r
        self.V_r[state] = self.U_r[state] + self.Q_r[state][best_action]
