import numpy as np

from ..base import Action
from ..empo import EmpoParameter
from ..env_base import StochasticGridWorldEnv


class StochasticBackwardInductionSolver:
    """
    Backward induction for goal-independent stochastic environments.

    Only usable on acyclic environments. Uses ``env.distribution(s, a)`` to
    take expectations over next states.
    """

    def __init__(
        self,
        func_env: StochasticGridWorldEnv,
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
            obs = self.env.observation(state)
            self.V_h[state] = [
                [goal(obs) for goal in human_goals]
                for human_goals in self.env.population
            ]
            self.V_r[state] = 0.0
            return

        actions = list(Action)
        distributions = {}
        for action in actions:
            next_states, probs = self.env.distribution(state, action)
            distributions[action] = (next_states, probs)
            for next_state in next_states:
                self.solve(next_state)

        # Q_r: expectation over next states
        q_values = []
        for action in actions:
            next_states, probs = distributions[action]
            q = self.params.gamma_r * sum(
                p * self.V_r[s_next] for s_next, p in zip(next_states, probs)
            )
            q_values.append(q)
        self.Q_r[state] = dict(zip(actions, q_values))

        # robot policy
        best_action = actions[int(np.argmax(q_values))]
        self.robot_policy[state] = best_action

        next_states, probs = distributions[best_action]

        # V_h: 1 if s in g_h, else gamma_h * E[V_h(next)]
        obs = self.env.observation(state)
        v_h_state = []
        for h, human_goals in enumerate(self.env.population):
            row = []
            for g_idx, goal in enumerate(human_goals):
                g_here = goal(obs)
                if g_here > 0:
                    row.append(g_here)
                else:
                    row.append(
                        self.params.gamma_h
                        * sum(
                            p * self.V_h[s_next][h][g_idx]
                            for s_next, p in zip(next_states, probs)
                        )
                    )
            v_h_state.append(row)
        self.V_h[state] = v_h_state

        # X_h
        self.X_h[state] = [
            sum(v**self.params.zeta for v in human_v) for human_v in self.V_h[state]
        ]

        # U_r
        fair_power = sum(x ** (-self.params.xi) for x in self.X_h[state])
        self.U_r[state] = -(fair_power**self.params.eta)

        # V_r
        self.V_r[state] = self.U_r[state] + self.Q_r[state][best_action]
