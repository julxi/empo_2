from dataclasses import dataclass


@dataclass(frozen=True)
class EmpoParameter:
    gamma_r: float
    beta_r: float
    gamma_h: float
    zeta: float
    xi: float
    eta: float


class BackwardInductionSolver:
    def __init__(self, env, params: EmpoParameter):
        self.env = env
        self.params = params

        self.robot_policy: dict = {}
        self.V_h: dict = {}
        self.X_h: dict = {}
        self.U_r: dict = {}
        self.V_r: dict = {}
        self.Q_r: dict = {}

    def compute_robot_policy(self, init_state):
        self._rec_compute_policy(init_state)
        return self.robot_policy

    def _rec_compute_policy(self, state):
        if state in self.V_r:
            return

        if self.env.is_terminal(state):
            n_h = self.env.num_humans(state)
            for h_idx in range(n_h):
                for goal in self.env.goals(h_idx):
                    self.V_h[h_idx, state, goal] = 0.001  # surviving?
            self.V_r[state] = -1e-3  # why small negative? # add goal here as well?
            return

        actions = self.env.actions()
        distributions = {a: self.env.step(state, a) for a in actions}

        for action in actions:
            for next_state in distributions[action][0]:
                self._rec_compute_policy(next_state)

        # (4*) Q_r
        for action in actions:
            self.Q_r[state, action] = 0.0
            for next_state, prob in zip(*distributions[action]):
                self.Q_r[state, action] += (
                    prob * self.params.gamma_r * self.V_r[next_state]
                )

        # (5) robot policy
        for action in actions:
            self.robot_policy[state, action] = (-self.Q_r[state, action]) ** (
                -self.params.beta_r
            )
        total = sum(self.robot_policy[state, a] for a in actions)
        for action in actions:
            self.robot_policy[state, action] /= total

        # (6*) V_h
        n_humans = self.env.num_humans(state)
        for h_idx in range(n_humans):
            for goal in self.env.goals(h_idx):
                acc = 0.0
                for action in actions:
                    for next_state, prob in zip(*distributions[action]):
                        weight = prob * self.robot_policy[state, action]
                        reward = (
                            1 if self.env.human_at_goal(next_state, h_idx, goal) else 0
                        )
                        acc += weight * (
                            reward
                            + self.params.gamma_h * self.V_h[h_idx, next_state, goal]
                        )
                self.V_h[h_idx, state, goal] = acc

        # (7) X_h
        for h_idx in range(n_humans):
            x = 0.0
            for goal in self.env.goals(h_idx):
                x += self.V_h[h_idx, state, goal] ** self.params.zeta
            self.X_h[h_idx, state] = x

        # U_r
        u = 0.0
        for h_idx in range(n_humans):
            u += self.X_h[h_idx, state] ** (-self.params.xi)
        self.U_r[state] = -(u**self.params.eta)

        # V_r
        v = self.U_r[state]
        for action in actions:
            v += self.robot_policy[state, action] * self.Q_r[state, action]
        self.V_r[state] = v
