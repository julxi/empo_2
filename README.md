# Empo

## Theory

### Main Equations

The equations for phase 2 are supposed to find a policy for the robot $r$, based on a given world model that contains:
- the set of all humans $\mathcal{H}$ (assumed to be immutable)
- a non-informative prior of the distribution of the possible goals $\mathcal{G}_h$ for each human, where each $g_h \in \mathcal{G}_h$ is a set of states.
- behaviour $\pi_h(s,g_h)$ of human $h$ if they pursued goal $g_h$.
- the environment dynamics $T(s,a)$ where $a = (a_r, a_\mathcal{H})$

Note: The theory is vague in how $g_h \sim \mathcal{G}_h$ has to be chosen. The idea is that the robot doesn't know and doesn't try to guess the humans' concrete goals but has a feasible idea of all possible goals. In vagueness lies safety?

Besides that we have got some hyperparameters: $\gamma_r, \beta_r, \gamma_h, \zeta, \xi, \eta$, and $U(s,g_h) = [s \in g_h]$ is the indicator function.

So here are the equations for phase 2:

- Empowerment state-action-value:
$$
Q_r(s,a_r) \gets \mathbb{E}_g \mathbb{E}_{a_\mathcal{H} \sim \pi_\mathcal{H}(s,g)} \mathbb{E}_{s' \sim T(s,a)} \gamma_r V_r(s')
\tag{4}
$$
- Power-law policy:
$$
\pi_r(s)(a_r) \propto (-Q_r(s,a_r))^{-\beta_r}
\tag{5}
$$
- Goal-fulfilment value:
$$
V_h(s,g_h) \gets \mathbb{E}_{a_r \sim \pi_r(s)} \mathbb{E}_{g_{-h}} \mathbb{E}_{a_\mathcal{H} \sim \pi_\mathcal{H}(s,g)} \mathbb{E}_{s' \sim T(s,a)} [U(s',g_h) + \gamma_h V_h(s', g_h)]
\tag{6}
$$
- Aggregation of human power:
$$
X_h(s) \gets \sum_{g_h \in \mathcal{G}_h} V_h(s,g_h)^\zeta
\tag{7}
$$
- Fair distribution of human power:
$$
U_r(s) \gets - (\sum_h X_h(s)^{-\xi})^\eta
\tag{8}
$$
- Empowerment state-value:
$$
V_r(s) \gets U_r(s) + \mathbb{E}_{a_r \sim \pi_r(s)} Q_r(s,a_r)
\tag{9}
$$

Comments:
- $g$ denotes the joint goal vector $(g_h)_{h \in \mathcal{H}}$, and $g_{-h}$ denotes the goals of all humans other than $h$.
- If $g_h$ only contains mutually unreachable goals – i.e., there is no trajectory containing distinct $s,s' \in g_h$ – and we further assume $\gamma_h = 1$, then $V_h(s,g_h)$ is the probability that $g_h$ gets fulfilled.
- To avoid problems calculating $U_r$ we have to make sure that the set of possible goals is so wide that in every state each human has at least one attainable goal. Then in the sum for $X_h(s)$ one of the $V_h(s,g_h) > 0$ and $X_h(s) > 0$.
- $V_h \geq 0$, $X_h > 0$, $U_r < 0$, $V_r < 0$, $Q_r < 0$. And $V_r$ and $Q_r$ are "better" the closer they are to zero.
- For $\beta_r = \infty$ we recover greedy action selection:
$$\pi_r(s) = \arg\max_{a_r} Q_r(s,a_r)$$

### Simplified case: Just greedy Robot, One goal, Deterministic Environment

- Just robot, no human agents. Simplifies MARL to RL.
- Greedy robot: deterministic policy $\pi_r(s)$
- one goal, i.e., only one human (without agency) and with one goal $g$. Let $U(s) = [s \in g]$
- deterministic environment: transition function $t(s,a)$
- ⇒ everything is deterministic; the equations simplify

Here are the simplified equations:

$$
Q_r(s,a_r) \gets \gamma_r V_r(t(s,a_r))
$$
$$
\pi_r(s) = \arg\max_{a_r} Q_r(s,a_r)
$$
$$
V_h(s) \gets U(t(s,\pi_r(s))) + \gamma_h V_h(t(s,\pi_r(s)))
$$
$$
U_r(s) \gets - V_h(s)^{-\xi_s}
$$
$$
V_r(s) \gets U_r(s) + \max_{a_r} Q_r(s,a_r)
$$


Note: $\xi_s = \zeta\,\xi\,\eta$, the product of $\zeta$, $\xi$, $\eta$ from the original equations.
