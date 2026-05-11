# Empo

## Theory

### Main equations

The equations for phase 2 are supposed to find a policy for the robot $r$, based on a given world model that contains:
- the set of all humans $\mathcal{H}$ (assumed to be immutable)
- a non-informative prior of the distribution of the possible goals $\mathcal{G}_h$ for each human, where each $g_h \in \mathcal{G}_h$ is a set of states.
- behaviour $\pi_h(s,g_h)$ of human $h$ if they pursued goal $g_h$.
- the environment dynamics $T(s,a)$ where $a = (a_r, a_\mathcal{H})$

Note: The theory is vague in how $g_h \sim \mathcal{G}_h$ has to be chosen. The idea is that the robot doesn't know and doesn't try to guess the humans' concrete goals but has a feasible idea of all possible goals. In vagueness lies safety?

Besides that, we have the hyperparameters $\gamma_r, \beta_r, \gamma_h, \zeta, \xi, \eta$, and $U_h(s,g_h) = [s \in g_h]$ is the indicator function.

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
V_h(s,g_h) \gets \mathbb{E}_{a_r \sim \pi_r(s)} \mathbb{E}_{g_{-h}} \mathbb{E}_{a_\mathcal{H} \sim \pi_\mathcal{H}(s,g)} \mathbb{E}_{s' \sim T(s,a)} [U_h(s',g_h) + \gamma_h V_h(s', g_h)]
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
- If $g_h$ only contains mutually unreachable states – i.e., there is no trajectory containing distinct $s,s' \in g_h$ – and we further assume $\gamma_h = 1$, then $V_h(s,g_h)$ is the probability that $g_h$ gets fulfilled.
- To avoid problems calculating $U_r$ we have to make sure that the set of possible goals is so wide that in every state each human has at least one attainable goal. Then in the sum for $X_h(s)$, at least one $V_h(s,g_h)$ is positive, so $X_h(s) > 0$.
- $V_h \geq 0$, $X_h > 0$, $U_r < 0$, $V_r < 0$, $Q_r < 0$ — and $V_r$, $Q_r$ are "better" the closer they are to zero.
- For $\beta_r = \infty$ we recover greedy action selection:
$$\pi_r(s) = \arg\max_{a_r} Q_r(s,a_r)$$

### Simplified case: greedy robot, deterministic environment

- Just robot, no human agency. Simplifies MARL to RL.
- Greedy robot: deterministic policy $\pi_r(s)$
- deterministic environment: transition function $t(s,a)$
- ⇒ everything is deterministic; the equations simplify

Here are the simplified equations:

$$
Q_r(s,a_r) \gets \gamma_r V_r(t(s,a_r))
$$
$$
\pi_r(s) \gets \arg\max_{a_r} Q_r(s,a_r)
$$
$$
V_h(s, g_h) \gets U_h\big(t(s,\pi_r(s)),g_h\big) + \gamma_h V_h\big(t(s,\pi_r(s)),g_h\big)
$$
$$
X_h(s) \gets \sum_{g_h \in \mathcal{G}_h} V_h(s,g_h)^\zeta
$$
$$
U_r(s) \gets -\Big( \sum_h X_h(s)^{-\xi} \Big)^\eta
$$
$$
V_r(s) \gets U_r(s) + \max_{a_r} Q_r(s,a_r)
$$

#### Compounding $U_r$ and pressure for shorter episodes

Assume that $U_h$ is zero except for terminal states, and additionally that $\gamma_h = \gamma_r = 1$.

Consider a solution and its trajectory of the deterministic policy from the starting state $s_0$.
$V_h(s,g_h) = V_h(s_{T-1},g_h)$ for all $s$, where $s_{T-1}$ is the penultimate state (note that by the definition of $V_h$, $U_h$ gives a signal one step before the goal is reached). Since $V_h$ is constant, $X_h$ is also constant and then $U_r$ is also constant.
So $V_r(s_0) = U_r(s_{T-1}) \cdot (T-1)$.

Note that $U_r < 0$ and values closer to zero are better. Thus the policy prefers shorter episodes even at the expense of ultimate goal fulfilment.

#### Scaling humans

How can we scale $U_h$ to simulate multiple humans $m$ with isomorphic goals?

We have $h_1,\dots,h_m$ with $V_{h_i}(s,g_{h_i}) = V_{h_j}(s,g_{h_j})$ for all $i, j$.
Then $X_{h_i} = X_{h_j}$ and 
$$
U_r(s) \gets -\Big( m \cdot X_{h_1}(s)^{-\xi} + \sum_{h \neq h_i} X_h(s)^{-\xi}\Big)^\eta
$$
This has the same effect as only having one human $h$ with the same goal structure and scaling their $U_h$ by $m^{-\frac{1}{\zeta \xi}}$. Curiously, scaling $U_h$ down corresponds to scaling the magnitude of $U_r$ up.

#### Backwards induction (maybe wrong)

If the environment is acyclic we can use backwards induction.
Another case where we can use backwards induction is when:
- all goals $g_h$ consist of terminal states
- $\gamma_h = 1$

In this case $U_r(s)$ only depends on the terminal state reached by the policy. So we can just do backwards induction on a spanning tree of the state space.
