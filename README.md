# Empo

## Theory

### Phase 2 equations

The phase 2 equations are designed to find a policy for the robot $r$, based on a given world model that contains:
- the set of all humans $\mathcal{H}$ (assumed to be immutable)
- a non-informative prior over the set of possible goals $\mathcal{G}_h$ for each human, where each $g_h \in \mathcal{G}_h$ is a set of states.
- behaviour $\pi_h(s,g_h)$ of human $h$ if they pursued goal $g_h$ (this might be counterfactual)
- the environment dynamics $T(s,a)$ where $a = (a_r, a_\mathcal{H})$

> Note: The theory does not specify how $g_h \sim \mathcal{G}_h$ has to be chosen; this is part of future work. The idea, however, is that the robot doesn't know and doesn't try to guess the humans' concrete goals but has a general idea of all possible goals.

In addition, we need to fix the normative parameters $\gamma_r, \beta_r, \gamma_h, \zeta, \xi, \eta$ that shape the robot policy.


**Empowerment state-action-value**

$$
Q_r(s,a_r) \gets \mathbb{E}_g \mathbb{E}_{a_\mathcal{H} \sim \pi_\mathcal{H}(s,g)} \mathbb{E}_{s' \sim T(s,a)} \gamma_r V_r(s')
\tag{4}
$$

**Power-law policy**
$$
\pi_r(s)(a_r) \propto (-Q_r(s,a_r))^{-\beta_r}
\tag{5}
$$

**Goal-fulfilment value**
$$
\begin{aligned}
V_h(s,g_h) \gets& U_h(s,g_h) + (1 - U_h(s,g_h)) \\
&\cdot \mathbb{E}_{a_r \sim \pi_r(s)} \mathbb{E}_{g_{-h}} \mathbb{E}_{a_\mathcal{H} \sim \pi_\mathcal{H}(s,g)} \mathbb{E}_{s' \sim T(s,a)} \gamma_h V_h(s', g_h)
\end{aligned}
\tag{6}
$$
$U_h(s,g_h) = [s \in g_h]$ is the indicator function

**Aggregation of human power**
$$
X_h(s) \gets \sum_{g_h \in \mathcal{G}_h} V_h(s,g_h)^\zeta
\tag{7}
$$

**Fair distribution of human power**
$$
U_r(s) \gets - (\sum_h X_h(s)^{-\xi})^\eta
\tag{8}
$$
**Empowerment state-value**
$$
V_r(s) \gets U_r(s) + \mathbb{E}_{a_r \sim \pi_r(s)} Q_r(s,a_r)
\tag{9}
$$

Comments:
- $g$ denotes the joint goal tuple $(g_h)_{h \in \mathcal{H}}$, and $g_{-h}$ denotes the goals of all humans other than $h$.
- If $g_h$ only contains mutually unreachable states – i.e., there is no trajectory containing distinct $s,s' \in g_h$ – and we further assume $\gamma_h = 1$, then $V_h(s,g_h)$ is the probability that $g_h$ gets fulfilled.
- To avoid problems calculating $U_r$ we have to make sure that the set of possible goals is so wide that in every state each human has at least one goal fulfilled. If episodes are finite, it's enough to restrict this to terminal states.
- $V_h \geq 0$, $X_h > 0$, $U_r < 0$, $V_r < 0$, $Q_r < 0$ 
- $V_r$, $Q_r$ are "better" the closer they are to zero.
- For $\beta_r = \infty$ we recover greedy action selection:
$$\pi_r(s) = \arg\max_{a_r} Q_r(s,a_r)$$

### Simplified case: greedy robot, deterministic environment

- just robot, no human agency
- greedy robot: deterministic policy $\pi_r(s)$ (easier to understand generally)
- deterministic environment: transition function $T(s,a)$
- ⇒ everything is deterministic; the equations simplify

Here are the simplified equations:

$$Q_r(s,a_r) \gets \gamma_r V_r(T(s,a_r))$$
$$\pi_r(s) \gets \arg\max_{a_r} Q_r(s,a_r)$$
$$V_h(s, g_h) \gets U_h(s,g_h) + \gamma_h\big(1- U_h(s,g_h)\big) V_h\big(T(s,\pi_r(s)),g_h\big)$$
$$X_h(s) \gets \sum_{g_h \in \mathcal{G}_h} V_h(s,g_h)^\zeta$$
$$U_r(s) \gets -\Big( \sum_h X_h(s)^{-\xi} \Big)^\eta$$
$$V_r(s) \gets U_r(s) + Q_r(s,\pi_r(s))$$

#### RL-like formulation

We can rearrange the equations into two parts – a utility component and an RL component. This formulation is less convenient for implementation but looks nice theoretically.

**Utility (unchanged)**:
$$V_h(s, g_h) \gets U_h(s,g_h) + \gamma_h\big(1- U_h(s,g_h)\big) V_h\big(T(s,\pi_r(s)),g_h\big)$$
$$X_h(s) \gets \sum_{g_h \in \mathcal{G}_h} V_h(s,g_h)^\zeta$$
$$U_r(s) \gets -\Big( \sum_h X_h(s)^{-\xi} \Big)^\eta$$
**RL**:
$$Q^*_r(s,a_r) \gets U_r(s) + \gamma_r V^*_r(T(s,a_r))$$
$$\pi_r(s) \gets \arg\max_{a_r} Q^*_r(s,a_r)$$
$$V^*_r(s) \gets Q^*_r(s,\pi_r(s))$$

(The starred versions are related to the originals by $V^*_r = V_r$ and $Q^*_r = Q_r + U_r$; the rearrangement is purely to emphasise the RL character.)

#### Compounding $U_r$ and pressure for shorter episodes

Assume that $U_h$ is zero except for terminal states, and additionally that $\gamma_h = \gamma_r = 1$.

Consider a solution and its trajectory of the deterministic policy $\pi_r$ from the starting state $s_0$.
$V_h(\cdot, g_h)$ is constant on that trajectory since $V_h(s_i,g_h) = V_h(s_{i+1},g_h)$ for all non-terminal states. Then $X_h$ and $U_r$ are also constant on the trajectory.
Note that we can write 
$$
V_r(s) \gets U_r(s) + \gamma_r V_r\big(T(s,\pi_r(s))\big)
$$
Let $N$ denote the trajectory length and $s_N$ its terminal state. Then $V_r(s_i) = U_r(s_i) + V_r(s_{i+1})$, and unrolling gives $V_r(s_i) = (N - i + 1) \cdot U_r(s_N)$.

Note that $U_r < 0$ and values closer to zero are better. So a policy that creates shorter trajectories with same terminal $U_r(s_N)$ performs better. If all episodes have equal length then relative performance between policies only depends on $U_r(s_N)$.

#### Scaling humans

How can we scale $U_h$ to simulate multiple humans with isomorphic goals?

> The text below is not quite right anymore due to changes in the formulas

Let's say we have humans $\mathcal{M}\subseteq \mathcal{H}$ with the same goals, i.e., $\mathcal{G}_h = \mathcal{G}_{h'}$ for $h,h' \in \mathcal{M}$.
Then for $h, h' \in \mathcal{M}$ we have $X_{h} = X_{h'}$. Let $h^* \in \mathcal{M}$ be a representative and then we can write
$$
U_r(s) \gets -\Big( |\mathcal{M}| \cdot X_{h^*}(s)^{-\xi} + \sum_{h \not\in \mathcal{M}} X_h(s)^{-\xi}\Big)^\eta
$$

We would get the same effect by only retaining the representative $h^*$ and scaling their $U_{h^*}$ by $|\mathcal{M}|^{-\frac{1}{\zeta \xi}}$. This might be surprising at first: scaling $U_{h^*}$ down corresponds to scaling the magnitude of $U_r$ up.

#### Backwards induction

If the environment is acyclic we can use backwards induction.

There is another situation where we might be able to use some form of backwards induction even in a cyclic environment, but this is less relevant. It is included here for completeness.
We need
- all goals $g_h$ consist of terminal states
- $\gamma_r = \gamma_h = 1$

In this case $V_r(s)$ only depends on the terminal state reached by the policy and the length of the trajectory created by the policy. So for a backwards induction we would need to take length into account.
