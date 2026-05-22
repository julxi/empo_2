#import "util.typ": *
#show: setup

= Simplified case: greedy robot, deterministic environment

make everything deterministic to simplify the equations:
- just robot, no human agency
- greedy robot: deterministic policy $π_r (s)$ (also easier to reason about in general)
- deterministic environment: transition function $T(s, a)$


Here are the simplified equations:

$ Q_r (s, a_r) <- gamma_r V_r (T(s, a_r)) $
$ pi_r (s) <- op("arg max", limits: #true)_(a_r) Q_r (s, a_r) $
$ V_h (s, g_h) <- U_h (s, g_h) + gamma_h (1 - U_h (s, g_h)) V_h (T(s, pi_r (s)), g_h) $
$ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^zeta $
$ U_r (s) <- - (sum_h X_h (s)^(-xi))^eta $
$ V_r (s) <- U_r (s) + Q_r (s, pi_r (s)) $

== RL-like formulation

We can rearrange the equations into two parts -- a utility component and an RL component. This formulation is less convenient for implementation but looks nice theoretically.

*Utility* (unchanged):

$ V_h (s, g_h) <- U_h (s, g_h) + gamma_h (1 - U_h (s, g_h)) V_h (T(s, pi_r (s)), g_h) $
$ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^zeta $
$ U_r (s) <- - (sum_h X_h (s)^(-xi))^eta $

*RL*:

$ Q^*_r (s, a_r) <- U_r (s) + gamma_r V^*_r (T(s, a_r)) $
$ pi_r (s) <- op("arg max", limits: #true)_(a_r) Q^*_r (s, a_r) $
$ V^*_r (s) <- Q^*_r (s, pi_r (s)) $

(The starred versions are related to the originals by $V^*_r = V_r$ and $Q^*_r = Q_r + U_r$; the rearrangement is purely to emphasise the RL character.)

== Compounding $U_r$ and pressure for shorter episodes

Assume that $U_h$ is zero except for terminal states, and additionally that $gamma_h = gamma_r = 1$.

Consider a solution and its trajectory of the deterministic policy $pi_r$ from the starting state $s_0$.
$V_h (dot, g_h)$ is constant on that trajectory since $V_h (s_i, g_h) = V_h (s_(i+1), g_h)$ for all non-terminal states. Then $X_h$ and $U_r$ are also constant on the trajectory.
Note that we can write

$ V_r (s) <- U_r (s) + gamma_r V_r (T(s, pi_r (s))) $

Let $N$ denote the trajectory length and $s_N$ its terminal state. Then $V_r (s_i) = U_r (s_i) + V_r (s_(i+1))$, and unrolling gives $V_r (s_i) = (N - i + 1) dot.c U_r (s_N)$.

Note that $U_r < 0$ and values closer to zero are better. So a policy that creates shorter trajectories with same terminal $U_r (s_N)$ performs better. If all episodes have equal length then relative performance between policies only depends on $U_r (s_N)$.

== Scaling humans

How can we scale $U_h$ to simulate multiple humans with isomorphic goals?

#note[The text below is not quite right anymore due to changes in the formulas. It does need some updating.]

Let's say we have humans $cal(M) subset.eq cal(H)$ with the same goals, i.e., $cal(G)_h = cal(G)_(h')$ for $h, h' in cal(M)$.
Then for $h, h' in cal(M)$ we have $X_h = X_(h')$. Let $h^* in cal(M)$ be a representative and then we can write

$ U_r (s) <- - (abs(cal(M)) dot.c X_(h^*) (s)^(-xi) + sum_(h in.not cal(M)) X_h (s)^(-xi))^eta $

We would get the same effect by only retaining the representative $h^*$ and scaling their $U_(h^*)$ by $abs(cal(M))^(-1/(zeta xi))$. This might be surprising at first: scaling $U_(h^*)$ down corresponds to scaling the magnitude of $U_r$ up.

== Backwards induction

If the environment is acyclic we can use backwards induction.

There is another situation where we might be able to use some form of backwards induction even in a cyclic environment, but this is less relevant. It is included here for completeness.
We need

- all goals $g_h$ consist of terminal states
- $gamma_r = gamma_h = 1$

In this case $V_r (s)$ only depends on the terminal state reached by the policy and the length of the trajectory created by the policy. So for a backwards induction we would need to take length into account.
