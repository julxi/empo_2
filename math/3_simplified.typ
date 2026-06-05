#set page(height: auto, width: 21cm, margin: 2cm)
#import "util.typ": *
#show: setup

= Simplified case: greedy robot, goal independent environment

Key simplifications:
- $T(s,a_r,g)$ is goal independent, so it's just $T(s,a_r)$
- robot policy deterministic (greedy), so it's just an action-selection function $π(s)$

This is very similar to #link("2_deterministic.typ")[the deterministic case], but the environment dynamic may be stochastic.


Here are the simplified equations:

$ Q_r (s, a_r) <- bb(E)_(s' tilde.op T(s, a_r)) γ_r V_r (s') $

$ π_r (s) <- op("arg max", limits: #true)_(a_r) Q_r (s, a_r) $

$ V_h (s, g_h) <- cases(1 &"if" s in g_h, bb(E)_(s' tilde.op T(s, π_r (s))) γ_h V_h (s', g_h) &"else") $

$ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^ζ $

$ U_r (s) <- - (sum_h X_h (s)^(-ξ))^η $

$ V_r (s) <- U_r (s) + Q_r (s, π_r (s)) $


== Scaling humans

How can we collapse multiple humans with isomorphic goals into a single representative?

Let's say we have humans $cal(M) subset.eq cal(H)$ with the same goals, i.e., $cal(G)_h = cal(G)_(h')$ for $h, h' in cal(M)$.
Then for $h, h' in cal(M)$ we have $X_h = X_(h')$. Let $h^* in cal(M)$ be a representative and then we can write

$ U_r (s) <- - (abs(cal(M)) dot.c X_(h^*) (s)^(-ξ) + sum_(h in.not cal(M)) X_h (s)^(-ξ))^η $

We would get the same effect by retaining only the representative $h^*$ and replacing the goal-fulfilment value at goal states with $abs(cal(M))^(-1/(ζ ξ))$:

$ V_h (s, g_h) = cases(abs(cal(M))^(-1/(ζ ξ)) "if" s in g_h, bb(E)_(s' tilde.op T(s, π_r (s))) γ_h V_h (s', g_h) "else") $

This might be surprising at first: scaling the goal-fulfilment value down corresponds to scaling the magnitude of $U_r$ up.


== Explicit Solutions

If the environment is acyclic we can use backwards induction. We can just recursively evaluate (1) - (6) in their order.
For that we need access to $T(s, a_r)$.