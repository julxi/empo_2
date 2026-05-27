#set page(height: auto, width: 21cm, margin: 2cm)
#import "util.typ": *
#show: setup

= Simplified case: greedy robot, deterministic environment

Make everything deterministic to simplify the equations:
- environment dynamic:
  - deterministic
  - goal independent
  - So we have a transition function $T(s, a)$
- greedy robot: deterministic policy $π_r (s)$ (also easier to reason about in general)



Here are the simplified equations:

$ Q_r (s, a_r) <- γ_r V_r (T(s, a_r)) $ <eq-Qr>

$ π_r (s) <- op("arg max", limits: #true)_(a_r) Q_r (s, a_r) $ <eq-pir>

$ V_h (s, g_h) <- cases(1 &"if" s in g_h, γ_h V_h (T(s, π_r (s)), g_h) &"else") $ <eq-Vh>

$ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^ζ $ <eq-Xh>

$ U_r (s) <- - (sum_h X_h (s)^(-ξ))^η $ <eq-Ur>

$ V_r (s) <- U_r (s) + Q_r (s, π_r (s)) $ <eq-Vr>

== RL-like formulation

We can rearrange the equations into two parts -- a utility component and an RL component. This formulation is less convenient for implementation but looks nice theoretically.

*Utility* (unchanged):

#unnumbered($ V_h (s, g_h) <- cases(1 &"if" s in g_h, γ_h V_h (T(s, π_r (s)), g_h) &"else") $)
#unnumbered($ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^ζ $)
#unnumbered($ U_r (s) <- - (sum_h X_h (s)^(-ξ))^η $)

*RL*:

#unnumbered($ Q^*_r (s, a_r) <- U_r (s) + γ_r V^*_r (T(s, a_r)) $)
#unnumbered($ π_r (s) <- op("arg max", limits: #true)_(a_r) Q^*_r (s, a_r) $)
#unnumbered($ V^*_r (s) <- Q^*_r (s, π_r (s)) $)

The starred versions are related to the originals by $V^*_r = V_r$ and $Q^*_r = Q_r + U_r$; the rearrangement is purely to emphasise the RL character.

== Compounding $U_r$ and pressure for shorter episodes

Assume that every $g_h$ contains only terminal states, and additionally that $γ_h = γ_r = 1$.

Consider a solution and its trajectory of the deterministic policy $π_r$ from the starting state $s_0$.
$V_h (dot, g_h)$ is constant on that trajectory since $V_h (s_i, g_h) = V_h (s_(i+1), g_h)$ for all non-terminal states. Then $X_h$ and $U_r$ are also constant on the trajectory.
Note that we can write

$ V_r (s) <- U_r (s) + γ_r V_r (T(s, π_r (s))) $ <eq-Vr-recursive>

Let $N$ denote the trajectory length and $s_N$ its terminal state. Then $V_r (s_i) = U_r (s_i) + V_r (s_(i+1))$, and unrolling gives
$ V_r (s_i) = (N - i + 1) dot.c U_r (s_N). $ <eq-Vr-unrolled>

Note that $U_r < 0$ and values closer to zero are better. So a policy that creates shorter trajectories with same terminal $U_r (s_N)$ performs better. If all episodes have equal length then relative performance between policies only depends on $U_r (s_N)$.


== Precise Solution

If the environment is acyclic we can use backwards induction. We can just recursively evaluate @eq-Qr -- @eq-Vr in their order.

We can solve some cyclic environments explicitly if

- all goals $g_h$ consist of terminal states
- $γ_r = γ_h = 1$

In this case $V_r (s)$ only depends on the terminal state reached by the policy and the length of the trajectory it creates, so a backwards induction would need to take length into account.


== Analytic Solutions

=== Trolley Problem

The trolley problem captures a classic ethical dilemma: given mutually exclusive
goals (saving different groups of humans), whose goals should be followed, and
how is the trade-off shifted by an additional preference for inactivity?

We work out the terminal utility $U_r (s_N)$ for the environment implemented in
[`experiments/trolley_problem.py`](../experiments/trolley_problem.py).

==== Setup

Let
- $n_p$ -- number of humans in the *if-pressed* column, killed when the robot presses the switch and the train deflects,
- $n_u$ -- number of humans in the *if-unpressed* column, killed when the switch is left alone and the train falls straight down,
- $m_s$ -- multiplicity of the per-human survival goal,
- $m_w$ -- multiplicity of the (shared) switch-unpressed goal.

Total humans $H = n_p + n_u$. We fix the Empo parameters
$ γ_r = γ_h = 1, quad ζ = 2, quad ξ = 1, quad η = 1. $ <eq-trolley-params>

All trajectories have equal length and end in one of two terminal states,
distinguished by the indicator $W in {0, 1}$:
- $W = 1$ -- switch *unpressed* (inaction): train falls straight, kills the $n_u$ if-unpressed humans,
- $W = 0$ -- switch *pressed* (action): train deflects, kills the $n_p$ if-pressed humans.

Each human $h$ shares the same goal list, consisting of three primitives:
+ a constant baseline $g^0 equiv 1$ (always satisfied),
+ $m_s$ copies of the *survival goal* $g^"surv"_h (s) = bb(1)[h "alive in" s]$,
+ $m_w$ copies of the *switch-unpressed goal* $g^"sw" (s) = bb(1)[s "has switch unpressed"]$.

Let $S_h in {0, 1}$ be the survival indicator of human $h$ at $s_N$. Since
$γ_h = 1$ and all goals are terminal-state goals, the per-goal values from
@eq-Vh collapse along the trajectory to their indicators at $s_N$:
$ V_h (s_N, g^0) = 1, quad V_h (s_N, g^"surv"_h) = S_h, quad V_h (s_N, g^"sw") = W. $

==== $X_h$ at the terminal

Applying @eq-Xh with $ζ = 2$ and using $S_h^2 = S_h$, $W^2 = W$ since both are binary:

$ X_h (s_N) = 1 + m_s S_h + m_w W. $ <eq-trolley-X>

==== $U_r$ at the terminal

Applying @eq-Ur with $ξ = η = 1$:

$ U_r (s_N) = - sum_(h = 1)^(H) (X_h (s_N))^(-1) = - sum_h 1/(1 + m_s S_h + m_w W). $ <eq-trolley-U>

Splitting the sum by which column the human is in:

*Case A -- switch unpressed ($W = 1$).*
The $n_u$ if-unpressed humans die ($S_h = 0$); the $n_p$ if-pressed humans survive ($S_h = 1$):
$ U_r^"unpressed" = - n_p/(1 + m_s + m_w) - n_u/(1 + m_w). $ <eq-trolley-Uu>

*Case B -- switch pressed ($W = 0$).*
The $n_p$ if-pressed humans die; the $n_u$ if-unpressed humans survive:
$ U_r^"pressed" = - n_p - n_u/(1 + m_s). $ <eq-trolley-Up>

==== Decision criterion

*Proposition.* Under the parameter choice @eq-trolley-params and with equal
trajectory length on both branches, the robot presses the switch if and only if
$ n_p + n_u/(1 + m_s) < n_p/(1 + m_s + m_w) + n_u/(1 + m_w). $ <eq-trolley-decision>

*Proof.* The two branches share the same trajectory length $N$, and $U_r$ is
constant along the trajectory (every goal is a terminal-state goal and
$γ_h = 1$), so by @eq-Vr-unrolled
#unnumbered($ V_r (s_0) = N dot.c U_r (s_N) $)
on both branches. Since $U_r < 0$, the branch with the larger (closer to zero)
$U_r (s_N)$ has the larger $V_r (s_0)$, and $π_r$ selects it via @eq-pir. The
robot therefore presses iff $U_r^"pressed" > U_r^"unpressed"$; substituting
@eq-trolley-Up and @eq-trolley-Uu and clearing the minus signs gives
@eq-trolley-decision. #h(1fr) $square$

==== Worked examples

*Script defaults: $n_p = 1, n_u = 3, m_s = 1, m_w = 0$.*

#unnumbered($ U_r^"unpressed" = -1/2 - 3 = -3.5, quad U_r^"pressed" = -1 - 3/2 = -2.5. $)

The robot presses ($-2.5 > -3.5$): one death is preferred to three.

*Switching on the inactivity preference: $n_p = 1, n_u = 3, m_s = 1, m_w = 1$.*

#unnumbered($ U_r^"unpressed" = -1/3 - 3/2 = -11/6 approx -1.833, quad U_r^"pressed" = -1 - 3/2 = -2.5. $)

Now the robot *refuses* to press: the shared preference "switch should stay unpressed" outweighs the lives saved.

==== When does survival win back? — finishing the derivation

Fix $m_w = 1, n_p = 1, n_u = 3$ as above, and ask: for which survival
multiplicity $m_s$ does the robot return to pressing? Specialising
@eq-trolley-decision gives
$ 1 + 3/(1 + m_s) < 1/(2 + m_s) + 3/2. $

Rearranging,
#unnumbered($ 3/(1 + m_s) - 1/(2 + m_s) < 1/2 quad <==> quad (5 + 2 m_s)/((1 + m_s)(2 + m_s)) < 1/2, $)

which clears to the quadratic
#unnumbered($ m_s^2 - m_s - 8 > 0 quad <==> quad m_s > (1 + sqrt(33))/2 approx 3.37. $)

So the robot presses iff $m_s >= 4$ (in integer multiplicities). A sanity
check at the boundary:

- $m_s = 3$: $quad U_r^"unpressed" = -1/5 - 3/2 = -17/10, quad U_r^"pressed" = -1 - 3/4 = -7/4$. Since $-17/10 > -7/4$, *don't press*.
- $m_s = 4$: $quad U_r^"unpressed" = -1/6 - 3/2 = -10/6, quad U_r^"pressed" = -1 - 3/5 = -8/5$. Since $-8/5 > -10/6$, *press*.

More generally, writing @eq-trolley-decision as
#unnumbered($ n_p dot (m_s + m_w)/(1 + m_s + m_w) < n_u dot (m_s - m_w)/((1 + m_s)(1 + m_w)), $)

the right-hand side is positive only when $m_s > m_w$, so the survival goal
must outweigh the inactivity goal *per copy* before any sacrifice in service
of more lives becomes possible. When $m_s = m_w$ the robot never presses,
regardless of how many lives are at stake -- a sharp threshold rather than a
trade-off.
