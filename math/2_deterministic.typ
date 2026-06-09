#set page(height: auto, width: 21cm, margin: 2cm)
#import "util.typ": *
#show: setup

#set heading(numbering: "1.")

= Simplified case: greedy roboti in deterministic environment

Make everything deterministic to simplify the equations:
- environment goal indepentend and deterministic, so we have a transition _function_ $T(s,a)$
- robot is greedy, so we have a policy selection function $π_r (s)$



Here are the simplified equations:

$ Q_r (s, a_r) <- γ_r V_r (T(s, a_r)) $ <eq-Qr>

$ π_r (s) <- op("arg max", limits: #true)_(a_r) Q_r (s, a_r) $ <eq-pir>

$ V_h (s, g_h) <- cases(1 &"if" s in g_h, γ_h V_h (T(s, π_r (s)), g_h) &"else") $ <eq-Vh>

$ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^ζ $ <eq-Xh>

$ U_r (s) <- - (sum_h X_h (s)^(-ξ))^η $ <eq-Ur>

$ V_r (s) <- U_r (s) + Q_r (s, π_r (s)) $ <eq-Vr>

== RL-like formulation

We can rearrange the equations into two parts -- a utility component and an RL component.
This rearrangement highlights the similarity to reinforcement learning.
The 3 RL equations look algebraicaly exactly like the Bellmann equations,
except of course that the utility function $U_r$ in endogenous, i.e., $U_r$ depends on $π_r$.

*Utility* (unchanged):

#unnumbered($ V_h (s, g_h) <- cases(1 &"if" s in g_h, γ_h V_h (T(s, π_r (s)), g_h) &"else") $)
#unnumbered($ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^ζ $)
#unnumbered($ U_r (s) <- - (sum_h X_h (s)^(-ξ))^η $)

*RL*:

#unnumbered($ Q^*_r (s, a_r) <- U_r (s) + γ_r V^*_r (T(s, a_r)) $)
#unnumbered($ π_r (s) <- op("arg max", limits: #true)_(a_r) Q^*_r (s, a_r) $)
#unnumbered($ V^*_r (s) <- Q^*_r (s, π_r (s)) $)

The starred versions are related to the originals by $V^*_r = V_r$ and $Q^*_r = Q_r + U_r$; the rearrangement is purely to emphasise the RL character.

== Only Terminal Goals <sec-only-terminal-goals>

We call a goal $g_h$ a terminal goal is if only contains terminal states.
In this section we show that if all goals are terminal and there's no discounting the robots $U_r$ only depends on the reached terminal goal and $V_r$ factorises in a terminal goal dependent component and a length component.


So, assume that every $g_h$ contains only terminal states, and additionally that $γ_h = γ_r = 1$.
Consider a solution and the trajectory $(s_1,...s_N)$ of its deterministic policy $π_r$ from the starting state $s_0$.
$V_h (dot, g_h)$ is constant on that trajectory since $V_h (s_i, g_h) = V_h (s_(i+1), g_h)$ for non-terminal states in our case due to  @eq-Vh. From this we get that also $X_h$ and $U_r$ are constant on the trajectory.
From @eq-Qr and @eq-Vr we get (generally)

$ V_r (s) <- U_r (s) + γ_r V_r (T(s, π_r (s))) $ <eq-Vr-recursive>

In our case this becomes

$ V_r (s_i) = (N - i + 1) dot.c U_r (s_N). $ <eq-Vr-unrolled>

Note that $U_r < 0$ and values closer to zero are better. So a policy that creates shorter trajectories with same terminal $U_r (s_N)$ performs better. If all episodes have equal length then relative performance between policies only depends on $U_r (s_N)$.


== Computational Solutions

If the environment is acyclic we can use backwards induction. We can just recursively evaluate @eq-Qr -- @eq-Vr in their order.

We can solve some cyclic environments explicitly if

- all goals $g_h$ consist of terminal states
- $γ_r = γ_h = 1$

In this case $V_r (s)$ only depends on the terminal state reached by the policy and the length of the trajectory it creates, so a backwards induction would need to take length into account.


== Examples with Analytical Solutions

=== Trolley Problem

The trolley problem captures a classic tension: given mutually exclusive
goals, whose goals should be followed, and
how is the trade-off shifted by a socially preference for certain behaviours?

#heading(numbering: none, level:4)[Setup]

- Robot has two actions $a ∈ {0,1}$ (inaction and action)
- $n_0$ -- number of people affacted if passive
- $n_1$ -- number of poeple affected if active
- $m_s$ -- survival goals per human
- $m_0$ -- passivity goals per human

Total humans $H = n_0 + n_1$. We fix the Empo parameters
$ γ_r = γ_h = 1, quad ζ = 2, quad ξ = 1, quad η = 1. $ <eq-trolley-params>

All trajectories have equal length and end in one of two terminal states,
depending on $a$:
- $a = 0$ -- inaction: train kills the $n_0$ humans,
- $a = 1$ -- action: train kills the $n_1$ humans.

Each human $h$ has three types of terminal goals:
+ a constant baseline $g equiv 1$ (always satisfied),
+ $m_s$ copies of the *survival goal* $g^"surv"_h (s) = bb(1)[h "alive in" s]$,
+ $m_0$ copies of the *switch-unpressed goal* $g^0 (s) = bb(1)["robot stayed passive"]$.

#heading(numbering: none, level:4)[Solution]

To find a solution for the empo equations, i.e., figuring out which actions the empo-bot will take, we simply have to minimise $U_r$ at the terminal states (see @sec-only-terminal-goals).

Let $S_h in {0, 1}$ be the survival indicator of human $h$ at $s_N$. Then the $V_h$ at $s_N$ are
$ V_h (s_N, g) = 1, quad V_h (s_N, g^"surv"_h) = S_h, quad V_h (s_N, g^0) = (1-a). $
Applying @eq-Xh with $ζ = 2$ and using $S_h^2 = S_h$, $(1-a)^2 = a$ since both are binary:

$ X_h (s_N) = 1 + m_s S_h + m_0 (1-a). $ <eq-trolley-X>

From this we can get $U_r$ (for $ξ = η = 1$):

$ U_r (s_N) = - sum_(h = 1)^(H) (X_h (s_N))^(-1) = - sum_h 1/(1 + m_s S_h + m_0 (1-a)). $ <eq-trolley-U>

Splitting the sum by which group of humans got killed we get:

- switch unpressed ($a = 0$):
  The $n_0$ humans die; the $n_1$ humans survive:
  $ U_0 = U_r (s_N | a = 0) = - n_0/(1 + m_0) - n_1/(1 + m_s + m_0) . $ <eq-trolley-Uu>
- switch pressed ($a = 1$).
  The $n_1$ humans die; the $n_0$ humans survive:
  $ U_1 = U_r (s_N | a = 1) = - n_0/(1+m_s) - n_1. $ <eq-trolley-Up>

Now the robot will just take whichever action results in a higher $U_r$.


#heading(numbering: none, level:4)[Examples]

We consider two versions of the trolley problem:

- version 1: $n_p = 1, n_u = 3, m_s = 1, m_w = 0$.
- version 2: $n_p = 1, n_u = 3, m_s = 1, m_w = 1$.

(I think of them as two different types, the first one is maybe best expressed that there is no human preference for the robot action, the second one, there is a preference)

#unnumbered($ U_0^"version 1" = -1/2 - 3 = -3.5, quad U_1^"version 1" = -1 - 3/2 = -2.5. $)

The robot takes an action ($-2.5 > -3.5$).



#unnumbered($ U_0^"version 2" = -1/3 - 3/2 = -11/6 approx -1.833, quad U_1^"version 2" = -1 - 3/2 = -2.5. $)

The robot takes no action

#heading(numbering: none, level:4)[Decision Boundaries]

We derived that the robot takes the action if $U_0 < U_1$.

$  - n_0/(1 + m_0) - n_1/(1 + m_s + m_0) < - n_0/(1+m_s) - n_1 $

Rearranging gives

$   n_0 ((m_s - m_0)/((1 + m_0)(1+m_s)))  >  n_1 ((m_s + m_0)/(1 + m_s + m_0)) $ <eq-action-condition>

We can solve this easily for a condition on $n_1 / n_0$:


$   n_1 / n_0 <  ((m_s - m_0)(1 + m_s + m_0))/((m_s + m_0)(1 + m_0)(1+m_s)) $

This can be read as "given $m_s$ and $m_0$ how has the ration $n_1 / n_0$ be so that the robot takes action". We can read of some siple things:
- if $m_0 = 0$ then $n_1 / n_0 < 1$, i.e., robot takes action if $n_1 < n_0$. Which makes sense
- if $m_0 > m_s$ this would require $n_1 / n_0 < 0$, which is not possible so robot doesn't take action.

We can also consider @eq-action-condition for the limit case $m_s -> oo$, which means there is some value for $m_s$ that lets the robot take action. The condition is
$ n_0 1/(1+m_0) > n_1 $
or equivalently
$ m_0 < n_0/n_1 - 1 $