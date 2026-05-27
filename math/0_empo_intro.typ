#import "util.typ": *
#show: setup

= Introduction to the Empo Equations

The Empo Equations are designed to find a policy for the robot $r$, based on a given world model that contains:

- the set of all humans $cal(H)$ (assumed to be immutable)
- a non-informative prior over the set of possible goals $cal(G)_h$ for each human, where each $g_h in cal(G)_h$ is a set of states.
- behaviour $π_h (s, g_h)$ of human $h$ if they pursued goal $g_h$ (hypothetical behaviour of human if they had this goal)
- the environment dynamics $T(s, a)$ where $a = (a_r, a_(cal(H)))$

#note[The theory does not specify how $g_h tilde.op cal(G)_h$ has to be chosen; this is part of future work. The idea, however, is that the robot doesn't know and doesn't try to guess the humans' concrete goals but has a general idea of all possible goals.]

In addition, we need to fix the normative parameters $γ_r, β_r, γ_h, ζ, ξ, η$ that shape the robot policy.

Here are the Empo equations:

#titled("Empowerment state-action-value")[
  $ Q_r (s, a_r) <- bb(E)_g space bb(E)_(a_(cal(H)) tilde.op π_(cal(H)) (s, g)) space bb(E)_(s' tilde.op T(s, a)) space γ_r V_r (s') $
]

#titled("Power-law policy")[
  $ π_r (s)(a_r) prop (-Q_r (s, a_r))^(-β_r) $
]

#titled("Goal-fulfilment value")[
  $ V_h (s, g_h) &<- cases(1 &"if" s in g_h,
     bb(E)_(a_r tilde.op π_r (s)) bb(E)_(g_(-h)) bb(E)_(a_(cal(H)) tilde.op π_(cal(H)) (s, g)) bb(E)_(s' tilde.op T(s, a)) γ_h V_h (s', g_h) &"else") $
]

#titled("Aggregation of human power")[
  $ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^ζ $
]

#titled("Fair distribution of human power")[
  $ U_r (s) <- - (sum_h X_h (s)^(-ξ))^η $
]

#titled("Empowerment state-value")[
  $ V_r (s) <- U_r (s) + bb(E)_(a_r tilde.op π_r (s)) Q_r (s, a_r) $
]

Comments:

1. $g$ denotes the joint goal tuple $(g_h)_(h in cal(H))$, and $g_(-h)$ denotes the goals of all humans other than $h$.
2. If $g_h$ only contains mutually unreachable states -- i.e., there is no trajectory containing distinct $s, s' in g_h$ -- and we further assume $γ_h = 1$, then $V_h (s, g_h)$ is the probability that $g_h$ gets fulfilled.
3. To avoid problems calculating $U_r$ we have to make sure that the set of possible goals is so wide that in every state each human has at least one goal fulfilled. If episodes are finite, it's enough to restrict this to terminal states.
4. $V_h >= 0$, $X_h > 0$, $U_r < 0$, $V_r < 0$, $Q_r < 0$
5. $V_r$, $Q_r$ are "better" the closer they are to zero.
6. For $β_r = oo$ we recover greedy action selection:

#unnumbered[$ π_r (s) = op("arg max", limits: #true)_(a_r) Q_r (s, a_r) $]

== Humans as part of the environment

As far as the math is concerned humans could be considered to be part of the environment. Since the human policy depends on the goal, this makes the environment goal dependent $T(s,a_r,g)$. For example (1) reads as 

#unnumbered[$ Q_r (s, a_r) <- bb(E)_g bb(E)_(s' tilde.op T(s, a_r, g)) γ_r V_r (s') $]