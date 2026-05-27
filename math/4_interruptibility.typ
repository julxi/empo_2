#set page(height: auto, width: 21cm, margin: 2cm)
#import "util.typ": *
#show: setup

= Interruptibility: decision boundaries

We work out the start-state value $V_r (s_0)$ for the environment in
[`experiments/interruptibility.py`](../experiments/interruptibility.py)
under three natural *committed* robot policies, and derive when each beats
the others. The dynamics are stochastic, so we work in the framework of
#link("3_simplified.typ")[the goal-independent stochastic case] rather than
the fully deterministic equations of #link("2_deterministic.typ")[`2_deterministic.typ`].

#note[*Committed-policy values vs. the greedy solver.* The "greedy robot" of
#link("3_simplified.typ")[`3_simplified.typ`] is $Q_r$-greedy: it selects the
action maximising $γ_r EE[V_r (s')]$ at each state. In the trolley case
($γ_h = 1$, terminal-only goals) $U_r$ is constant along every trajectory, so
$V_r$ is monotone in $U_r$ at the terminal and $Q_r$-greedy coincides with
$V_r$-greedy. Here the goals are *not* terminal-only ($g^"reach"$ activates
whenever the robot is at $G$, $g^"sw"$ whenever the switch is unpressed), so
$U_r$ varies along the trajectory and the action at $s_0$ also affects
$U_r (s_0)$ through $V_h (s_0, dot)$. The $Q_r$-greedy step does not see this
indirect effect, so the implemented solver can pick a policy with strictly
lower $V_r$ than the committed-policy optimum derived below -- see the final
section for an empirical comparison.]

== Setup

The grid (top row $y = 2$):

#unnumbered($
mat(
delim: "|",
"G", " ", "P", "R";
W, W, " ", " ";
W, W, W, "S";
)
$)

where $W$ is a wall, $G = (0, 2)$ the goal, $P = (2, 2)$ the *pause-button*,
$R = (3, 2)$ the robot's start, $S = (3, 0)$ the *switch*.

*Pause-button dynamics.* When the robot enters $P$ and the pause-button is
still active, with probability $p in [0, 1]$ a flag is set on the next state
that freezes the robot (every subsequent action is a no-op until the episode
ends at step $N$). The pause-button is deactivated once the robot enters $S$.
The only way to reach $G$ from $R$ goes through $P$ -- either risking the
pause (path of length $3$) or first walking around via $S$, pressing it, and
returning ($7$ steps, deterministic).

*Parameters.* One human, with three goal primitives (and constant baseline
$g^0 equiv 1$):
- $m_g$ copies of the *reach-goal* goal $g^"reach" (s) = bb(1)[s_"robot" = G]$,
- $m_w$ copies of the (shared) *switch-unused* goal $g^"sw" (s) = bb(1)[s_"switch" = "unpressed"]$.

Empo parameters are the defaults of `EmpoParameter`:
$ γ_r = γ_h = 1, quad ζ = 2, quad ξ = 1, quad η = 1, $ <eq-int-params>
and the episode length is $N = 10$.

*Candidate policies.* Three deterministic robot policies cover the
non-dominated behaviours:

#table(
  columns: 3,
  inset: 8pt,
  align: left,
  [*Name*], [*Trajectory*], [*Remarks*],
  [$A$ direct],
  [$R -> P -> (1, 2) -> G$],
  [stochastic fork at $P$; reaches $G$ with prob $1 - p$],
  [$B$ switch-first],
  [$R -> (3, 1) -> S -> (3, 1) -> R -> P -> (1, 2) -> G$],
  [deterministic; switch pressed at $t = 2$, $G$ reached at $t = 7$],
  [$C$ idle],
  [stay at $R$ for all $N$ steps],
  [reference; never reaches $G$, never presses switch],
)

These three are *natural* committed policies, not exhaustive: in particular
policy $A$ reaches $G$ as fast as possible (and so spends the most steps with
$X_h$ depressed by the paused-branch contribution), while $B$ presses the
switch as early as possible (and so spends the fewest steps with $g^"sw"$
satisfied). A "delay-then-switch" family that idles at $R$ for $k$ steps
before taking $B$'s route can outperform $B$ when $m_g$ is large and $m_w$ is
moderate -- the empirical section returns to this.

== $V_r$ unrolling

With $γ_r = 1$ and the convention $V_r (s_N) = 0$ used by the solver
[`stochastic_backward_induction.py`](../src/grid_world/solvers/stochastic_backward_induction.py),
the $V_r$ recursion of #link("3_simplified.typ")[`3_simplified.typ`] unrolls to
$ V_r (s_0) = sum_(t = 0)^(N - 1) EE [U_r (s_t)], $ <eq-int-Vr-unroll>
where the expectation is over the trajectory induced by the policy. So we just
need $EE[U_r (s_t)]$ at each time $t$ along each candidate trajectory. With
$H = 1$, $ξ = η = 1$ the per-state utility collapses to
$ U_r (s) = -1/(X_h (s)) = -1/(1 + m_g V_h (s, g^"reach")^2 + m_w V_h (s, g^"sw")^2). $ <eq-int-Ur>

The three recurring per-state $X_h$ levels are abbreviated
$ α := 1 + m_g (1 - p)^2 + m_w, quad
β := 1 + m_w, quad
γ := 1 + m_g + m_w, quad
δ := 1 + m_g, $ <eq-int-abbrev>
and correspond to: $α$ = at $s_0$ of policy $A$ (the pre-fork state, where the
$V_h$ for "reach" is the mixed value $1 - p$); $β$ = "switch unpressed and $G$
out of reach"; $γ$ = "switch unpressed and $G$ reachable"; $δ$ = "switch
pressed and $G$ reachable".

== Strategy $A$ -- direct path

At $s_0$, $g^"reach" (s_0) = 0$ and the next-state distribution puts mass
$1 - p$ on a future that reaches $G$ deterministically (the non-paused branch)
and mass $p$ on a future that stays at $P$ (the paused branch). With $γ_h = 1$,
the $V_h$ recursion gives
#unnumbered($ V_h (s_0, g^"reach") = (1 - p) dot 1 + p dot 0 = 1 - p, quad
V_h (s_0, g^"sw") = 1. $)

So by @eq-int-Ur,
#unnumbered($ X_h (s_0) = 1 + m_g (1 - p)^2 + m_w = α, quad EE[U_r (s_0)] = -1/α. $)

For $t >= 1$ the stochastic fork has been resolved:

- *paused branch* (prob $p$): robot frozen at $P$, never at $G$, switch
  untouched. So $V_h (g^"reach") = 0$, $V_h (g^"sw") = 1$, $X_h = β$, $U_r = -1/β$.
- *non-paused branch* (prob $1 - p$): robot reaches $G$ at $t = 3$ and stays;
  switch untouched. So $V_h (g^"reach") = 1$, $V_h (g^"sw") = 1$, $X_h = γ$,
  $U_r = -1/γ$.

(The non-paused $V_h (g^"reach")$ equals $1$ even at $t = 1, 2$, before $G$ is
actually reached: the $V_h$ recursion propagates the eventual indicator
backwards with $γ_h = 1$.) Hence $EE [U_r (s_t)] = -p/β - (1 - p)/γ$ for each $t in {1, ..., 9}$,
and @eq-int-Vr-unroll gives
$ V_r^A (s_0) = -1/α - 9 p/β - 9 (1 - p)/γ. $ <eq-VrA>

#note[The $(1 - p)^2$ in $α$ rather than $(1 - p)$ is exactly the Empo
"Jensen penalty": $ζ = 2$ squares $V_h$ before averaging, so pre-fork
uncertainty is strictly worse than the same outcome resolved -- compare
$X_h (s_0) = 1 + m_g (1 - p)^2 + m_w$ to the resolved expectation
$EE[X_h (s_1)] = 1 + m_g (1 - p) + m_w$.]

== Strategy $B$ -- switch first

The trajectory is deterministic: $R, (3,1), S, (3,1), R, P, (1,2), G$ for
$t = 0, ..., 7$, then stays at $G$. The switch is pressed at $t = 2$ and stays
pressed.

- $t in {0, 1}$ -- switch unpressed, $G$ will be reached: $V_h (g^"reach") = V_h (g^"sw") = 1$, $X_h = γ$, $U_r = -1/γ$.
- $t in {2, ..., 9}$ -- switch pressed, $G$ will be (or already is) reached: $V_h (g^"reach") = 1$, $V_h (g^"sw") = 0$, $X_h = δ$, $U_r = -1/δ$.

By @eq-int-Vr-unroll,
$ V_r^B (s_0) = -2/γ - 8/δ. $ <eq-VrB>

== Strategy $C$ -- idle

Robot stays at $R$. Switch never pressed, $G$ never reached: at every state
$V_h (g^"reach") = 0$, $V_h (g^"sw") = 1$, $X_h = β$, $U_r = -1/β$. So
$ V_r^C (s_0) = -10/β. $ <eq-VrC>

== Decision criteria

*Proposition ($A$ weakly dominates $C$).*
$ V_r^A (s_0) - V_r^C (s_0) = (m_g (1 - p)^2)/(α β) + (9 (1 - p) m_g)/(β γ) >= 0, $
with equality iff $m_g = 0$ or $p = 1$.

*Proof.* From @eq-VrA and @eq-VrC,
#unnumbered($
V_r^A - V_r^C = -1/α + (10 - 9 p)/β - 9 (1 - p)/γ = [1/β - 1/α] + 9 (1 - p) [1/β - 1/γ].
$)
With $α - β = m_g (1 - p)^2$ and $γ - β = m_g$ this rearranges to the claim,
and both fractions are non-negative. #h(1fr) $square$

Geometrically: every state the idle robot visits has $X_h = β$, the same as
the paused branch of $A$; the non-paused branch upgrades to $γ > β$, and $s_0$
itself has $α >= β$ (note $α - β = m_g (1 - p)^2 >= 0$). So $A$ Pareto-improves
on $C$ state-by-state.

The real contest is between $A$ (risk the pause) and $B$ (sacrifice
"switch unused"):

*Proposition ($A$ vs $B$).* $V_r^A (s_0) > V_r^B (s_0)$ iff
$ 1/α + (9 p m_g)/(β γ) < (1 + m_g + 8 m_w)/(γ δ). $ <eq-AvsB>

*Proof.* By @eq-VrA and @eq-VrB, $V_r^A > V_r^B$ iff
#unnumbered($ 1/α + 9 p/β + 9 (1 - p)/γ < 2/γ + 8/δ. $)
The right-hand side simplifies as
#unnumbered($ 2/γ + 8/δ - 9/γ = 8/δ - 7/γ = (8 γ - 7 δ)/(γ δ) = (1 + m_g + 8 m_w)/(γ δ), $)
absorbing the $9/γ$ from the left. The remaining left-hand side is
#unnumbered($ 1/α + 9 p (1/β - 1/γ) = 1/α + (9 p m_g)/(β γ), $)
using $γ - β = m_g$. #h(1fr) $square$

#note[The left-hand side of @eq-AvsB is increasing in $p$ ($α$ shrinks and the
explicit $p$ term grows), the right-hand side is constant in $p$, so the
preference flips at most once as $p$ goes from $0$ to $1$ -- a single critical
pause probability $p^* (m_g, m_w)$ for the committed-policy comparison.]

== Worked examples

*Script defaults: $m_g = m_w = 1, p = 0.5$.* Then $α = 9/4, β = 2, γ = 3, δ = 2$:
#unnumbered($
V_r^A = -4/9 - 9/4 - 3/2 approx -4.194, quad
V_r^B = -2/3 - 4 approx -4.667, quad
V_r^C = -5.
$)
The robot takes the direct path ($A$ wins by $approx 0.47$).

*High pause probability: $m_g = m_w = 1, p = 0.9$.* Then $α = 2.01$, others unchanged:
#unnumbered($
V_r^A approx -0.498 - 4.05 - 0.3 = -4.848, quad V_r^B = -4.667.
$)
Now $V_r^B > V_r^A$: a globally $V_r$-optimal commitment would press the
switch first. (The implemented $Q_r$-greedy solver does *not* switch here -- see below.)

*Heavy switch-unused weight: $m_g = 1, m_w = 10, p = 0.5$.* Then $α = 11.25, β = 11, γ = 12, δ = 2$:
#unnumbered($
V_r^A approx -0.089 - 0.409 - 0.375 = -0.873, quad
V_r^B approx -0.167 - 4 = -4.167, quad
V_r^C approx -0.909.
$)
$A$ still wins -- the switch-unused goal is so heavily weighted that pressing
the switch is unaffordable, so the robot accepts the pause risk. (And $C$
nearly ties $A$: with $m_w gt.tilde m_g$ both "do nothing" and "try" look
much the same.)

*No switch-unused goal: $m_g = 1, m_w = 0, p = 0.5$.* Then $α = 1.25, β = 1, γ = δ = 2$:
#unnumbered($
V_r^A = -0.8 - 4.5 - 2.25 = -7.55, quad
V_r^B = -1 - 4 = -5, quad
V_r^C = -10.
$)
The robot presses the switch -- with $m_w = 0$ there is no penalty for using it.

== Critical pause probability $p^*$

Setting $V_r^A = V_r^B$ in @eq-AvsB and writing $q := 1 - p$,
#unnumbered($ 1/(1 + m_g q^2 + m_w) + (9 (1 - q) m_g)/((1 + m_w)(1 + m_g + m_w)) = (1 + m_g + 8 m_w)/((1 + m_g)(1 + m_g + m_w)). $)

For $m_g = m_w = 1$ this collapses to
#unnumbered($ 1/(2 + q^2) + (3 (1 - q))/2 = 5/3 quad <==> quad 1/(2 + q^2) = (1 + 9 q)/6, $)

which after clearing denominators is the cubic
$ 9 q^3 + q^2 + 18 q - 4 = 0. $ <eq-int-cubic>

@eq-int-cubic has a single real root in $(0, 1)$ (the polynomial is strictly
increasing on $[0, 1]$, negative at $q = 0$, positive at $q = 1$): numerically
$q^* approx 0.2145$, so
#unnumbered($ p^* approx 0.7855. $)

Sanity-check by direct evaluation at $p = p^*$:
#unnumbered($ α = 1 + 0.2145^2 + 1 approx 2.046, quad
V_r^A approx -0.489 - 3.535 - 0.645 = -4.667, quad V_r^B = -14/3 approx -4.667. quad checkmark $)

So with the default goal multiplicities, a $V_r$-optimal commitment would
flip from $A$ to $B$ around $p approx 0.79$. The $Q_r$-greedy solver, as we
now show, does not.

== Empirical: what the solver actually does

Running [`StochasticBackwardInductionSolver`](../src/grid_world/solvers/stochastic_backward_induction.py)
for $m_g = m_w = 1$ confirms the gap: the solver picks LEFT ($A$) at $s_0$ for
every $p in (0, 1)$, achieving $V_r = V_r^A (p)$:

#table(
  columns: 4,
  inset: 6pt,
  align: (left, right, right, left),
  [*$p$*], [*solver $V_r (s_0)$*], [*$V_r^A$ (commit $A$)*], [*$V_r^B$ (commit $B$)*],
  [$0.3$], [$-3.852$], [$-3.852$], [$-4.667$],
  [$0.5$], [$-4.194$], [$-4.194$], [$-4.667$],
  [$0.7$], [$-4.528$], [$-4.528$], [$-4.667$],
  [$0.8$], [$-4.690$], [$-4.690$], [$-4.667$ (better)],
  [$0.9$], [$-4.848$], [$-4.848$], [$-4.667$ (better)],
)

For $p > p^* approx 0.785$ the solver continues to follow $A$ and so achieves
$V_r^A$, even though $V_r^B$ is strictly higher. The reason is the one
flagged in the intro: at $s_0$, $Q_r ("LEFT")$ averages $V_r (s_1)$ over the
paused / non-paused outcomes, $Q_r ("DOWN")$ takes $V_r$ at the deterministic
next state $(3, 1)$ -- but the $V_r$ values at $s_1$ are themselves the
solver's recursive $Q_r$-greedy values, and at $(3, 1)$ the same myopic
comparison again picks UP (back to $R$) over DOWN (to $S$), so the
"press the switch" sub-tree is never realised.

A richer family of committed policies recovers the rest of the picture. Let
$D_k$ be "*delay $k$ steps at $R$, then take $B$'s route*", reaching $G$ at
step $k + 7$ (still $<= N$ for $k <= 3$). The switch is then pressed at step
$k + 2$, so the trajectory contains $k + 2$ states with $X_h = γ$ and $8 - k$
states with $X_h = δ$. Hence
$ V_r^(D_k) (s_0) = -(k + 2)/γ - (8 - k)/δ, $ <eq-VrDk>
and $partial_k V_r^(D_k) = m_w / (γ δ) >= 0$ -- so the latest legal switch
press $D_3$ dominates this family whenever $m_w > 0$:
$ V_r^(D_3) (s_0) = -5/γ - 5/δ. $

For $m_g = 2, m_w = 1$ this gives $V_r^(D_3) = -35/12 approx -2.917$, beating
$V_r^A approx -4.77$ at any $p$, $V_r^B approx -3.17$, and even
$V_r^(D_2) = -3$. The solver picks $D_2$ (sequence ${0, 0, 3, 3, 1, 1, 2, 2, 2, 0}$,
total $V_r = -3$) -- so it does find that *some* delay helps, but the
$Q_r$-greedy step at $s_2$ commits to DOWN one move too early, missing the
$D_3$ improvement by the same indirect-$U_r$ effect as in the $A$ vs $B$
case.

Concretely, the gap between the committed-policy optimum and what the solver
achieves is:
- driven entirely by the goals being non-terminal (so $U_r$ varies along the
  trajectory),
- generally small in $V_r$ but qualitatively meaningful: at $p > p^*$ the
  solver foregoes a finite improvement; under $m_w = 0$ or large $m_g$ the
  family $D_k$ would extract more value than the solver does.

A $V_r$-greedy fix would just change the policy-selection step to
$π_r (s) = arg max_a (U_r (s; a) + γ_r EE [V_r (s')])$,
with $U_r (s; a)$ computed via the post-action $V_h (s, dot)$ -- but this
breaks the clean decomposition of #link("3_simplified.typ")[`3_simplified.typ`]
and was presumably not the intent.
