#set page(height: auto, width: 21cm, margin: 2cm)
#import "util.typ": *
#show: setup

#set heading(numbering: "1.")

= The Race: helping the underdog

This note works out, analytically, why the empo-bot in
#link("../experiments/helping.py")[`experiments/helping.py`] behaves the way it
does. The robot can only ever _pull a racer backwards_, yet along the trajectory
where racer 0 sprints ahead while racer 1 is stuck at the start it does *nothing*
until racer 0 is one square from the line, and then *pulls the leader back*:

#unnumbered($
  (4,0) #h(0.5em) -> #h(0.5em) "pull racer 0 to" (3,0).
$)

We use the simplified, stochastic equations of
#link("3_simplified.typ")[the simplified case] (the environment dynamic is
stochastic but goal-independent, and the robot is greedy).

== Setup

Two racers run a track of length `len_track` $= 6$; the finish line is the last
cell $F = 5$. State $s = ("progress", "race_result", "step")$ records each
racer's cell, the finishing order so far, and the step counter. The episode is a
fixed-length window of $N = 12$ steps (`max_steps`), and a state is _terminal_
exactly when $"step" >= 12$ -- the race never ends early.

*Dynamics* (`RaceEnv._next_states`). Each step: the robot first nudges one racer
(here a pull, $-1$), then every racer that has not yet finished independently
either *trips* (probability $p = 1/2$, stays put) or *steps* forward one cell.
A racer that reaches $F$ is recorded in `race_result` and frozen. So, absent
interference, a racer advances by $+1$ with probability $1/2$ each step; reaching
the line needs the right number of successes inside the remaining steps.

*Actions* (`RaceEnv.translate_action`, mode `PULL`). There are three:
pull racer 0, pull racer 1, or noop ($a = 2$). A pull subtracts one cell from the
target before it takes its own step; `racer_affected = a mod 2`.

*Population* (`winner_population`). There is one human per racer, and human $h$
cares about racer $h$ through three goals:

#unnumbered($
  g^"win"_h &= bb(1)["racer" h "crosses the line first"], \
  g^"fin"_h &= bb(1)["racer" h "finishes in the top 2"] = bb(1)["racer" h "finishes at all"], \
  g^"term" &= bb(1)["episode reached step" N] equiv 1,
$)

where the top-2 goal collapses to "finishes at all" because there are only two
racers. The Empo parameters are the defaults

#unnumbered($ gamma_r = gamma_h = 1, quad zeta = 2, quad xi = 1, quad eta = 1. $)

== The Empo quantities for this environment

The goal value $V_h (s, g)$ obeys the goal-value recursion of
#link("3_simplified.typ")[the simplified case]: it equals the goal indicator when
the goal fires, and otherwise the (undiscounted) expectation over the next state
under $pi_r$. Each goal here fires *exactly once* along any
trajectory -- at the single step a racer crosses with the required placing -- so

#unnumbered($
  V_h (s, g^"term") = 1 quad "(always reached)", quad
  V_h (s, g^"win"_h) = w_h (s), quad
  V_h (s, g^"fin"_h) = f_h (s),
$)

where $w_h (s)$ and $f_h (s)$ are the probabilities that, _from $s$ onwards under
$pi_r$_, racer $h$ still goes on to win, resp. to finish. These are
*forward-looking*: once racer $h$ has crossed, the win/finish events lie in the
past, the indicator can no longer fire, and $w_h, f_h$ collapse to $0$ at all
_later_ states (their single contribution of $1$ is booked at the crossing step
itself). Two structural facts matter:

#unnumbered($ w_h <= f_h, quad "and" quad w_0 + w_1 <= 1. $)

The win goal is *rivalrous*: at most one racer can be first, so a unit of
win-probability handed to one human is taken from the other. (Simultaneous
crossings are broken towards the lower index in `_next_states`.)

With $zeta = 2$ the per-human aggregate $X_h = sum_(g) V_h (s, g)^zeta$ is

#unnumbered($ X_h (s) = w_h (s)^2 + f_h (s)^2 + 1, $)

so $X_h in [1, 3]$: it is $1$ once the racer's win and finish potential are spent
(only the constant terminal goal remains) and $3$ at the instant a racer secures
both. With $xi = eta = 1$ the robot's instantaneous utility
$U_r = -(sum_h X_h^(-xi))^eta$ is the *negated sum of reciprocals*

$
  U_r (s) = -(1/(X_0 (s)) + 1/(X_1 (s)))
          = -(1/(w_0^2 + f_0^2 + 1) + 1/(w_1^2 + f_1^2 + 1)),
$ <eq-race-Ur>

and, since $gamma_r = 1$ and the episode has fixed length, the robot value
$V_r (s) = U_r (s) + gamma_r V_r (s')$ unrolls to simply the *sum of utilities
over the remaining steps*:

$ V_r (s) = sum_(t >= "step"(s)) bb(E)[U_r (s_t) | s, pi_r]. $ <eq-race-Vr>

This integral-over-a-fixed-window structure is the whole story.

== Why pull at all: live, balanced potential

Two properties of @eq-race-Ur drive everything.

*Potential is consumed.* $U_r$ ranges from $-2$ (both $X_h = 1$: nothing left to
achieve) up to $-2 \/ 3$ (both $X_h = 3$). During an active, competitive race
$X_h approx 1.5 - 1.9$ and $U_r approx -1.3$; once a racer has finished and parked,
its $w_h, f_h$ fall back to $0$, $X_h -> 1$, and $U_r$ drifts towards $-2$.
Because $V_r$ (@eq-race-Vr) *sums $U_r$ over a window of fixed length*, spending a
racer's potential _early_ means many subsequent steps sit in the bad
$U_r approx -2$ regime. The robot is therefore pushed to *keep achievement
potential alive as long as possible* -- i.e. to delay finishing.

*Reciprocals reward balance.* $x |-> 1\/x$ is convex and decreasing, so
@eq-race-Ur is dominated by the *worst-off* human: a small $X_h$ contributes a
large penalty. Letting one racer monopolise the (rivalrous) win goal drives the
other's $w_h$ to $0$, and an early, lopsided finish is the worst case of all.

Both pressures point the same way, and the robot's only lever -- `PULL` -- serves
both: pulling the leader back postpones the moment a racer's potential is spent
*and* keeps the trailing human's win-probability from being zeroed. Pulling the
_trailing_ racer would do the opposite, so action 1 is never chosen.

== The pivotal decision at $(4,0)$

At $s_4 = ("progress" = (4,0), "step" = 4)$ racer 0 is one cell from the line. The
solver's successor values (all at step 5) are:

#titled("noop")[
  #unnumbered($
    1/4 underbrace((5,1)\,(0), V_r = -13.94) + 1/4 underbrace((5,0)\,(0), V_r = -14.77)
    + 1/4 underbrace((4,1), V_r = -11.26) + 1/4 underbrace((4,0), V_r = -12.12)
    = -13.02
  $)
]
#titled("pull racer 0")[
  #unnumbered($
    1/4 underbrace((4,1), -11.26) + 1/4 underbrace((4,0), -12.12)
    + 1/4 underbrace((3,1), -11.26) + 1/4 underbrace((3,0), -12.12)
    = -11.69
  $)
]

The pull wins by $1.33$, and the gap is *entirely* the two branches in which the
noop lets racer 0 cross first this step (`race_result` $= (0)$, the
$V_r approx -14$ outcomes). Pulling racer 0 from cell 4 *guarantees it cannot
reach $F = 5$ this step*, so those lock-in branches are replaced by branches where
the race is still alive ($V_r approx -11.7$).

Why is the lock-in branch $(5,0), (0)$ worth only $V_r = -14.77$? At that state
racer 0 has just won, so momentarily $X_0 = 1^2 + 1^2 + 1 = 3$, but racer 1 is
locked out of winning ($w_1 = 0$) with only a finishing chance left,
$f_1 = 29\/128 approx 0.227$ (it needs $>= 5$ successes in the $7$ remaining steps),
giving $X_1 = 1 + f_1^2 approx 1.05$ and $U_r = -(1\/3 + 1\/1.05) approx -1.28$.
Worse, over the $approx 7$ remaining steps racer 0 is parked ($X_0 -> 1$) and racer
1 almost surely fails to finish, so $U_r -> -2$; the sum @eq-race-Vr accumulates to
$approx -14.8$. The competitive branch $(4,1)$ keeps both racers in play at
$U_r approx -1.3$ per step and totals only $approx -11.3$.

== Why the robot waits until cell 4

The same comparison along the sprint trajectory $(i, 0)$ shows noop is preferred
at every earlier cell, with the margin collapsing just before the brink:

#align(center, table(
  columns: 6,
  align: center,
  table.header($"racer 0 at"$, $0$, $1$, $2$, $3$, $bold(4)$),
  $Q_r ("noop")$, $-15.96$, $-14.70$, $-13.62$, $-12.64$, $-13.02$,
  $Q_r ("pull 0")$, $-16.19$, $-15.14$, $-13.82$, $-12.70$, $bold(-11.69)$,
  $"chosen"$, "noop", "noop", "noop", "noop", $bold("pull")$,
))

For cells $<= 3$ a single pull is *almost undone*: racer 0 simply re-climbs next
step, no finish is imminent either way, and the eventual win/finish
probabilities barely move -- so the pull buys nothing while slightly eroding racer
0's own potential (a sustained pull could even stop it finishing, hurting $f_0$).
Hence $Q_r ("pull") lt.tilde Q_r ("noop")$ and the robot stays passive. The lever's
value is *concentrated at the brink*: only at cell 4 does a noop carry a $1\/2$
chance of an *irreversible* lock-in this very step, and only there does delaying
it pay. So the robot waits and intervenes at the last possible moment.

#note[
  The behaviour is "helping the underdog", but note its origin: it is not a
  preference for racer 1, it is the convex, forward-looking utility
  @eq-race-Ur integrated over a fixed window. The robot keeps the rivalrous win
  goal *alive and shared* rather than letting the leader spend it early and
  alone. With only a `PULL` lever available, the cheapest way to do that is to
  hold the leader at the line until the field has a chance to catch up.
]

== Verification

#link("../experiments/helping.py")[`experiments/helping.py`] prints the chosen
action and $Q_r$ at each cell of the sprint trajectory and reproduces the table
above; running it shows `RaceAction(offset=-1, racer=0)` (pull racer 0) appearing
only at $(4,0)$. The successor values, $f_1 = 29\/128$, and the
$Q_r ("noop") = -13.02$ / $Q_r ("pull") = -11.69$ decomposition can be checked
against `solver.V_r`, `solver.Q_r`, and `solver.V_h` after `solver.solve(start)`.
