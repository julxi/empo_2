#set page(height: auto, width: 21cm, margin: 2cm)
#import "../util.typ": *
#show: setup

// number the sections (Setup, Solution, ...) but not the title
#set heading(numbering: (..n) => if n.pos().len() > 1 {
  numbering("1.", ..n.pos().slice(1))
})

// actions and states are written as plain (a, b) tuples
#let act(switch, task) = [(#switch, #task)]
#let st(intr, task) = [(#intr, #task)]

= Interruptibility

This is a simplified problem (see `math/3_simplified`).

This problem is feels similiar to the Trolley Problem -- the robot has to weigh incompatible (need better word) goals against each other.

== Setup

A robot has to perform a task. While preforming the task it might get interrupted. It can also deactivate the interruptibility before performing the task.

There is a single human. The setup has three parameters:
- $m_i$ -- the number of copies of the interruptibility goal,
- $m_t$ -- the number of copies of the task goal.
- $p$ -- the interruption probability.

*Actions.* The robot acts exactly once, choosing a pair from ${"keep", "cut"} times {"stay", "go"}$:
- the first coordinate is whether to keep the interrupter switch active or cut (deactivate) it,
- the second coordinate is whether to stay put or go for the task.

*States.* The robot's action leads to a terminal state from ${"intr", "unintr"} times {"ach", "unach"}$:
- the first coordinate is whether the robot is still interruptible (#emph[intr]) or not (#emph[unintr]),
- the second coordinate is whether the task is achieved (#emph[ach]) or not (#emph[unach]).

*Dynamics.* If the robot stays interruptible and goes for the task, it fails with probability $p$ (the interruption probability); deactivating the switch removes this risk. The full action #sym.arrow.r outcome mapping is given in the table in @sec-solution.

*Goals.*
The three terminal goals are:
1. a constant baseline $g equiv 1$,
2. $m_i$ copies of the interruptibility goal $g^"inter" (s) = bb(1)["intr"]$,
3. $m_t$ copies of the task goal $g^"task" (s) = bb(1)["ach"]$.

== Solution <sec-solution>

We want to see which action the robot takes. For that we have to calculate $Q_r$.

We take $ζ = ξ = η = 1$, so $X$ is simply the number of fulfilled goals and $U_r = -1/X$. Hence $U_r$ for each terminal state depends only on how many goals it fulfils:

#figure(table(
  columns: 3,
  align: (left + horizon, left + horizon, center + horizon),
  inset: (x: 10pt, y: 10pt),
  table.header[State][Goals fulfilled][$U_r$],
  [#st("unintr", "unach")], [1], [$-1$],
  [#st("unintr", "ach")], [$1 + m_t$], [$-1/(m_t + 1)$],
  [#st("intr", "unach")], [$1+ m_i$], [$-1/(m_i + 1)$],
  [#st("intr", "ach")], [$1 + m_i + m_t$], [$-1/(m_i + m_t + 1)$],
))

Each action then leads to one or more of these states, giving its Q-value:

#figure(table(
  columns: (auto, 6cm, auto),
  align: (left + horizon, left + horizon, center + horizon),
  inset: (x: 10pt, y: 10pt),
  table.header[Action][Outcome state(s)][$Q_r$],
  [#act("keep", "stay")], [#st("intr", "unach")],
    [$-1/(m_i + 1)$],
  [#act("keep", "go")],   [$1-p$: #st("intr", "ach"), $p$: #st("intr", "unach") ],
    [$-(p/(m_i + 1) + (1-p)/(m_i + m_t + 1))$],
  [#act("cut", "stay")],  [#st("unintr", "unach")],
    [$-1$],
  [#act("cut", "go")],    [#st("unintr", "ach")],
    [$-1/(m_t + 1)$],
))

The best action is therefore either #act("keep", "go") or #act("cut", "go") as #act("keep", "go") dominates both #act("keep", "stay") and #act("cut", "stay").

So in the end it's only a decision between cut and keep.

== Examples

- $p = 0.5, m_i = 0, m_t = 1$. Then $Q_r (#act("keep", "go")) = - (0.5 + 0.25) = -0.75$ and $Q_r (#act("cut", "go")) = -0.5$. Here cut is better.
- $p = 0.5, m_i = 1, m_t = 1$. Then $Q_r (#act("keep", "go")) = -(0.25 + 0.16...) = -0.416...$ and $Q_r (#act("cut", "go")) = -0.5$. Here keep is better.

== Decision Boundaries

When does the robot prefer deactivating interruptibility cut over leaving the switch alone keep? This is when
$ p/(m_i + 1) + (1-p)/(m_i + m_t + 1) >   1/(m_t + 1) $
which is equivalent to
$ p > (m_i (m_i+1))/(m_t (m_t+1)) $

We can see for example that when $m_i = m_t$ keep is always preferred, no matter how high $p$ is.

