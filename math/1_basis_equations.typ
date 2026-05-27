#import "util.typ": *
#show: setup

= Basis for Implementation

We want to solve the equations from #link("0_empo_intro.typ")[the introduction].

For the mindset of the implementation we might have some slight changes:

As far as the math is concerned humans could be considered to be part of the environment. Since the human policy depends on the goal, this makes the environment goal dependent $T(s,a_r,g)$.

I'll be working with this slightly changed definition of the intro equations (only (1) and (3) look different):


$ Q_r (s, a_r) <- bb(E)_g space bb(E)_(s' tilde.op T(s, a_r, g)) space γ_r V_r (s') $

$ π_r (s)(a_r) prop (-Q_r (s, a_r))^(-β_r) $

$ V_h (s, g_h) &<- cases(1 &"if" s in g_h,
     bb(E)_(a_r tilde.op π_r (s)) space bb(E)_(g_(-h)) space bb(E)_(s' tilde.op T(s, a_r, g)) space γ_h V_h (s', g_h) &"else") $


$ X_h (s) <- sum_(g_h in cal(G)_h) V_h (s, g_h)^ζ $


$ U_r (s) <- - (sum_h X_h (s)^(-ξ))^η $


$ V_r (s) <- U_r (s) + bb(E)_(a_r tilde.op π_r (s)) Q_r (s, a_r) $



