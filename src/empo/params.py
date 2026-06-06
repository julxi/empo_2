"""Parameters of the Empo objective."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmpoParameter:
    gamma_r: float = 1
    beta_r: float = 1
    gamma_h: float = 1
    zeta: float = 2
    xi: float = 1
    eta: float = 1
