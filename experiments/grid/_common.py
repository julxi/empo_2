"""Shared scaffolding for experiment scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from empo.core import DeterministicEnv, Population
from empo.solvers.params import EmpoParameter
from empo.envs.grid.base import GridConfig, GridState
from empo.solvers.backward_induction import BackwardInductionSolver
from empo.solvers.trajectory import rollout, wrap_dict


@dataclass
class Instance:
    name: str
    config: GridConfig
    population: Population
    start: GridState


def solve_and_rollout(
    env: DeterministicEnv,
    start: GridState,
    params: EmpoParameter = EmpoParameter(),
) -> tuple[list[GridState], list[int]]:
    solver = BackwardInductionSolver(env, params)
    solver.solve(start)
    return rollout(env, wrap_dict(solver.robot_policy), start)


def add_common_az_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags shared by every AlphaZero experiment script."""
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--episodes-per-iter", type=int, default=8)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--eval-sims", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temp-drop",
        type=int,
        default=8,
        help="step after which self-play becomes greedy",
    )
    parser.add_argument("--trunk-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="restrict to a single instance by name",
    )
