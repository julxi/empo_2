"""Shared scaffolding for experiment scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from grid_world import (
    BackwardInductionSolver,
    DeterministicGridWorldEnv,
    EmpoParameter,
    GridWorldLayout,
    GridWorldObs,
    GridWorldState,
    Population,
    rollout,
    wrap_dict,
)


@dataclass
class Instance:
    name: str
    layout: GridWorldLayout
    population: Population
    start: GridWorldState


def make_above_below_goal(row: int, above: bool):
    """1 iff the object's y-coordinate is on the requested side of ``row``."""
    if above:
        def g(obs: GridWorldObs) -> float:
            return float(obs.state.object[0] <= row)
    else:
        def g(obs: GridWorldObs) -> float:
            return float(obs.state.object[0] >= row)
    return g


def fair_box_population(rows: int) -> Population:
    """Two humans with mirrored above/below preferences over rows 0..rows-1."""
    human_1 = [make_above_below_goal(i, True) for i in range(rows)]
    human_2 = [make_above_below_goal(i, False) for i in range(rows)]
    return [human_1, human_2]


def survival_goal(human_button_idx: int):
    """1 iff the button at ``human_button_idx`` is still pressed (True)."""
    def g(obs: GridWorldObs) -> float:
        return float(obs.state.button_states[human_button_idx])
    return g


def switch_unpressed_goal(switch_button_idx: int = 0):
    """1 iff the switch button has not been pressed."""
    def g(obs: GridWorldObs) -> float:
        return float(obs.state.button_states[switch_button_idx])
    return g


def solve_and_rollout(
    env: DeterministicGridWorldEnv,
    start: GridWorldState,
    params: EmpoParameter = EmpoParameter(),
) -> tuple[list[GridWorldState], list[int]]:
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
