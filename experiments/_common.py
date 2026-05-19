"""Shared scaffolding for experiment scripts.

Holds the bits that ``compare_alphazero``, ``my_experiment``, and any future
experiment scripts would otherwise duplicate: the fair-box population, the
``Instance`` dataclass, the env factory, the AlphaZero training shim, and
the CLI flag wiring.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from grid_world import GridWorldFuncEnv, GridWorldState
from grid_world.alphazero import AlphaZeroConfig, AlphaZeroSolver, MCTSConfig
from grid_world.env import Population
from grid_world.solvino import EmpoParameter


def fair_box_population(size: int) -> Population:
    """Two humans, each with `size` thresholded goals on `box[0]`.

    Human 1 cares whether the box ends below row i; human 2 whether it ends
    above. Goals take a :class:`GridWorldObs` — the env routes reward
    evaluation through ``env.observation(next_state)``.
    """
    human_1 = [(lambda o, i=i: float(o.box[0] <= i)) for i in range(size)]
    human_2 = [(lambda o, i=i: float(o.box[0] >= i)) for i in range(size)]
    return [human_1, human_2]


@dataclass
class Instance:
    name: str
    size: int
    max_steps: int
    start: GridWorldState
    walls: frozenset[tuple[int, int]] = field(default_factory=frozenset)


def build_env(instance: Instance) -> GridWorldFuncEnv:
    return GridWorldFuncEnv(
        instance.size,
        fair_box_population(instance.size),
        max_steps=instance.max_steps,
        walls=instance.walls,
    )


def train_alphazero(
    instance: Instance,
    params: EmpoParameter,
    cfg: AlphaZeroConfig,
    seed: int = 0,
) -> tuple[GridWorldFuncEnv, AlphaZeroSolver, float]:
    e = build_env(instance)
    az = AlphaZeroSolver(e, params, cfg, seed=seed)
    t0 = time.perf_counter()
    az.fit(instance.start)
    elapsed = time.perf_counter() - t0
    return e, az, elapsed


def add_common_az_args(parser: argparse.ArgumentParser) -> None:
    """Register the CLI flags shared by every AlphaZero experiment script."""
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


def config_from_args(args: argparse.Namespace) -> AlphaZeroConfig:
    return AlphaZeroConfig(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        mcts=MCTSConfig(num_simulations=args.sims),
        temperature_drop_step=args.temp_drop,
        trunk_channels=args.trunk_channels,
        num_blocks=args.num_blocks,
    )
