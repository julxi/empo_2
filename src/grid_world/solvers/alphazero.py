"""AlphaZero-style solver for the simplified Empo equations.

The grid-world MDP is deterministic and acyclic (the `step` field of the state
strictly increases), so the value of a complete trajectory is well-defined and
can be computed by a single backward sweep (see `empo_eval.evaluate_trajectory`).
We treat that V_r as the optimisation target for the robot policy.

Pieces:
- `PolicyValueCNN`: small residual CNN that consumes the channel-stacked state
  description (robot / object / walls / step) and outputs
  (policy logits, value). Convolutions share weights spatially, which is the
  right inductive bias for sokoban-style problems with walls and box pushing.
- `MCTS`: standard PUCT search. Since dynamics are deterministic, Q at edge
  (s,a) directly tracks V_r at the child state T(s,a); we do not add per-step
  U_r terms inside the tree (they are constant in `a` and so do not affect
  argmax). The leaf evaluation uses the network's V_r prediction.
- `self_play`: runs MCTS at every step to produce an episode, then turns it
  into (state, mcts_policy, V_r_target) tuples via the trajectory evaluator.
- `train`: trains the network on those tuples.

The encoder below is currently box-flavoured (robot / object / walls / step
planes). It is enough for `BoxToMiddleEnv`; trolley / interruptibility will
want their own planes (e.g. button_states) once we train on them. Deferred.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn

from ..base import Action, GridWorldObs, GridWorldState
from ..empo import EmpoParameter
from ..empo_eval import evaluate_trajectory
from ..env_base import DeterministicGridWorldEnv

# --------------------------------------------------------------------------- #
# State encoding                                                              #
# --------------------------------------------------------------------------- #


CHANNEL_NAMES: tuple[str, ...] = ("robot", "object", "walls", "step")


def encode_obs(obs: GridWorldObs) -> np.ndarray:
    """Channel-stacked spatial encoding with shape ``(C, width, height)``.

    Channels (see :data:`CHANNEL_NAMES`):
      0. robot indicator
      1. object indicator
      2. walls indicator (1 at every wall cell)
      3. normalised step, broadcast across the grid (``step / max_steps``)
    """
    state = obs.state
    layout = obs.layout
    channels = np.zeros(
        (len(CHANNEL_NAMES), layout.width, layout.height), dtype=np.float32
    )
    channels[0, state.robot[0], state.robot[1]] = 1.0
    channels[1, state.object[0], state.object[1]] = 1.0
    for wx, wy in layout.walls:
        channels[2, wx, wy] = 1.0
    channels[3, :, :] = state.step / layout.max_steps
    return channels


# --------------------------------------------------------------------------- #
# Network                                                                     #
# --------------------------------------------------------------------------- #


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.body(x))


class PolicyValueCNN(nn.Module):
    """Small residual CNN suited to sokoban-style grid problems.

    No batch norm: self-play replay batches are small and skewed, which makes
    BN's running statistics unstable in this regime. Plain ReLU activations
    are enough at this scale.
    """

    def __init__(
        self,
        in_channels: int = len(CHANNEL_NAMES),
        board_width: int = 5,
        board_height: int = 5,
        trunk_channels: int = 32,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        cells = board_width * board_height
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResBlock(trunk_channels) for _ in range(num_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(trunk_channels, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * cells, len(Action)),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(trunk_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(cells, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# MCTS                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class MCTSNode:
    state: GridWorldState
    terminal: bool
    prior: np.ndarray = field(
        default_factory=lambda: np.ones(len(Action)) / len(Action)
    )
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    N: np.ndarray = field(default_factory=lambda: np.zeros(len(Action), dtype=np.int64))
    W: np.ndarray = field(
        default_factory=lambda: np.zeros(len(Action), dtype=np.float64)
    )
    expanded: bool = False
    value: float = 0.0  # network value estimate when expanded

    def Q(self, default: float = 0.0) -> np.ndarray:
        return np.where(self.N > 0, self.W / np.maximum(self.N, 1), default)

    def total_N(self) -> int:
        return int(self.N.sum())


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    root_noise_frac: float = 0.25
    rollout_temperature: float = 1.0
    # Default Q value for an action that has never been visited at a node.
    # 0 is optimistic for this all-negative-valued problem, which is useful for
    # encouraging exploration; can be set to the network value with "fpu" mode.
    fpu_mode: str = "zero"  # "zero" or "value"
    fpu_reduction: float = 0.0


class MCTS:
    """PUCT search with trajectory-based backup.

    Unlike the canonical AlphaZero backup (which propagates the leaf's value
    unchanged up the tree), the Empo V_r is depth-dependent: per-state U_r
    contributions accumulate along the trajectory. Averaging V_r values at
    different depths is meaningless.

    Instead, every simulation rolls out from the leaf to terminal using the
    network's policy (sampled with `rollout_temperature`), giving a full
    trajectory root -> ... -> leaf -> ... -> terminal. We compute V_r at every
    state in that trajectory via `evaluate_trajectory`, then back up the
    *child* V_r for each tree-explored edge (so Q(s, a) tracks V_r at T(s, a),
    which is what argmax_a Q(s, a) needs).
    """

    def __init__(
        self,
        env: DeterministicGridWorldEnv,
        params: EmpoParameter,
        net: PolicyValueCNN,
        config: MCTSConfig,
        device: torch.device,
    ) -> None:
        self.env = env
        self.params = params
        self.net = net
        self.config = config
        self.device = device

    def _evaluate(self, state: GridWorldState) -> tuple[np.ndarray, float]:
        encoded = encode_obs(self.env.observation(state))
        x = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        priors = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return priors, float(value.cpu().item())

    def _expand(self, node: MCTSNode) -> float:
        if node.terminal:
            node.value = 0.0
            node.expanded = True
            return 0.0
        priors, value = self._evaluate(node.state)
        node.prior = priors
        node.value = value
        node.expanded = True
        return value

    def _add_dirichlet_noise(self, node: MCTSNode, rng: np.random.Generator) -> None:
        noise = rng.dirichlet([self.config.dirichlet_alpha] * len(Action))
        f = self.config.root_noise_frac
        node.prior = (1 - f) * node.prior + f * noise

    def _fpu(self, node: MCTSNode) -> float:
        if self.config.fpu_mode == "value":
            return node.value - self.config.fpu_reduction
        return 0.0 - self.config.fpu_reduction  # "zero"

    def _select_action(self, node: MCTSNode) -> int:
        total = max(node.total_N(), 1)
        q = node.Q(default=self._fpu(node))
        u = self.config.c_puct * node.prior * np.sqrt(total) / (1 + node.N)
        return int(np.argmax(q + u))

    def _sample_action(self, priors: np.ndarray, rng: np.random.Generator) -> int:
        temp = self.config.rollout_temperature
        if temp <= 1e-6:
            return int(np.argmax(priors))
        p = priors ** (1.0 / temp)
        p = p / p.sum()
        return int(rng.choice(len(Action), p=p))

    def _rollout_to_terminal(
        self,
        start_state: GridWorldState,
        rng: np.random.Generator,
    ) -> tuple[list[GridWorldState], list[int]]:
        states = [start_state]
        actions: list[int] = []
        current = start_state
        while not self.env.terminal(current):
            priors, _ = self._evaluate(current)
            a = self._sample_action(priors, rng)
            actions.append(a)
            current = self.env.transition(current, a)
            states.append(current)
        return states, actions

    def run(
        self,
        root_state: GridWorldState,
        add_root_noise: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> MCTSNode:
        rng = rng or np.random.default_rng()
        root = MCTSNode(state=root_state, terminal=self.env.terminal(root_state))
        self._expand(root)
        if add_root_noise and not root.terminal:
            self._add_dirichlet_noise(root, rng)

        for _ in range(self.config.num_simulations):
            self._simulate(root, rng)
        return root

    def _simulate(self, root: MCTSNode, rng: np.random.Generator) -> None:
        path: list[tuple[MCTSNode, int]] = []
        node = root

        # Descend the tree by PUCT until we hit an unexpanded leaf or terminal.
        while node.expanded and not node.terminal:
            action = self._select_action(node)
            path.append((node, action))
            if action not in node.children:
                child_state = self.env.transition(node.state, action)
                child = MCTSNode(
                    state=child_state, terminal=self.env.terminal(child_state)
                )
                node.children[action] = child
                node = child
                break
            node = node.children[action]

        # Make sure the leaf is expanded (gets priors and a network V estimate).
        if not node.expanded:
            self._expand(node)

        # Roll out from the leaf to a terminal state using the network policy.
        if node.terminal:
            rollout_states = [node.state]
            rollout_actions: list[int] = []
        else:
            rollout_states, rollout_actions = self._rollout_to_terminal(node.state, rng)

        # Stitch the tree path with the rollout to get the full trajectory.
        full_states = [p[0].state for p in path] + rollout_states
        full_actions = [p[1] for p in path] + rollout_actions
        traj = evaluate_trajectory(self.env, self.params, full_states, full_actions)

        # Back up V_r at the *child* of each tree-explored edge.
        for i, (parent, action) in enumerate(path):
            child_V_r = traj.V_r[i + 1]
            parent.N[action] += 1
            parent.W[action] += child_V_r


# --------------------------------------------------------------------------- #
# Self-play                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class TrainingExample:
    features: np.ndarray
    policy: np.ndarray
    value: float


def visit_policy(node: MCTSNode, temperature: float) -> np.ndarray:
    visits = node.N.astype(np.float64)
    if visits.sum() == 0:
        return np.ones(len(Action)) / len(Action)
    if temperature <= 1e-6:
        out = np.zeros_like(visits)
        out[np.argmax(visits)] = 1.0
        return out
    counts = visits ** (1.0 / temperature)
    return counts / counts.sum()


def self_play_episode(
    env: DeterministicGridWorldEnv,
    params: EmpoParameter,
    mcts: MCTS,
    start_state: GridWorldState,
    temperature: float = 1.0,
    temperature_drop_step: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[TrainingExample], list[GridWorldState], list[int]]:
    """Play one episode using MCTS, return per-step training examples plus the
    realised (state, action) trajectory for downstream inspection.
    """
    rng = rng or np.random.default_rng()
    states: list[GridWorldState] = [start_state]
    actions: list[int] = []
    policies: list[np.ndarray] = []
    feats: list[np.ndarray] = []

    current = start_state
    step_idx = 0
    while not env.terminal(current):
        temp = temperature
        if temperature_drop_step is not None and step_idx >= temperature_drop_step:
            temp = 0.0
        root = mcts.run(current, add_root_noise=True, rng=rng)
        pi = visit_policy(root, temp)
        feats.append(encode_obs(env.observation(current)))
        policies.append(pi)
        action = int(rng.choice(len(Action), p=pi))
        actions.append(action)
        current = env.transition(current, action)
        states.append(current)
        step_idx += 1

    traj = evaluate_trajectory(env, params, states, actions)
    examples = [
        TrainingExample(features=feats[t], policy=policies[t], value=traj.V_r[t])
        for t in range(len(actions))
    ]
    return examples, states, actions


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs_per_iter: int = 4
    value_loss_weight: float = 1.0


def train_step(
    net: PolicyValueCNN,
    optimizer: torch.optim.Optimizer,
    batch: Iterable[TrainingExample],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, float]:
    feats = torch.from_numpy(np.stack([b.features for b in batch])).to(device)
    target_pi = torch.from_numpy(np.stack([b.policy for b in batch])).float().to(device)
    target_v = torch.tensor(
        [b.value for b in batch], dtype=torch.float32, device=device
    )

    logits, value = net(feats)
    log_p = torch.log_softmax(logits, dim=-1)
    policy_loss = -(target_pi * log_p).sum(dim=-1).mean()
    value_loss = (value - target_v).pow(2).mean()
    loss = policy_loss + cfg.value_loss_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(policy_loss.item()), float(value_loss.item())


# --------------------------------------------------------------------------- #
# Top-level solver                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class AlphaZeroConfig:
    iterations: int = 20
    episodes_per_iter: int = 8
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    temperature: float = 1.0
    temperature_drop_step: Optional[int] = None
    replay_buffer_size: int = 4096
    trunk_channels: int = 32
    num_blocks: int = 3


class AlphaZeroSolver:
    def __init__(
        self,
        env: DeterministicGridWorldEnv,
        params: EmpoParameter,
        config: AlphaZeroConfig,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.params = params
        self.config = config
        self.device = device or torch.device("cpu")
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = PolicyValueCNN(
            in_channels=len(CHANNEL_NAMES),
            board_width=env.width,
            board_height=env.height,
            trunk_channels=config.trunk_channels,
            num_blocks=config.num_blocks,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.replay: list[TrainingExample] = []
        self.history: list[dict] = []

    def mcts(self, num_simulations: Optional[int] = None) -> MCTS:
        cfg = self.config.mcts
        if num_simulations is not None:
            cfg = MCTSConfig(
                num_simulations=num_simulations,
                c_puct=cfg.c_puct,
                dirichlet_alpha=cfg.dirichlet_alpha,
                root_noise_frac=cfg.root_noise_frac,
                rollout_temperature=cfg.rollout_temperature,
                fpu_mode=cfg.fpu_mode,
                fpu_reduction=cfg.fpu_reduction,
            )
        return MCTS(self.env, self.params, self.net, cfg, self.device)

    def fit(self, start_state: GridWorldState) -> None:
        mcts = self.mcts()
        for it in range(self.config.iterations):
            iter_examples: list[TrainingExample] = []
            iter_V_r: list[float] = []
            for _ in range(self.config.episodes_per_iter):
                examples, _, _ = self_play_episode(
                    self.env,
                    self.params,
                    mcts,
                    start_state,
                    temperature=self.config.temperature,
                    temperature_drop_step=self.config.temperature_drop_step,
                    rng=self.rng,
                )
                iter_examples.extend(examples)
                if examples:
                    iter_V_r.append(examples[0].value)
            self.replay.extend(iter_examples)
            if len(self.replay) > self.config.replay_buffer_size:
                self.replay = self.replay[-self.config.replay_buffer_size :]

            pi_loss, v_loss = self._train_epochs()
            best_V = max(iter_V_r) if iter_V_r else float("nan")
            mean_V = float(np.mean(iter_V_r)) if iter_V_r else float("nan")
            self.history.append(
                {
                    "iteration": it,
                    "policy_loss": pi_loss,
                    "value_loss": v_loss,
                    "best_V_r": best_V,
                    "mean_V_r": mean_V,
                    "buffer_size": len(self.replay),
                }
            )

    def _train_epochs(self) -> tuple[float, float]:
        if not self.replay:
            return 0.0, 0.0
        cfg = self.config.train
        last_pi, last_v = 0.0, 0.0
        for _ in range(cfg.epochs_per_iter):
            idx = self.rng.permutation(len(self.replay))
            for start in range(0, len(self.replay), cfg.batch_size):
                batch = [self.replay[i] for i in idx[start : start + cfg.batch_size]]
                if not batch:
                    continue
                last_pi, last_v = train_step(
                    self.net, self.optimizer, batch, cfg, self.device
                )
        return last_pi, last_v

    def greedy_policy(self, num_simulations: int = 256) -> "GreedyMCTSPolicy":
        return GreedyMCTSPolicy(self.mcts(num_simulations=num_simulations))


@dataclass
class GreedyMCTSPolicy:
    mcts: MCTS

    def __call__(self, state: GridWorldState) -> int:
        root = self.mcts.run(state, add_root_noise=False)
        return int(np.argmax(root.N))
