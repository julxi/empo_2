import numpy as np
import pytest
import torch

import grid_world as env
import grid_world.empo_eval as empo_eval
from grid_world.solvers import alphazero as az
from grid_world.solvers import backward_induction as solvino


def _fair_population(size: int):
    h1 = [(lambda x, i=i: float(x.state.object[0] <= i)) for i in range(size)]
    h2 = [(lambda x, i=i: float(x.state.object[0] >= i)) for i in range(size)]
    return [h1, h2]


def _params():
    return env.EmpoParameter(
        gamma_r=1, beta_r=1, gamma_h=1, zeta=2, xi=1, eta=1
    )


def test_encode_obs_shape_and_planes() -> None:
    size, max_steps = 5, 10
    walls = frozenset({(0, 1), (2, 3)})
    func_env = env.MovingBoxEnv(
        env.GridWorldLayout(width=size, height=size, max_steps=max_steps, walls=walls), []
    )
    state = env.GridWorldState(robot=(0, 0), object=(4, 4), step=5)
    obs = func_env.observation(state)
    enc = az.encode_obs(obs)

    assert enc.shape == (len(az.CHANNEL_NAMES), size, size)
    assert enc.dtype == np.float32

    # One-hot entity planes.
    assert enc[0].sum() == 1.0 and enc[0, 0, 0] == 1.0
    assert enc[1].sum() == 1.0 and enc[1, 4, 4] == 1.0

    # Wall plane matches the layout walls exactly.
    assert enc[2].sum() == len(walls)
    for (x, y) in walls:
        assert enc[2, x, y] == 1.0

    # Step plane is constant at step / max_steps.
    assert np.allclose(enc[3], 5 / 10)


def test_policy_value_cnn_forward() -> None:
    net = az.PolicyValueCNN(
        in_channels=len(az.CHANNEL_NAMES),
        board_width=5,
        board_height=5,
        trunk_channels=8,
        num_blocks=1,
    )
    x = torch.randn(3, len(az.CHANNEL_NAMES), 5, 5)
    logits, value = net(x)
    assert logits.shape == (3, 4)
    assert value.shape == (3,)


def test_mcts_finds_optimum_with_random_network() -> None:
    """With many simulations the trajectory-based backup converges to the
    optimal V_r at the root, even with an untrained network."""
    size = 5
    func_env = env.MovingBoxEnv(
        env.GridWorldLayout(width=size, height=size, max_steps=10, walls=frozenset()),
        _fair_population(size),
    )
    start = env.GridWorldState(robot=(0, 0), object=(1, 0), step=0)
    params = _params()

    solver = solvino.BackwardInductionSolver(func_env, params)
    solver.solve(start)
    optimal_action_Q = solver.Q_r[start]
    optimal_V = solver.V_r[start]

    cfg = az.AlphaZeroConfig(
        iterations=0,
        episodes_per_iter=0,
        mcts=az.MCTSConfig(num_simulations=512, c_puct=2.0, root_noise_frac=0.1),
    )
    solver_az = az.AlphaZeroSolver(func_env, params, cfg, seed=0)
    mcts = solver_az.mcts(num_simulations=512)
    root = mcts.run(start, add_root_noise=True, rng=np.random.default_rng(0))

    chosen = int(np.argmax(root.N))
    chosen_state = func_env.transition(start, chosen)
    realised_V_r = optimal_action_Q[env.Action(chosen)]
    # Every visited action's Q should be near the true V_r(child) under optimal play.
    # The realised V_r at the chosen child is the same as Q_r(start, action) when γ_r=1.
    assert root.Q()[chosen] == pytest.approx(realised_V_r, abs=0.5)
    # And one of the realised first-action V_r values is the true optimum at start.
    # (V_r* at start = max U_r + γ V_r(child); for this instance all actions tie.)
    assert chosen_state in solver.V_r
    assert solver.V_r[chosen_state] + solver.U_r[start] == pytest.approx(optimal_V)


def test_visit_policy_normalises() -> None:
    node = az.MCTSNode(state=None, terminal=False)
    node.N = np.array([3, 1, 0, 6], dtype=np.int64)
    pi_t1 = az.visit_policy(node, 1.0)
    assert pi_t1.sum() == pytest.approx(1.0)
    assert pi_t1[3] > pi_t1[0] > pi_t1[1] > pi_t1[2]
    pi_t0 = az.visit_policy(node, 0.0)
    assert pi_t0.sum() == pytest.approx(1.0)
    assert int(np.argmax(pi_t0)) == 3
    assert pi_t0[3] == pytest.approx(1.0)


def test_self_play_episode_targets_match_evaluator() -> None:
    size = 5
    func_env = env.MovingBoxEnv(
        env.GridWorldLayout(width=size, height=size, max_steps=10, walls=frozenset()),
        _fair_population(size),
    )
    start = env.GridWorldState(robot=(0, 0), object=(1, 0), step=0)
    params = _params()
    cfg = az.AlphaZeroConfig(
        iterations=0,
        episodes_per_iter=0,
        mcts=az.MCTSConfig(num_simulations=8),
    )
    solver = az.AlphaZeroSolver(func_env, params, cfg, seed=0)
    mcts = solver.mcts()
    examples, states, actions = az.self_play_episode(
        func_env, params, mcts, start, temperature=1.0,
        rng=np.random.default_rng(0),
    )
    traj = empo_eval.evaluate_trajectory(func_env, params, states, actions)
    assert len(examples) == len(actions)
    for t, ex in enumerate(examples):
        assert ex.value == pytest.approx(traj.V_r[t])


def test_alphazero_solver_reaches_optimal_V_r() -> None:
    """End-to-end: after a short training run, AlphaZero's greedy policy
    achieves the optimal V_r given by backward induction."""
    size = 5
    func_env = env.MovingBoxEnv(
        env.GridWorldLayout(width=size, height=size, max_steps=10, walls=frozenset()),
        _fair_population(size),
    )
    start = env.GridWorldState(robot=(0, 0), object=(1, 0), step=0)
    params = _params()

    exact = solvino.BackwardInductionSolver(func_env, params)
    exact.solve(start)

    cfg = az.AlphaZeroConfig(
        iterations=8,
        episodes_per_iter=3,
        mcts=az.MCTSConfig(num_simulations=48),
        temperature_drop_step=8,
    )
    solver = az.AlphaZeroSolver(func_env, params, cfg, seed=0)
    solver.fit(start)

    greedy = solver.greedy_policy(num_simulations=256)
    traj = empo_eval.evaluate_policy(func_env, params, greedy, start)
    # Tight tolerance: trajectory-eval V_r should match the exact optimum.
    assert traj.V_r[0] == pytest.approx(exact.V_r[start], abs=1e-6)
    # The terminal box should be at the middle row for the fair-box instance.
    assert traj.states[-1].object[0] == (size - 1) // 2
