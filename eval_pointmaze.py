"""
Evaluate actor and critic of a trained DIME model on PointMaze.

Produces three artefacts (saved in the Hydra output dir):
  q_landscape.npy      Q-values over a (pos × vel) grid under the policy
  q_act.npy            Q(s, a) over a fixed pos-grid × dense action grid
  acts_landscape.npy   Actions sampled at each grid point

Usage:
    python eval_pointmaze.py --config-name eval_pointmaze \\
        eval.model_path=./models eval.model_steps=1000000

    # override env or grid resolution:
    python eval_pointmaze.py --config-name eval_pointmaze \\
        env_name=pointmaze/medium-sparse-v0 eval.pos_grid_res=30
"""
import os
import jax
import hydra
import wandb
import traceback
import numpy as np
import jax.numpy as jnp

from functools import partial
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common.envs.pointmaze import register_pointmaze_envs, register_custom_pointmaze_env
from diffusion.diffusion_policy import DiffPol
from diffusion.dime import DIME


# ─────────────────────────────────────────────────────────────────────────────
# JIT-compiled primitives
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=["sampler"])
def _sample_actions(actor_state, obs, key, sampler):
    """Single-key action sample from the diffusion actor."""
    actions, *_ = DiffPol.sample_action(
        actor_state, actor_state.params, obs, key, sampler
    )
    return actions  # (batch, a_dim)


@partial(jax.jit, static_argnames=["v_min", "v_max", "n_atoms"])
def _eval_q(qf_state, obs, actions, key, v_min, v_max, n_atoms):
    """Expected Q-value (mean over critics) for obs/action pairs."""
    z_atoms = jnp.linspace(v_min, v_max, n_atoms)
    q_dist = qf_state.apply_fn(
        {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
        obs,
        actions,
        rngs={"dropout": key},
        train=False,
    )  # (n_critics, batch, n_atoms)
    return (q_dist * z_atoms).sum(-1).mean(0)  # (batch,)


def _mean_q_over_samples(model, obs, key, n_samples):
    """Estimate V(s) ≈ E_π[Q(s,a)] by averaging Q over n_samples action draws."""
    sample_keys = jax.random.split(key, n_samples)
    # vmap over different random keys → (n_samples, batch, a_dim)
    all_acts = jax.vmap(
        lambda k: _sample_actions(model.policy.actor_state, obs, k, model.policy.sampler)
    )(sample_keys)
    eval_keys = jax.random.split(key, n_samples)
    # vmap Q evaluation over the n_samples draws → (n_samples, batch)
    all_q = jax.vmap(
        lambda a, k: _eval_q(
            model.policy.qf_state, obs, a, k,
            model.cfg.alg.critic.v_min,
            model.cfg.alg.critic.v_max,
            model.cfg.alg.critic.n_atoms,
        )
    )(all_acts, eval_keys)
    return all_q.mean(0), all_acts  # (batch,), (n_samples, batch, a_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Grid evaluations  (mirrors maze_eval_main.py lines 373-448)
# ─────────────────────────────────────────────────────────────────────────────

def eval_value_landscape(model, goal, key, cfg_eval):
    """
    Q-value under the policy over a (x, y, vx, vy) × goal grid.

    Returns
    -------
    positions  : (N, 2)  [x, y] of every query point
    q_values   : (N,)    estimated V(s) at each point
    acts       : (n_samples, N, a_dim)  sampled actions
    """
    res = cfg_eval.pos_grid_res
    pr = cfg_eval.pos_range
    vr = cfg_eval.vel_range
    nv = cfg_eval.n_vel_grid

    coords = jnp.linspace(-pr, pr, res)
    vel    = jnp.linspace(-vr, vr, nv)

    xs, ys = jnp.meshgrid(coords, coords)
    xv, yv = jnp.meshgrid(vel, vel)
    xs, ys = xs.ravel(), ys.ravel()
    xv, yv = xv.ravel(), yv.ravel()

    N, M = xs.shape[0], xv.shape[0]
    # Broadcast: each position paired with every velocity
    obs = jnp.stack([
        jnp.repeat(xs, M),               # x
        jnp.repeat(ys, M),               # y
        jnp.tile(xv, N),                 # vx
        jnp.tile(yv, N),                 # vy
        jnp.full(N * M, goal[0]),        # goal_x
        jnp.full(N * M, goal[1]),        # goal_y
    ], axis=-1)  # (N*M, 6)

    key, q_key = jax.random.split(key)
    q_values, acts = _mean_q_over_samples(
        model, obs, q_key, cfg_eval.n_action_samples
    )

    positions = jnp.stack([jnp.repeat(xs, M), jnp.repeat(ys, M)], axis=-1)
    return np.array(positions), np.array(q_values), np.array(acts)


def eval_q_action_grid(model, goal, key, cfg_eval):
    """
    Q(s, a) over a position grid (zero velocity) × a dense action grid.
    Mirrors the q_act evaluation in maze_eval_main.py lines 427-448.

    Returns
    -------
    q_act      : (N_pos, N_act)  Q-value at each (pos, action) pair
    acts_grid  : (N_act, a_dim)  the action grid used
    positions  : (N_pos, 2)      the position grid used
    """
    res = cfg_eval.pos_grid_res
    pr  = cfg_eval.pos_range
    nag = cfg_eval.n_act_grid

    coords = jnp.linspace(-pr, pr, res)
    xs, ys = jnp.meshgrid(coords, coords)
    xs, ys = xs.ravel(), ys.ravel()
    N_pos = xs.shape[0]

    obs = jnp.stack([
        xs, ys,
        jnp.zeros_like(xs), jnp.zeros_like(ys),
        jnp.full(N_pos, goal[0]),
        jnp.full(N_pos, goal[1]),
    ], axis=-1)  # (N_pos, 6)

    a_vals = jnp.linspace(-1, 1, nag)
    xa, ya = jnp.meshgrid(a_vals, a_vals)
    acts_grid = jnp.stack([xa.ravel(), ya.ravel()], axis=-1)  # (N_act, 2)
    N_act = acts_grid.shape[0]

    key, k = jax.random.split(key)
    dropout_keys = jax.random.split(k, N_act)

    # Q(obs, acts_grid[j]) for all positions, vmapped over actions
    q_act = jax.vmap(
        lambda a, dk: _eval_q(
            model.policy.qf_state,
            obs,
            jnp.broadcast_to(a[None], (N_pos, a.shape[0])),
            dk,
            model.cfg.alg.critic.v_min,
            model.cfg.alg.critic.v_max,
            model.cfg.alg.critic.n_atoms,
        ),
        in_axes=(0, 0),
    )(acts_grid, dropout_keys)  # (N_act, N_pos)

    positions = jnp.stack([xs, ys], axis=-1)
    return np.array(q_act.T), np.array(acts_grid), np.array(positions)


# ─────────────────────────────────────────────────────────────────────────────
# Episode rollouts  (mirrors get_episode_rewards in maze_eval_main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_episodes(model, env, key, n_episodes):
    """
    Roll out n_episodes with the deterministic (mean) policy action.

    Returns a dict with per-episode rewards, positions, Q-values, and
    aggregated success_rate / mean_return.
    """
    all_rewards, all_positions, all_q_values, all_successes = [], [], [], []

    for ep in tqdm(range(n_episodes), desc="eval episodes"):
        key, ep_key = jax.random.split(key)
        obs, info = env.reset(seed=int(np.array(ep_key)[0]))

        rewards, positions, q_vals = [], [], []
        done = False

        while not done:
            obs_j = jnp.array(obs[None])  # (1, obs_dim)
            key, act_key, q_key = jax.random.split(key, 3)

            action = _sample_actions(
                model.policy.actor_state, obs_j, act_key, model.policy.sampler
            )  # (1, a_dim)

            q = _eval_q(
                model.policy.qf_state, obs_j, action, q_key,
                model.cfg.alg.critic.v_min,
                model.cfg.alg.critic.v_max,
                model.cfg.alg.critic.n_atoms,
            )  # (1,)

            positions.append(obs[:2].tolist())
            q_vals.append(float(q[0]))

            obs, reward, terminated, truncated, info = env.step(np.array(action[0]))
            rewards.append(float(reward))
            done = terminated or truncated

        all_rewards.append(rewards)
        all_positions.append(positions)
        all_q_values.append(q_vals)
        all_successes.append(bool(info.get("is_success", False)))

    return {
        "rewards":             all_rewards,
        "positions":           all_positions,
        "q_values":            all_q_values,
        "successes":           all_successes,
        "mean_return":         float(np.mean([sum(r) for r in all_rewards])),
        "success_rate":        float(np.mean(all_successes)),
        "mean_episode_length": float(np.mean([len(r) for r in all_rewards])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(cfg):
    """Create a DIME instance from config (no callbacks, no training)."""
    import gymnasium as gym

    training_env = gym.make(cfg.env_name)
    model = DIME(
        "MlpPolicy",
        env=training_env,
        model_save_path=None,
        save_every_n_steps=1,
        cfg=cfg,
        tensorboard_log=None,
    )
    return model, training_env


@hydra.main(version_base=None, config_path="configs", config_name="eval_pointmaze")
def main(cfg: DictConfig) -> None:
    try:
        _run(cfg)
    except Exception:
        traceback.print_exc()


def _run(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    register_pointmaze_envs()

    env_name = cfg.env_name
    if cfg.env_name.split('/')[1] == 'custom':
        m = cfg.maze
        env_name = register_custom_pointmaze_env(
            rows=m.rows,
            cols=m.cols,
            narrow_passage=m.narrow_passage,
            init_radius=m.init_radius,
            reward_type=m.reward_type,
            max_episode_steps=m.max_episode_steps,
        )
    # Temporarily patch cfg so _build_model picks up the resolved name
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(cfg, {"env_name": env_name})

    model, env = _build_model(cfg)

    # ── load checkpoint if provided ───────────────────────────────────────
    ecfg = cfg.eval
    if ecfg.model_path is not None and ecfg.model_steps is not None:
        model.load_model(ecfg.model_path, ecfg.model_steps, ecfg.model_steps)
        print(f"Loaded checkpoint from {ecfg.model_path} @ step {ecfg.model_steps}")
    else:
        print("No checkpoint specified — evaluating randomly-initialised model")

    key = jax.random.PRNGKey(cfg.seed)

    # ── sample a goal to fix for landscape evaluations ────────────────────
    key, goal_key = jax.random.split(key)
    obs0, _ = env.reset(seed=int(np.array(goal_key)[0]))
    goal = obs0[4:6]  # desired_goal is dims [4:6] in our 6-dim wrapper obs
    print(f"Goal for landscape evals: {goal}")

    # ── Q-value landscape  (q.npy / act.npy in maze_eval_main.py) ────────
    print("Computing Q-value landscape...")
    key, lk = jax.random.split(key)
    positions, q_landscape, acts_landscape = eval_value_landscape(model, goal, lk, ecfg)
    np.save("q_landscape.npy", q_landscape)
    np.save("positions_landscape.npy", positions)
    np.save("acts_landscape.npy", acts_landscape)
    print(f"  Q landscape  min={q_landscape.min():.3f}  "
          f"max={q_landscape.max():.3f}  mean={q_landscape.mean():.3f}")

    # ── Q vs action grid  (q_act.npy in maze_eval_main.py) ───────────────
    print("Computing Q-action grid...")
    key, ak = jax.random.split(key)
    q_act, acts_grid, pos_grid = eval_q_action_grid(model, goal, ak, ecfg)
    np.save("q_act.npy", q_act)
    np.save("acts_grid.npy", acts_grid)
    np.save("pos_grid.npy", pos_grid)
    print(f"  Q-action grid  min={q_act.min():.3f}  max={q_act.max():.3f}")

    # ── rollout episodes ──────────────────────────────────────────────────
    print(f"Running {ecfg.n_episodes} evaluation episodes...")
    key, ek = jax.random.split(key)
    results = run_eval_episodes(model, env, ek, ecfg.n_episodes)
    np.save("eval_results.npy", results, allow_pickle=True)
    print(f"  mean_return={results['mean_return']:.2f}  "
          f"success_rate={results['success_rate']:.1%}  "
          f"mean_ep_len={results['mean_episode_length']:.1f}")

    # ── wandb logging ─────────────────────────────────────────────────────
    if cfg.wandb["activate"]:
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=cfg.wandb["project"],
            group=cfg.wandb.get("group", f"eval_{cfg.env_name}"),
            name=f"eval_seed{cfg.seed}",
            config=wandb_config,
        )

        art = wandb.Artifact("eval_arrays", type="dataset")
        for fname in ["q_landscape.npy", "positions_landscape.npy",
                      "acts_landscape.npy", "q_act.npy", "acts_grid.npy",
                      "pos_grid.npy"]:
            if os.path.exists(fname):
                art.add_file(fname)
        wandb.log_artifact(art)

        wandb.log({
            "eval/mean_return":         results["mean_return"],
            "eval/success_rate":        results["success_rate"],
            "eval/mean_episode_length": results["mean_episode_length"],
            "eval/q_landscape_mean":    float(q_landscape.mean()),
            "eval/q_landscape_max":     float(q_landscape.max()),
            "eval/q_landscape_min":     float(q_landscape.min()),
            **{f"eval/return/{i}": sum(r) for i, r in enumerate(results["rewards"])},
            **{f"eval/success/{i}": float(s) for i, s in enumerate(results["successes"])},
            **{f"eval/q_start/{i}": vals[0] for i, vals in enumerate(results["q_values"])},
            **{f"eval/q_end/{i}":   vals[-1] for i, vals in enumerate(results["q_values"])},
        })
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
