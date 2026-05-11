import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PointMazeMultiGoalWrapper(gym.Wrapper):
    """Multi-goal PointMaze: goal stripped from obs, reward vs nearest goal.

    Observation = [x, y, vx, vy] (4-dim, no goal).
    Reward = 0.0 if within threshold of ANY goal, else -1.0.
    Episode never terminates early (continuing task); only truncates on timeout.
    """

    def __init__(self, env, threshold: float = 0.45):
        super().__init__(env)
        self.threshold = threshold
        self._goal_locations = np.array(env.unwrapped.maze.unique_goal_locations)  # (N, 2)
        obs_dim = env.observation_space["observation"].shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64,
        )

    def _compute_reward(self, achieved_pos):
        dists = np.linalg.norm(self._goal_locations - achieved_pos, axis=-1)
        return 0.0 if dists.min() <= self.threshold else -1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["observation"].copy(), info

    def step(self, action):
        obs, _, _terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs["achieved_goal"])
        info["is_success"] = reward == 0.0
        if "final_observation" in info and info["final_observation"] is not None:
            fo = info["final_observation"]
            info["final_observation"] = fo["observation"].copy()
        return obs["observation"].copy(), reward, False, truncated, info


class PointMazeGCWrapper(gym.Wrapper):
    """Flattens PointMaze Dict obs (observation + desired_goal) into a single Box.

    reward_offset is added to every reward.  Use -1 for sparse envs to shift
    the {0, 1} signal to {-1, 0}, which is easier to learn from and matches
    the convention used in the rest of the DIME codebase.
    """

    def __init__(self, env, reward_offset: float = 0.0):
        super().__init__(env)
        self.reward_offset = reward_offset
        obs_dim = env.observation_space["observation"].shape[0]
        goal_dim = env.observation_space["desired_goal"].shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim + goal_dim,),
            dtype=np.float64,
        )

    @staticmethod
    def _flat(obs_dict):
        return np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flat(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "final_observation" in info and info["final_observation"] is not None:
            fo = info["final_observation"]
            info["final_observation"] = np.concatenate(
                [fo["observation"], fo["desired_goal"]]
            )
        return self._flat(obs), reward + self.reward_offset, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Maze map generation  (ported from mad-td-dev D4RLMazeGC.make_map)
# ─────────────────────────────────────────────────────────────────────────────

def make_map(rows, cols, narrow_passage=False, init_radius=0):
    """
    Build a maze_map list accepted by gymnasium-robotics PointMaze envs.

    Cells are:
      0   open
      1   wall
      'G' goal (sampled by Diverse_GR env variants)
      'R' reset / start position

    Parameters
    ----------
    rows, cols      : grid dimensions (number of open rooms, not counting walls)
    narrow_passage  : wide open grid vs. rooms connected by narrow doorways
    init_radius     : half-side of the square reset region around the centre
                      (0 means no explicit R cells → env picks randomly)
    """
    if narrow_passage:
        # Each room is 3 open cells wide with a single-cell doorway between rooms.
        # Map width = 4*cols + 1,  map height = 4*(rows-1) + 3 + 1
        maze_map = [[1 for _ in range(4 * cols + 1)]]
        for _ in range(rows - 1):
            row = [1]
            for _ in range(cols):
                row.extend([0, 0, 0, 1])
            maze_map.append(row)

            row = [1]
            for _ in range(4 * cols - 1):
                row.append(0)
            row.append(1)
            maze_map.append(row)

            row = [1]
            for _ in range(cols):
                row.extend([0, 0, 0, 1])
            maze_map.append(row)

            row = [1]
            for _ in range(cols):
                row.extend([1, 0, 1, 1])
            maze_map.append(row)

        row = [1]
        for _ in range(cols):
            row.extend([0, 0, 0, 1])
        maze_map.append(row)

        row = [1]
        for _ in range(4 * cols - 1):
            row.append(0)
        row.append(1)
        maze_map.append(row)

        row = [1]
        for _ in range(cols):
            row.extend([0, 0, 0, 1])
        maze_map.append(row)
        maze_map.append([1 for _ in range(4 * cols + 1)])

        # Goals at the four near-corner open cells
        maze_map[2][2]   = "G"
        maze_map[2][-3]  = "G"
        maze_map[-3][2]  = "G"
        maze_map[-3][-3] = "G"

    else:
        # Simple grid: each pair of adjacent rooms is separated by a single wall
        # with one open cell gap.  Map width = 2*cols + 1.
        maze_map = [[1 for _ in range(2 * cols + 1)]]
        for _ in range(rows - 1):
            row = [1]
            for _ in range(cols * 2 - 1):
                row.append(0)
            row.append(1)
            maze_map.append(row)

            row = [1]
            for _ in range(cols):
                row.extend([0, 1])
            maze_map.append(row)

        row = [1]
        for _ in range(cols * 2 - 1):
            row.append(0)
        row.append(1)
        maze_map.append(row)
        maze_map.append([1 for _ in range(2 * cols + 1)])

        # Goals in the four corner open cells
        maze_map[1][1]   = "G"
        maze_map[1][-2]  = "G"
        maze_map[-2][1]  = "G"
        maze_map[-2][-2] = "G"

    # Mark reset region around the centre
    if init_radius > 0:
        center_row = len(maze_map) // 2
        center_col = len(maze_map[0]) // 2
        for i in range(
            max(0, center_row - init_radius),
            min(center_row + init_radius + 1, len(maze_map)),
        ):
            for j in range(
                max(0, center_col - init_radius),
                min(center_col + init_radius + 1, len(maze_map[0])),
            ):
                if maze_map[i][j] == 0:
                    maze_map[i][j] = "R"

    return maze_map


# ─────────────────────────────────────────────────────────────────────────────
# Registration helpers
# ─────────────────────────────────────────────────────────────────────────────

# (base_gymnasium_robotics_id, max_episode_steps, reward_offset)
_VARIANTS = {
    "pointmaze/umaze-sparse-v0":  ("PointMaze_UMaze-v3",                  300, -1.0),
    "pointmaze/umaze-dense-v0":   ("PointMaze_UMazeDense-v3",              300,  0.0),
    "pointmaze/medium-sparse-v0": ("PointMaze_Medium_Diverse_GR-v3",       600, -1.0),
    "pointmaze/medium-dense-v0":  ("PointMaze_Medium_Diverse_GRDense-v3",  600,  0.0),
    "pointmaze/large-sparse-v0":  ("PointMaze_Large_Diverse_GR-v3",        800, -1.0),
    "pointmaze/large-dense-v0":   ("PointMaze_Large_Diverse_GRDense-v3",   800,  0.0),
}

# Multi-goal variants: (base_id, max_episode_steps, threshold)
_MULTI_VARIANTS = {
    "pointmaze/medium-multi-v0": ("PointMaze_Medium_Diverse_GR-v3",  600, 0.45),
    "pointmaze/large-multi-v0":  ("PointMaze_Large_Diverse_GR-v3",   800, 0.45),
}


def register_pointmaze_envs():
    """Register wrapped PointMaze env IDs into gymnasium. Safe to call multiple times."""
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    for custom_id, (base_id, max_steps, reward_offset) in _VARIANTS.items():
        if custom_id not in gym.registry:
            gym.register(
                id=custom_id,
                entry_point=lambda b=base_id, ms=max_steps, ro=reward_offset: (
                    PointMazeGCWrapper(gym.make(b, max_episode_steps=ms), reward_offset=ro)
                ),
                max_episode_steps=None,  # TimeLimit already applied inside
            )

    for custom_id, (base_id, max_steps, threshold) in _MULTI_VARIANTS.items():
        if custom_id not in gym.registry:
            gym.register(
                id=custom_id,
                entry_point=lambda b=base_id, ms=max_steps, t=threshold: (
                    PointMazeMultiGoalWrapper(gym.make(b, max_episode_steps=ms), threshold=t)
                ),
                max_episode_steps=None,
            )


def register_custom_pointmaze_env(
    rows=3,
    cols=3,
    narrow_passage=False,
    init_radius=0,
    reward_type="sparse",
    max_episode_steps=800,
    multi_goal=False,
):
    """
    Generate a maze map from the given parameters, register it as a gymnasium env,
    and return its ID.  Safe to call multiple times with the same arguments.

    The returned ID is deterministic from the arguments, so the same call always
    produces the same ID and the env is only registered once.

    multi_goal=True: uses PointMazeMultiGoalWrapper (obs=[x,y,vx,vy], reward vs nearest goal).
    multi_goal=False: uses PointMazeGCWrapper (obs=[x,y,vx,vy,gx,gy], reward_type controls signal).
    """
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    np_tag = "np" if narrow_passage else "op"
    mg_tag = "mg" if multi_goal else reward_type
    env_id = (
        f"pointmaze/custom_{rows}x{cols}_{np_tag}_ir{init_radius}_{mg_tag}-v0"
    )

    if env_id not in gym.registry:
        maze_map = make_map(rows, cols, narrow_passage, init_radius)
        # Multi-goal always uses the sparse base (reward is recomputed in wrapper anyway)
        dense_suffix = "Dense" if (not multi_goal and reward_type == "dense") else ""
        base_id = f"PointMaze_Large_Diverse_GR{dense_suffix}-v3"

        if multi_goal:
            # Threshold: narrow passages need a larger capture radius
            threshold = 1.5 if narrow_passage else 0.45
            gym.register(
                id=env_id,
                entry_point=lambda m=maze_map, b=base_id, ms=max_episode_steps, t=threshold: (
                    PointMazeMultiGoalWrapper(gym.make(b, maze_map=m, max_episode_steps=ms), threshold=t)
                ),
                max_episode_steps=None,
            )
        else:
            reward_offset = 0.0 if reward_type == "dense" else -1.0
            gym.register(
                id=env_id,
                entry_point=lambda m=maze_map, b=base_id, ms=max_episode_steps, ro=reward_offset: (
                    PointMazeGCWrapper(gym.make(b, maze_map=m, max_episode_steps=ms), reward_offset=ro)
                ),
                max_episode_steps=None,
            )

    return env_id
