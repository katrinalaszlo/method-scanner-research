"""Phase 23 shim: old-gym API over gymnasium v4 MuJoCo envs + the original D4RL v2 HDF5 datasets.
Diffuser was written against gym 0.18 / d4rl: env.reset() -> obs, env.step -> (obs, r, done, info),
env.seed(), env._max_episode_steps, env.get_dataset(), env.get_normalized_score(), env.state_vector().
Deviation from the paper: dynamics run on gymnasium *-v4 (new mujoco bindings) instead of *-v2 (mujoco-py 2.0).
"""
import os
import numpy as np
import gymnasium
import h5py

DATA_DIR = os.environ.get("D4RL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "d4rl_data"))

# D4RL reference scores (d4rl/infos.py, gym_mujoco v2)
REF = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
}
GYMNASIUM_ID = {"halfcheetah": "HalfCheetah-v4", "hopper": "Hopper-v4", "walker2d": "Walker2d-v4"}


class D4RLEnv:
    def __init__(self, name):
        self.name = name
        base = name.split("-")[0].replace("FullObs", "").lower()  # 'HalfCheetahFullObs-v2' (renderer) -> halfcheetah
        self._base = base
        self._env = gymnasium.make(GYMNASIUM_ID[base])
        self._max_episode_steps = self._env.spec.max_episode_steps
        self.max_episode_steps = self._max_episode_steps
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._seed = None
        self.unwrapped = self

    def seed(self, seed=None):
        self._seed = seed

    def reset(self):
        obs, _ = self._env.reset(seed=self._seed)
        self._seed = None
        return obs

    def step(self, action):
        obs, r, terminated, truncated, info = self._env.step(action)
        return obs, r, bool(terminated or truncated), info

    def state_vector(self):
        u = self._env.unwrapped
        return np.concatenate([u.data.qpos.flat, u.data.qvel.flat])

    def get_dataset(self):
        path = os.path.join(DATA_DIR, f"{self.name.replace('-v2', '')}-v2.hdf5".replace("-medium", "_medium"))
        with h5py.File(path, "r") as f:
            return {k: f[k][()] for k in ("observations", "actions", "rewards", "terminals", "timeouts")}

    def get_normalized_score(self, score):
        lo, hi = REF[self._base]
        return (score - lo) / (hi - lo)


def make(name):
    return D4RLEnv(name)


def register(**kwargs):
    pass  # diffuser.environments registers FullObs rendering envs; rendering is disabled
