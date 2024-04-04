import random

import gymnasium as gym
import numpy as np
from ale_py import ALEInterface, LoggerMode
from ray.rllib.env.wrappers import atari_wrappers as wrappers
from skimage import transform
from skimage.util import img_as_ubyte

ALEInterface.setLoggerMode(LoggerMode.Error)


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        # pylint: disable=unused-argument

        self.end_pos = 10
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            0.0, self.end_pos, shape=(1,), dtype=np.float32
        )
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(0)

    def reset(self, *, seed=None, options=None):
        self.seed(seed)

        self.cur_pos = 0
        return [self.cur_pos], {}

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        # Produce a random reward when we reach the goal.
        return [self.cur_pos], random.random() * 2 if done else -0.1, done, False, {}

    def seed(self, seed):
        random.seed(seed)

    def render(self):
        return None


class AtariEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config):
        self._fix_seed = config.get("fix_seed", False)

        self._env = gym.make(config["pg_name"])
        wrap = config.get("wrap", {})
        self._env = wrap_deepmind(self._env, **wrap)

        self.video_dir = config.get("video_dir")
        self.episodes = 0
        self.time_steps = 0
        self.obs_raw = None

        self.action_space = self._env.action_space
        self._config = config

    @property
    def observation_space(self):
        return self._env.observation_space

    def step(self, action):
        step_result = self._env.step(action)
        obs = self.process_obs(step_result[0])
        self.obs_raw = obs
        return obs, *step_result[1:]

    def reset(self, *, seed=None, options=None):
        if self._fix_seed:
            seed = 0

        random.seed(seed)
        np.random.seed(seed)

        obs, info = self._env.reset(seed=seed, options=options)
        obs = self.process_obs(obs)
        self.obs_raw = obs
        self.episodes += 1

        return obs, info

    def render(self):
        # if self.video_dir is None:
        #     return None

        assert self.render_mode == "rgb_array"

        stack_depth = self.obs_raw.shape[2] // 3
        img = np.concatenate(
            [self.obs_raw[:, :, i * 3 : (i + 1) * 3] for i in range(stack_depth)], 1
        )
        return img

    def close(self):
        self._env.close()

    def process_obs(self, obs):
        return obs


class CropEnv(gym.ObservationWrapper):
    def __init__(self, env, crop):
        super().__init__(env)

        # crop = (top, bottom, left, right)
        if isinstance(crop, int):
            crop = [crop] * 4

        orig = env.observation_space.shape
        cropped_shape = (
            orig[0] - (crop[0] + crop[1]),
            orig[1] - (crop[2] + crop[3]),
            orig[2],
        )
        assert (
            cropped_shape[0] > 0 and cropped_shape[1] > 1
        ), f"Crop too big, resulting shape: {cropped_shape}"

        self._crop = (
            crop[0],
            orig[0] - crop[1],
            crop[2],
            orig[1] - crop[3],
        )

        self.observation_space = gym.spaces.Box(
            low=np.amin(env.observation_space.low),
            high=np.amax(env.observation_space.high),
            shape=cropped_shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return observation[self._crop[0] : self._crop[1], self._crop[2] : self._crop[3]]


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        super().__init__(env)

        if isinstance(size, int):
            size = [size] * 2

        self._size = size

        self.observation_space = gym.spaces.Box(
            low=np.amin(env.observation_space.low),
            high=np.amax(env.observation_space.high),
            shape=(size[0], size[1], env.observation_space.shape[-1]),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return img_as_ubyte(
            transform.resize(
                observation,
                self._size,
                order=0,
                anti_aliasing=False,
            )
        )


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, truncated, info


def wrap_deepmind(env, skip=0, stack=4, resize=None, crop=None):
    """Configure environment for DeepMind-style Atari. See:
    https://github.com/ray-project/ray/blob/master/rllib/env/wrappers/atari_wrappers.py

    For default env settings see:
    https://www.gymlibrary.dev/environments/atari/

    Note: ALE v5 envs have frameskip by default.
    """

    env = wrappers.MonitorEnv(env)
    env = wrappers.NoopResetEnv(env, noop_max=30)
    if skip > 0:
        env = SkipEnv(env, skip=skip)  # Don't use max, only skip
        # env = wrappers.MaxAndSkipEnv(env, skip=skip)

    env = wrappers.EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = wrappers.FireResetEnv(env)

    if crop is not None:
        env = CropEnv(env, crop)

    if resize is not None:
        # Don't warp, resize instead
        # env = wrappers.WarpFrame(env, dim)
        env = ResizeEnv(env, resize)

    if stack > 1:
        env = wrappers.FrameStack(env, stack)

    # env = wrappers.NormalizedImageEnv(env)

    return env
