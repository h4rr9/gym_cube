from collections import namedtuple
from pickle import dumps, loads

import numpy as np
from gym.core import Wrapper

# a container for get_result function below. Works just like tuple, but prettier
ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "reward", "is_done", "info")
)


def getchildren(env):
    """simple function to return wrapped env"""
    return GetChildren(WithSnapshots(env))


class WithSnapshots(Wrapper):
    def __init__(self, env):
        super(WithSnapshots, self).__init__(env)

    def get_snapshot(self):
        return dumps(self.env)

    def load_snapshot(self, snapshot, render=False):
        self.env = loads(snapshot)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GetChildren(Wrapper):
    """Returns the children of the current state"""

    def __init__(self, env):
        super(GetChildren, self).__init__(env)

    def step(self, action):

        obs, rew, done, info = self.env.step(action)

        childrenInfo = self._get_children_info()
        for key, value in childrenInfo.items():
            info[key] = value

        return obs, rew, done, info

    def _get_children_info(self):

        children = []
        rewards = []
        dones = []

        assert isinstance(
            self.env, WithSnapshots
        ), "Requires WithSnapshots to reuse env states"

        for move in range(self.env.action_space.n):
            snapshot = self.env.get_snapshot()

            obs, rew, done, info = self.env.step(move)
            children.append(obs)
            rewards.append(rew)
            dones.append(done)

            self.env.load_snapshot(snapshot)

        return {
            "children": np.stack(children),
            "reward": np.array(rewards),
            "done": np.array(dones),
        }

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
