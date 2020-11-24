from pickle import dumps, loads

import numpy as np
from gym import Wrapper


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

        for move in range(self.env.action_space.n):
            snapshot = self.get_snapshot()

            obs, rew, done, info = self.env.step(move)
            children.append(obs)
            rewards.append(rew)
            dones.append(done)

            self.load_snapshot(snapshot)

        return {
            "children": np.stack(children),
            "reward": np.array(rewards),
            "done": np.array(dones),
        }

    def get_snapshot(self):
        return dumps(self.env)

    def load_snapshot(self, snapshot):
        self.env = loads(snapshot)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
