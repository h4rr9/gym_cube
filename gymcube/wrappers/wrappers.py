import gym

import numpy as np

from gym.core import Wrapper
from pickle import dumps, loads
from collections import namedtuple

# a container for get_result function below. Works just like tuple, but prettier
ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "reward", "is_done", "info")
)


class WithSnapshots(Wrapper):
    """
    Creates a wrapper that supports saving and loading environemnt states.
    Required for planning algorithms.

    This class will have access to the core environment as self.env, e.g.:
    - self.env.reset()           #reset original env
    - self.env.ale.cloneState()  #make snapshot for atari. load with .restoreState()
    - ...

    You can also use reset() and step() directly for convenience.
    - s = self.reset()                   # same as self.env.reset()
    - s, r, done, _ = self.step(action)  # same as self.env.step(action)
    
    Note that while you may use self.render(), it will spawn a window that cannot be pickled.
    Thus, you will need to call self.close() before pickling will work again.
    """

    def get_snapshot(self):
        """
        :returns: environment state that can be loaded with load_snapshot 
        Snapshots guarantee same env behaviour each time they are loaded.
        """
        return dumps(self.env)

    def load_snapshot(self, snapshot, render=False):
        """
        Loads snapshot as current env state.
        Should not change snapshot inplace (in case of doubt, deepcopy).
        """
        self.env = loads(snapshot)

    def get_result(self, snapshot, action):
        """
        A convenience function that 
        - loads snapshot, 
        - commits action via self.step,
        - and takes snapshot again :)

        :returns: next snapshot, next_observation, reward, is_done, info

        Basically it returns next snapshot and everything that env.step would have returned.
        """

        self.load_snapshot(snapshot)

        obs, rew, done, info = self.env.step(action)

        return ActionResult(self.get_snapshot(), obs, rew, done, info)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GetChildren(Wrapper):
    """Returns the children of the current state"""

    def __init__(self, env):
        super(GetChildren, self).__init__(env)

    def step(self, action):

        obs, rew, done, info = self.env.step(action)

        for key, value in self._get_children_info().items():
            info[key] = value

        return obs, rew, done, info

    def _get_children_info(self):

        children = []
        rewards = []
        dones = []

        assert isinstance(
            self.env, WithSnapshots
        ), "Requires WithSnapshots reuse env states"

        for move in range(self.env.action_space.n):
            snapshot = self.env.get_snapshot()

            obs, rew, done, info = self.env.step(move)
            children.append(obs)
            rewards.append(rew)
            dones.append(done)

            self.env.load_snapshot(snapshot)

        return {
            "children": np.stack(children, axis=-1),
            "reward": np.array(rewards),
            "done": np.array(dones),
        }

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
