import unittest
import numpy as np
from gymcube.envs.cube_gym import RubiksCubeEnv
from gymcube.wrappers import WithSnapshots, GetChildren


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.env = RubiksCubeEnv()
        self.env_child = GetChildren(WithSnapshots(RubiksCubeEnv()))

    def tearDown(self):
        del self.env
        del self.env_child

    def test_sanity_check(self):

        assert np.all(
            self.env.reset().reshape(20, 24).sum(axis=-1) == np.ones(20)
        )

        for _ in range(1_000):
            a = self.env.action_space.sample()

            obs, _, done, _ = env.step(a)

            assert np.all(obs.reshape(20, 24).sum(axis=-1) == np.ones(20))

            if done:
                assert np.all(
                    self.env.reset().reshape(20, 24).sum(axis=-1) == np.ones(20)
                )

    def test_corner_value(self):

        valid_corner_values = set(
            [
                frozenset([0, 1, 2]),
                frozenset([0, 1, 3]),
                frozenset([1, 2, 5]),
                frozenset([1, 3, 5]),
                frozenset([0, 4, 2]),
                frozenset([0, 4, 3]),
                frozenset([4, 2, 5]),
                frozenset([4, 3, 5]),
            ]
        )

        self.env.reset()

        for _ in range(1_000):
            a = self.env.action_space.sample()

            _, _, done, _ = self.env.step(a)
            corner_values = self.env._faces[
                self.env.corner_position_indices
            ].reshape(8, 3)

            for value in corner_values:
                assert frozenset(value) in valid_corner_values

