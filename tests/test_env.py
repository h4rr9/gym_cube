import unittest

import numpy as np
from gymcube.envs.cube_gym import RubiksCubeEnv
from gymcube.wrappers import GetChildren, WithSnapshots


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

            obs, _, done, _ = self.env.step(a)

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

    def test_edge_value(self):

        valid_edge_values = set(
            [
                frozenset([0, 1]),
                frozenset([0, 2]),
                frozenset([0, 3]),
                frozenset([0, 4]),
                frozenset([5, 1]),
                frozenset([5, 2]),
                frozenset([5, 3]),
                frozenset([5, 4]),
                frozenset([3, 1]),
                frozenset([2, 1]),
                frozenset([2, 4]),
                frozenset([3, 4]),
            ]
        )

        self.env.reset()

        for _ in range(1_000):
            a = self.env.action_space.sample()

            _, _, done, _ = self.env.step(a)
            edge_values = self.env._faces[
                self.env.edge_position_indices
            ].reshape(12, 2)

            for value in edge_values:
                assert frozenset(value) in valid_edge_values

    def test_scramble_A(self):

        self.env.reset(type="solved")

        scramble_seq = ["L~", "B", "R~", "R", "U", "D", "R", "R", "L", "F"]

        final_state = np.array(
            [
                [[4, 1, 4], [4, 0, 5], [1, 1, 5]],
                [[0, 0, 1], [3, 1, 2], [0, 4, 2]],
                [[3, 1, 5], [0, 2, 2], [5, 4, 4]],
                [[0, 0, 2], [0, 3, 4], [0, 3, 2]],
                [[3, 3, 3], [1, 4, 3], [5, 2, 3]],
                [[4, 5, 1], [5, 5, 2], [1, 5, 2]],
            ]
        )

        for step in scramble_seq:
            self.env._turn(step)

        assert np.all(final_state == self.env._faces)

    def test_scramble_B(self):

        self.env.reset(type="solved")

        scramble_seq = [
            "L",
            "R",
            "B~",
            "F",
            "D",
            "U~",
            "D~",
            "D~",
            "L~",
            "U~",
            "U",
            "B~",
            "L~",
            "L",
            "U~",
            "F",
            "B~",
            "U~",
            "R~",
            "R~",
            "U",
            "F~",
            "L~",
            "F",
            "L~",
        ]
        final_state = np.array(
            [
                [[3, 1, 3], [2, 0, 4], [1, 5, 1]],
                [[0, 4, 0], [4, 1, 0], [4, 1, 5]],
                [[3, 3, 5], [3, 2, 1], [4, 2, 2]],
                [[4, 4, 2], [2, 3, 0], [2, 3, 0]],
                [[1, 3, 0], [5, 4, 0], [4, 2, 1]],
                [[2, 0, 3], [5, 5, 5], [5, 1, 5]],
            ]
        )

        for step in scramble_seq:
            self.env._turn(step)

        assert np.all(final_state == self.env._faces)

    def test_sanity_check_turns(self):

        self.env.reset()

        old = self.env._faces.copy()

        self.env._turn("F")
        self.env._turn("F~")

        assert np.all(self.env._faces == old)

        self.env._turn("B")
        self.env._turn("B~")

        assert np.all(self.env._faces == old)

        self.env._turn("U")
        self.env._turn("U~")

        assert np.all(self.env._faces == old)

        self.env._turn("D")
        self.env._turn("D~")

        assert np.all(self.env._faces == old)

        self.env._turn("R")
        self.env._turn("R~")

        assert np.all(self.env._faces == old)

        self.env._turn("L")
        self.env._turn("L~")

        assert np.all(self.env._faces == old)

    def test_children_wrapper(self):

        self.env_child.reset()

        assert isinstance(self.env_child, GetChildren)
        assert isinstance(self.env_child.env, WithSnapshots)

        _, _, done, info = self.env_child.step(0)

        assert set(set(["children", "done", "reward"])).issubset(
            set(info.keys())
        )

        assert info["children"].shape[0] == len(self.env_child.VALID_MOVES)
