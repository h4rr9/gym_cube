import gym
import numpy as np
from gym import spaces
from gymcube.envs.RubiksCube import RubiksCube


class RubiksCubeEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        scramble_moves=25,
        get_children=False,
        half_turns=False,
        flatten=False,
    ):
        super(RubiksCubeEnv, self).__init__()
        self.scramble_moves = scramble_moves
        self.get_children = get_children
        self.flatten = flatten

        if not half_turns:

            self.VALID_MOVES = [
                "F",
                "B",
                "U",
                "D",
                "L",
                "R",
                "F~",
                "B~",
                "U~",
                "D~",
                "L~",
                "R~",
            ]
        else:
            self.VALID_MOVES = [
                "F",
                "B",
                "U",
                "D",
                "L",
                "R",
                "F~",
                "B~",
                "U~",
                "D~",
                "L~",
                "R~",
                "F2",
                "B2",
                "U2",
                "D2",
                "L2",
                "R2",
            ]

        self.action_space = spaces.Discrete(len(self.VALID_MOVES))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(480,), dtype=np.uint8
        )

        """each of the 12 edge position is w.r.t to the faces pieces of the cube with orientation mentioned above"""

        self.edge_position_indices = (
            (
                1,
                0,
                1,
                2,
                1,
                5,
                1,
                3,
                0,
                4,
                4,
                2,
                4,
                5,
                4,
                3,
                0,
                3,
                0,
                2,
                2,
                5,
                3,
                5,
            ),
            (
                0,
                2,
                1,
                1,
                2,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                2,
                2,
                1,
                1,
                1,
                0,
                1,
                0,
                2,
                1,
                2,
                1,
            ),
            (
                1,
                1,
                2,
                0,
                1,
                1,
                0,
                2,
                1,
                1,
                0,
                2,
                1,
                1,
                2,
                0,
                0,
                1,
                2,
                1,
                1,
                2,
                1,
                0,
            ),
        )
        """each of the 8 corner position is w.r.t to the faces pieces of the cube with orientation mentioned above"""

        self.corner_position_indices = (
            (
                1,
                0,
                3,
                1,
                2,
                0,
                1,
                2,
                5,
                1,
                3,
                5,
                4,
                2,
                0,
                4,
                0,
                3,
                4,
                3,
                5,
                4,
                2,
                5,
            ),
            (
                0,
                2,
                0,
                0,
                0,
                2,
                2,
                2,
                0,
                2,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                2,
                2,
                2,
                2,
                2,
            ),
            (
                0,
                0,
                2,
                2,
                0,
                2,
                2,
                0,
                2,
                0,
                2,
                0,
                0,
                2,
                2,
                2,
                0,
                0,
                2,
                0,
                0,
                0,
                2,
                2,
            ),
        )

        self.cube = RubiksCube(scramble_moves=scramble_moves)

    def step(self, action):

        assert action < len(self.VALID_MOVES) and action >= 0, "Unknown action"

        action = self.VALID_MOVES[action]

        self.cube._turn(action)

        if self.cube._solved():
            reward = 1.0
            done = True
        else:
            reward = -1.0
            done = False

        return self._get_observation(), reward, done, {}

    def reset(self, type="scramble", cube=None):

        assert type in {"scramble", "solved", "cube"}
        assert (
            cube is not None if type == "cube" else True
        ), "cube argument cannot be None"

        if type == "scramble":
            self.cube._solved_cube()
            self._scramble()
        elif type == "solved":
            self.cube._solved_cube()
        else:
            self.cube._set_state(cube.astype(np.uint8))

        return self._get_observation()

    def render(self, mode="human"):
        if mode == "ascii":
            return self.cube.render()
        elif mode == "array":
            return self._get_faces()
        else:
            raise NotImplementedError
        pass  # TODO

    def close(self):
        pass

    def _scramble(self):

        random_sequence = np.random.choice(
            self.VALID_MOVES, self.scramble_moves, replace=True
        )

        for turn in random_sequence:
            self.cube._turn(turn)

    def _get_observation(self):

        edge_positions, edge_orientations = (
            self._get_all_edge_priorities_and_orientations()
        )

        unique_edge_id = edge_positions * 2 + edge_orientations

        corner_positions, corner_orientations = (
            self._get_all_corner_priorities_and_orientations()
        )

        unique_corner_id = corner_positions * 3 + corner_orientations

        one_hot = np.zeros(shape=(20, 24), dtype=np.uint8)
        one_hot[np.arange(12), unique_edge_id] = 1
        one_hot[np.arange(12, 20), unique_corner_id] = 1

        if self.flatten:
            one_hot = one_hot.flatten()

        return one_hot

    def _get_all_edge_priorities_and_orientations(self):

        colours = self.cube[self.edge_position_indices].reshape(12, 2)

        return (self._get_edge_priorities(colours), np.argmin(colours, axis=1))

    def _get_edge_priorities(self, edge_colours):
        def equation(a, b):
            return 3 * a + 5 * b

        edge_colours = np.sort(edge_colours, axis=-1)
        val = equation(edge_colours[:, 0], edge_colours[:, 1])

        return np.argsort(val)

    def _get_all_corner_priorities_and_orientations(self):

        colours = self.cube[self.corner_position_indices].reshape(8, 3)

        return (
            self._get_corner_priorities(colours),
            np.argmin(colours, axis=1),
        )

    def _get_corner_priorities(self, corner_colours):
        def equation(a, b, c):
            return 3 * a + 5 * b + 7 * c

        corner_colours = np.sort(corner_colours, axis=-1)
        val = equation(
            corner_colours[:, 0], corner_colours[:, 1], corner_colours[:, 2]
        )

        return np.argsort(val)

    def _get_faces(self):
        return self.cube._faces.copy()

    def _turn(self, *args, **kwargs):
        self.cube._turn(*args, **kwargs)
