import gym
import numpy as np
from gym import spaces


class RubiksCubeEnv(gym.Env):
    """RubiksCube class used to simulate rubiks cube.

    The cube is help with the white side facing up and green side facing
    towards you.

    ### RubiksCube Notation ###
        F, R, U, L, B, D --- clockwise 90°
        F', R', U', L', B', D' --- counter clockwise 90°
        F2, R2, U2, L2, B2, D2 ---  180°

    ### Rubiks cube faces ###
      B
    O W R
      G

      W
    O G R
      Y

      W
    B O R
      Y

      W
    G R B
      Y

      W
    R B O
      Y

      G
    O Y R
      B

    The faces are shown above
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, scramble_moves=25, get_children=False, half_turns=False):
        super(RubiksCubeEnv, self).__init__()
        self.scramble_moves = scramble_moves
        self.get_children = get_children

        self.action_space = spaces.Discrete(18 if half_turns else 12)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(480,), dtype=np.uint8
        )

        if not half_turns:

            self.VALID_MOVES = [
                "F",
                "B",
                "U",
                "D",
                "L",
                "R",
                "F'",
                "B'",
                "U'",
                "D'",
                "L'",
                "R'",
            ]
        else:
            self.VALID_MOVES = [
                "F",
                "B",
                "U",
                "D",
                "L",
                "R",
                "F'",
                "B'",
                "U'",
                "D'",
                "L'",
                "R'",
                "F2",
                "B2",
                "U2",
                "D2",
                "L2",
                "R2",
            ]

        self.COLOUR_MAP = {"W": 0, "G": 1, "R": 2, "O": 3, "B": 4, "Y": 5}
        self.INVERSE_COLOUR_MAP = {
            value: key for key, value in self.COLOUR_MAP.items()
        }

        """each of the 12 edge position is w.r.t to the faces pieces of the cube with orientation mentioned above"""

        self.edge_position_indices = (
            self._get_edge_indices("G", "T"),
            self._get_edge_indices("W", "B"),
            # corresponds to edge position id 0
            self._get_edge_indices("G", "R"),
            self._get_edge_indices("R", "L"),
            # corresponds to edge position id 1
            self._get_edge_indices("G", "B"),
            self._get_edge_indices("Y", "T"),
            # corresponds to edge position id 2
            self._get_edge_indices("G", "L"),
            self._get_edge_indices("O", "R"),
            # corresponds to edge position id 3
            self._get_edge_indices("W", "T"),
            self._get_edge_indices("B", "T"),
            # corresponds to edge position id 4
            self._get_edge_indices("B", "L"),
            self._get_edge_indices("R", "R"),
            # corresponds to edge position id 5
            self._get_edge_indices("B", "B"),
            self._get_edge_indices("Y", "B"),
            # corresponds to edge position id 6
            self._get_edge_indices("B", "R"),
            self._get_edge_indices("O", "L"),
            # corresponds to edge position id 7
            self._get_edge_indices("W", "L"),
            self._get_edge_indices("O", "T"),
            # corresponds to edge position id 8
            self._get_edge_indices("W", "R"),
            self._get_edge_indices("R", "T"),
            # corresponds to edge position id 9
            self._get_edge_indices("R", "B"),
            self._get_edge_indices("Y", "R"),
            # corresponds to edge position id 10
            self._get_edge_indices("O", "B"),
            self._get_edge_indices("Y", "L"),
            # corresponds to edge position id 11
        )

        """each of the 8 corner position is w.r.t to the faces pieces of the cube with orientation mentioned above"""

        self.corner_position_indices = (
            self._get_corner_indices("G", "TL"),
            self._get_corner_indices("W", "BL"),
            self._get_corner_indices("O", "TR"),
            # corresponds to corner position 0
            self._get_corner_indices("G", "TR"),
            self._get_corner_indices("R", "TL"),
            self._get_corner_indices("W", "BR"),
            # corresponds to corner position 1
            self._get_corner_indices("G", "BR"),
            self._get_corner_indices("R", "BL"),
            self._get_corner_indices("Y", "TR"),
            # corresponds to corner position 2
            self._get_corner_indices("G", "BL"),
            self._get_corner_indices("O", "BR"),
            self._get_corner_indices("Y", "TL"),
            # corresponds to corner position 3
            self._get_corner_indices("B", "TL"),
            self._get_corner_indices("R", "TR"),
            self._get_corner_indices("W", "TR"),
            # corresponds to corner position 4
            self._get_corner_indices("B", "TR"),
            self._get_corner_indices("W", "TL"),
            self._get_corner_indices("O", "TL"),
            # corresponds to corner position 5
            self._get_corner_indices("B", "BR"),
            self._get_corner_indices("O", "BL"),
            self._get_corner_indices("Y", "BL"),
            # corresponds to corner position 6
            self._get_corner_indices("B", "BL"),
            self._get_corner_indices("R", "BR"),
            self._get_corner_indices("Y", "BR"),
            # corresponds to corner position 6
        )

        self._faces = np.empty(shape=(6, 3, 3), dtype=np.uint8)
        self._cube()

    def step(self, action):

        action = self.VALID_MOVES[action]

        self.turn(action)

        if self._solved():
            reward = 1.0
            done = True
        else:
            reward = -1.0
            done = False

        return self._get_observation(), reward, done, {"debug": True}

    def reset(self, type="scramble", cube=None):

        assert type in {"scramble", "solved", "cube"}
        assert (
            cube is not None if type == "cube" else True
        ), "cube argument cannot be None"

        if type == "scramble":
            self._cube()
            self._scramble()
        elif type == "solved":
            self._cube()
        else:
            self._faces = cube.astype(np.uint8)

        return self._get_observation()

    def render(self, mode="human"):
        faces = self._number_to_letters(self._faces)
        blue_face = faces[self.COLOUR_MAP["B"]]
        orange_face = faces[self.COLOUR_MAP["O"]]
        white_face = faces[self.COLOUR_MAP["W"]]
        red_face = faces[self.COLOUR_MAP["R"]]
        green_face = faces[self.COLOUR_MAP["G"]]
        yellow_face = faces[self.COLOUR_MAP["Y"]]
        space_line = "       "

        top = f"""
{space_line}{blue_face[2][2]} {blue_face[2][1]} {blue_face[2][0]}
{space_line}{blue_face[1][2]} {blue_face[1][1]} {blue_face[1][0]}
{space_line}{blue_face[0][2]} {blue_face[0][1]} {blue_face[0][0]}
    """

        middle = f"""
{orange_face[2][0]} {orange_face[1][0]} {orange_face[0][0]}  \
{white_face[0][0]} {white_face[0][1]} {white_face[0][2]}  \
{red_face[0][2]} {red_face[1][2]} {red_face[2][2]}
{orange_face[2][1]} {orange_face[1][1]} {orange_face[0][1]}  \
{white_face[1][0]} {white_face[1][1]} {white_face[1][2]}  \
{red_face[0][1]} {red_face[1][1]} {red_face[2][1]}
{orange_face[2][2]} {orange_face[1][2]} {orange_face[0][2]}  \
{white_face[2][0]} {white_face[2][1]} {white_face[2][2]}  \
{red_face[0][0]} {red_face[1][0]} {red_face[2][0]}
    """

        bottom = f"""
{space_line}{green_face[0][0]} {green_face[0][1]} {green_face[0][2]}
{space_line}{green_face[1][0]} {green_face[1][1]} {green_face[1][2]}
{space_line}{green_face[2][0]} {green_face[2][1]} {green_face[2][2]}
    """

        back = f"""
{space_line}{yellow_face[0][0]} {yellow_face[0][1]} {yellow_face[0][2]}
{space_line}{yellow_face[1][0]} {yellow_face[1][1]} {yellow_face[1][2]}
{space_line}{yellow_face[2][0]} {yellow_face[2][1]} {yellow_face[2][2]}
    """

        value = top + middle + bottom + back

        print(value)

    def _number_to_letters(self, faces):
        return np.array(
            [
                [
                    [self.INVERSE_COLOUR_MAP[value] for value in row]
                    for row in face
                ]
                for face in faces
            ],
            dtype="<U1",
        )

    def close(self):
        pass

    def _cube(self):
        """cube Creates a normal RubiksCube
        """

        for i in range(len(self.COLOUR_MAP)):
            self._faces[i] = i

        self._faces = self._faces.astype(np.uint8)

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

        return one_hot.flatten().copy()

    def _get_edge_indices(self, face, edge):

        assert face in self.COLOUR_MAP.keys()
        assert edge in {"T", "B", "L", "R"}

        if edge == "T":
            x, y = 0, 1
        elif edge == "B":
            x, y = 2, 1
        elif edge == "L":
            x, y = 1, 0
        else:
            x, y = 1, 2

        return self.COLOUR_MAP[face], x, y

    def _get_all_edge_priorities_and_orientations(self):

        colours = self._faces[tuple(zip(*self.edge_position_indices))].reshape(
            12, 2
        )

        return (self._get_edge_priorities(colours), np.argmin(colours, axis=1))

    def _get_edge_priorities(self, edge_colours):
        def equation(a, b):
            return 3 * a + 5 * b

        edge_colours = np.sort(edge_colours, axis=-1)
        val = equation(edge_colours[:, 0], edge_colours[:, 1])

        return np.argsort(val)

    def _get_corner_indices(self, face, corner):

        assert corner in ("TL", "TR", "BL", "BR"), "unkown corner specified"

        assert face in self.COLOUR_MAP.keys(), "unknown face specified"

        x = 0 if corner[0] == "T" else 2
        y = 0 if corner[1] == "L" else 2

        return self.COLOUR_MAP[face], x, y

    def _get_all_corner_priorities_and_orientations(self):

        colours = self._faces[
            tuple(zip(*self.corner_position_indices))
        ].reshape(8, 3)

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

    def _solved(self):

        return np.all(
            self._faces.max(axis=(1, 2)) == self._faces.min(axis=(1, 2))
        )

    def _scramble(self):

        random_sequence = np.random.choice(
            self.VALID_MOVES, self.scramble_moves, replace=True
        )

        for turn in random_sequence:
            self.turn(turn)

    def _turn_F(self, type=0):
        """turn_F turns the front of the self
        
        Args:
            type ([int]): type of turn
        """
        front = self._faces[self.COLOUR_MAP["G"]]
        top = self._faces[self.COLOUR_MAP["W"]]
        left = self._faces[self.COLOUR_MAP["O"]]
        right = self._faces[self.COLOUR_MAP["R"]]
        bottom = self._faces[self.COLOUR_MAP["Y"]]

        if type == 0:
            top[2, :], right[:, 0], bottom[0, :], left[:, 2] = (
                np.flipud(left[:, 2]).copy(),
                top[2, :].copy(),
                np.flipud(right[:, 0]).copy(),
                bottom[0, :].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(1, 0))
        elif type == 1:
            top[2, :], right[:, 0], bottom[0, :], left[:, 2] = (
                right[:, 0].copy(),
                np.flipud(bottom[0, :]).copy(),
                left[:, 2].copy(),
                np.flipud(top[2, :]).copy(),
            )

            front = np.rot90(m=front, k=1, axes=(0, 1))
        elif type == 2:
            top[2, :], right[:, 0], bottom[0, :], left[:, 2] = (
                np.flipud(bottom[0, :]).copy(),
                np.flipud(left[:, 2]).copy(),
                np.flipud(top[2, :]).copy(),
                np.flipud(right[:, 0]).copy(),
            )

            front = np.rot90(m=front, k=2, axes=(1, 0))

        self._faces[self.COLOUR_MAP["G"]] = front
        self._faces[self.COLOUR_MAP["W"]] = top
        self._faces[self.COLOUR_MAP["O"]] = left
        self._faces[self.COLOUR_MAP["R"]] = right
        self._faces[self.COLOUR_MAP["Y"]] = bottom

    def _turn_B(self, type):
        """turn_B turns the back of the self
        
        Args:
            type (int): denotes type of turn
        """
        front = self._faces[self.COLOUR_MAP["B"]]
        top = self._faces[self.COLOUR_MAP["W"]]
        left = self._faces[self.COLOUR_MAP["R"]]
        right = self._faces[self.COLOUR_MAP["O"]]
        bottom = self._faces[self.COLOUR_MAP["Y"]]

        if type == 0:
            top[0, :], right[:, 0], bottom[2, :], left[:, 2] = (
                np.flipud(right[:, 0]).copy(),
                bottom[2, :].copy(),
                np.flipud(left[:, 2]).copy(),
                top[0, :].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(0, 1))
        elif type == 1:
            top[0, :], right[:, 0], bottom[2, :], left[:, 2] = (
                left[:, 2].copy(),
                np.flipud(top[0, :]).copy(),
                right[:, 0].copy(),
                np.flipud(bottom[2, :]).copy(),
            )

            front = np.rot90(m=front, k=1, axes=(1, 0))
        elif type == 2:
            top[0, :], right[:, 0], bottom[2, :], left[:, 2] = (
                np.flipud(bottom[2, :]).copy(),
                np.flipud(left[:, 2]).copy(),
                np.flipud(top[0, :]).copy(),
                np.flipud(right[:, 0]).copy(),
            )

            front = np.rot90(m=front, k=2, axes=(1, 0))

        self._faces[self.COLOUR_MAP["B"]] = front
        self._faces[self.COLOUR_MAP["W"]] = top
        self._faces[self.COLOUR_MAP["R"]] = left
        self._faces[self.COLOUR_MAP["O"]] = right
        self._faces[self.COLOUR_MAP["Y"]] = bottom

    def _turn_U(self, type):
        """turn_U turns the top of the cube
        
        Args:
            type (int): denotes type of turn
        """
        front = self._faces[self.COLOUR_MAP["W"]]
        top = self._faces[self.COLOUR_MAP["B"]]
        right = self._faces[self.COLOUR_MAP["R"]]
        left = self._faces[self.COLOUR_MAP["O"]]
        bottom = self._faces[self.COLOUR_MAP["G"]]

        if type == 0:
            top[0, :], right[0, :], bottom[0, :], left[0, :] = (
                left[0, :].copy(),
                top[0, :].copy(),
                right[0, :].copy(),
                bottom[0, :].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(1, 0))
        elif type == 1:
            top[0, :], right[0, :], bottom[0, :], left[0, :] = (
                right[0, :].copy(),
                bottom[0, :].copy(),
                left[0, :].copy(),
                top[0, :].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(0, 1))
        elif type == 2:
            top[0, :], right[0, :], bottom[0, :], left[0, :] = (
                bottom[0, :].copy(),
                left[0, :].copy(),
                top[0, :].copy(),
                right[0, :].copy(),
            )

            front = np.rot90(m=front, k=2, axes=(1, 0))

        self._faces[self.COLOUR_MAP["W"]] = front
        self._faces[self.COLOUR_MAP["B"]] = top
        self._faces[self.COLOUR_MAP["R"]] = right
        self._faces[self.COLOUR_MAP["O"]] = left
        self._faces[self.COLOUR_MAP["G"]] = bottom

    def _turn_D(self, type):
        """turn_D turns the bottom of the cube
        
        Args:
            type (int): denotes type of turn
        """
        front = self._faces[self.COLOUR_MAP["Y"]]
        top = self._faces[self.COLOUR_MAP["G"]]
        left = self._faces[self.COLOUR_MAP["O"]]
        right = self._faces[self.COLOUR_MAP["R"]]
        bottom = self._faces[self.COLOUR_MAP["B"]]

        if type == 0:
            top[2, :], right[2, :], bottom[2, :], left[2, :] = (
                left[2, :].copy(),
                top[2, :].copy(),
                right[2, :].copy(),
                bottom[2, :].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(1, 0))
        elif type == 1:
            top[2, :], right[2, :], bottom[2, :], left[2, :] = (
                right[2, :].copy(),
                bottom[2, :].copy(),
                left[2, :].copy(),
                top[2, :].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(0, 1))
        elif type == 2:
            top[2, :], right[2, :], bottom[2, :], left[2, :] = (
                bottom[2, :].copy(),
                left[2, :].copy(),
                top[2, :].copy(),
                right[2, :].copy(),
            )

            front = np.rot90(m=front, k=2, axes=(1, 0))

        self._faces[self.COLOUR_MAP["Y"]] = front
        self._faces[self.COLOUR_MAP["G"]] = top
        self._faces[self.COLOUR_MAP["O"]] = left
        self._faces[self.COLOUR_MAP["R"]] = right
        self._faces[self.COLOUR_MAP["B"]] = bottom

    def _turn_L(self, type):
        """turn_L turns the left of the self
        
        Args:
            type (int): denotes type of turn
        """
        front = self._faces[self.COLOUR_MAP["O"]]
        top = self._faces[self.COLOUR_MAP["W"]]
        left = self._faces[self.COLOUR_MAP["B"]]
        right = self._faces[self.COLOUR_MAP["G"]]
        bottom = self._faces[self.COLOUR_MAP["Y"]]

        if type == 0:
            top[:, 0], right[:, 0], bottom[:, 0], left[:, 2] = (
                np.flipud(left[:, 2]).copy(),
                top[:, 0].copy(),
                right[:, 0].copy(),
                np.flipud(bottom[:, 0]).copy(),
            )

            front = np.rot90(m=front, k=1, axes=(1, 0))
        elif type == 1:
            top[:, 0], right[:, 0], bottom[:, 0], left[:, 2] = (
                right[:, 0].copy(),
                bottom[:, 0].copy(),
                np.flipud(left[:, 2]).copy(),
                np.flipud(top[:, 0]).copy(),
            )

            front = np.rot90(m=front, k=1, axes=(0, 1))
        elif type == 2:
            top[:, 0], right[:, 0], bottom[:, 0], left[:, 2] = (
                bottom[:, 0].copy(),
                np.flipud(left[:, 2]).copy(),
                top[:, 0].copy(),
                np.flipud(right[:, 0]).copy(),
            )

            front = np.rot90(m=front, k=2, axes=(1, 0))

        self._faces[self.COLOUR_MAP["O"]] = front
        self._faces[self.COLOUR_MAP["W"]] = top
        self._faces[self.COLOUR_MAP["B"]] = left
        self._faces[self.COLOUR_MAP["G"]] = right
        self._faces[self.COLOUR_MAP["Y"]] = bottom

    def _turn_R(self, type):
        """turn_R turns the right of the self
        
        Args:
            type (int): denotes type of turn
        """
        front = self._faces[self.COLOUR_MAP["R"]]
        top = self._faces[self.COLOUR_MAP["W"]]
        left = self._faces[self.COLOUR_MAP["G"]]
        right = self._faces[self.COLOUR_MAP["B"]]
        bottom = self._faces[self.COLOUR_MAP["Y"]]

        if type == 0:
            top[:, 2], right[:, 0], bottom[:, 2], left[:, 2] = (
                left[:, 2].copy(),
                np.flipud(top[:, 2]).copy(),
                np.flipud(right[:, 0]).copy(),
                bottom[:, 2].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(1, 0))
        elif type == 1:
            top[:, 2], right[:, 0], bottom[:, 2], left[:, 2] = (
                np.flipud(right[:, 0]).copy(),
                np.flipud(bottom[:, 2]).copy(),
                left[:, 2].copy(),
                top[:, 2].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(0, 1))
        elif type == 2:
            top[:, 2], right[:, 0], bottom[:, 2], left[:, 2] = (
                bottom[:, 2].copy(),
                np.flipud(left[:, 2]).copy(),
                top[:, 2].copy(),
                np.flipud(right[:, 0]).copy(),
            )

            front = np.rot90(m=front, k=2, axes=(1, 0))

        self._faces[self.COLOUR_MAP["R"]] = front
        self._faces[self.COLOUR_MAP["W"]] = top
        self._faces[self.COLOUR_MAP["G"]] = left
        self._faces[self.COLOUR_MAP["B"]] = right
        self._faces[self.COLOUR_MAP["Y"]] = bottom

    def turn(self, turn):
        """turn Performs turn o f cube
        
        Args:
            turn (str): turn in Rubiks Notation
        """
        turns = {
            "F": {"turn": self._turn_F, "kind": 0},
            "F'": {"turn": self._turn_F, "kind": 1},
            "F2": {"turn": self._turn_F, "kind": 2},
            "B": {"turn": self._turn_B, "kind": 0},
            "B'": {"turn": self._turn_B, "kind": 1},
            "B2": {"turn": self._turn_B, "kind": 2},
            "L": {"turn": self._turn_L, "kind": 0},
            "L'": {"turn": self._turn_L, "kind": 1},
            "L2": {"turn": self._turn_L, "kind": 2},
            "R": {"turn": self._turn_R, "kind": 0},
            "R'": {"turn": self._turn_R, "kind": 1},
            "R2": {"turn": self._turn_R, "kind": 2},
            "U": {"turn": self._turn_U, "kind": 0},
            "U'": {"turn": self._turn_U, "kind": 1},
            "U2": {"turn": self._turn_U, "kind": 2},
            "D": {"turn": self._turn_D, "kind": 0},
            "D'": {"turn": self._turn_D, "kind": 1},
            "D2": {"turn": self._turn_D, "kind": 2},
        }

        if turn in self.VALID_MOVES:
            turns[turn]["turn"](turns[turn]["kind"])
        else:
            raise KeyError("Unknown turn command.")
