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

        self.edges_priorities = {
            frozenset(["W", "G"]): 0,
            frozenset(["G", "R"]): 1,
            frozenset(["G", "Y"]): 2,
            frozenset(["G", "O"]): 3,
            frozenset(["B", "W"]): 4,
            frozenset(["B", "R"]): 5,
            frozenset(["B", "Y"]): 6,
            frozenset(["B", "O"]): 7,
            frozenset(["W", "O"]): 8,
            frozenset(["W", "R"]): 9,
            frozenset(["R", "Y"]): 10,
            frozenset(["Y", "O"]): 11,
        }

        self.corners_priorities = {
            frozenset(["W", "O", "G"]): 0,
            frozenset(["W", "R", "G"]): 1,
            frozenset(["Y", "G", "R"]): 2,
            frozenset(["O", "G", "Y"]): 3,
            frozenset(["B", "R", "W"]): 4,
            frozenset(["B", "O", "W"]): 5,
            frozenset(["O", "Y", "B"]): 6,
            frozenset(["B", "R", "Y"]): 7,
        }

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

        return self._faces.copy(), reward, done, {"debug": True}

    def reset(self, type="scramble", cube=None):

        assert type in {"scramble", "solved", "cube"}
        assert cube is not None if type == "cube" else False

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

        edges_one_hot = self._get_edge_one_hot()
        corners_one_hot = self._get_corner_one_hot()

        obs = np.concatenate([edges_one_hot, corners_one_hot], axis=0).astype(
            np.uint8
        )

        return obs.flatten().copy()

    def _get_edge_one_hot(self):
        # def get_edge_priority(edge):
        # edges = [
        # {"W", "G"},
        # {"G", "R"},
        # {"G", "Y"},
        # {"G", "O"},
        # {"B", "W"},
        # {"B", "R"},
        # {"B", "Y"},
        # {"B", "O"},
        # {"W", "O"},
        # {"W", "R"},
        # {"R", "Y"},
        # {"Y", "O"},
        # ]

        # return edges.index(edge)

        edge_colours, orientations = zip(
            *[self._get_colours_from_edge_id(i) for i in range(12)]
        )
        edge_positions = [
            self.edges_priorities[edge_colour] for edge_colour in edge_colours
        ]

        edge_positions = np.array(edge_positions, dtype=np.uint8)
        orientations = np.array(orientations)

        unique_edge_id = edge_positions * 2 + orientations

        one_hot = np.zeros(shape=(12, 24))
        one_hot[range(12), unique_edge_id] = 1

        return one_hot

    def _get_edge_face(self, face, edge, return_colour=True):

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

        if return_colour:
            return self.INVERSE_COLOUR_MAP[
                self._faces[self.COLOUR_MAP[face], x, y]
            ]
        else:
            return self._faces[self.COLOUR_MAP[face], x, y]

    def _get_colours_from_edge_id(self, edge_id):

        if edge_id == 0:
            colours = [
                self._get_edge_face(*args) for args in [("G", "T"), ("W", "B")]
            ]
        elif edge_id == 1:
            colours = [
                self._get_edge_face(*args) for args in [("G", "R"), ("R", "L")]
            ]
        elif edge_id == 2:
            colours = [
                self._get_edge_face(*args) for args in [("G", "B"), ("Y", "T")]
            ]
        elif edge_id == 3:
            colours = [
                self._get_edge_face(*args) for args in [("G", "L"), ("O", "R")]
            ]
        elif edge_id == 4:
            colours = [
                self._get_edge_face(*args) for args in [("W", "T"), ("B", "T")]
            ]
        elif edge_id == 5:
            colours = [
                self._get_edge_face(*args) for args in [("B", "L"), ("R", "R")]
            ]
        elif edge_id == 6:
            colours = [
                self._get_edge_face(*args) for args in [("B", "B"), ("Y", "B")]
            ]
        elif edge_id == 7:
            colours = [
                self._get_edge_face(*args) for args in [("B", "R"), ("O", "L")]
            ]
        elif edge_id == 8:
            colours = [
                self._get_edge_face(*args) for args in [("W", "L"), ("O", "T")]
            ]
        elif edge_id == 9:
            colours = [
                self._get_edge_face(*args) for args in [("W", "R"), ("R", "T")]
            ]
        elif edge_id == 10:
            colours = [
                self._get_edge_face(*args) for args in [("R", "B"), ("Y", "R")]
            ]
        else:
            colours = [
                self._get_edge_face(*args) for args in [("O", "B"), ("Y", "L")]
            ]

        return (
            frozenset(colours),
            np.argmin([self.COLOUR_MAP[colour] for colour in colours]),
        )

    def _get_corner_one_hot(self):
        # def get_corner_priority(corner):
        # corners = [
        # {"W", "O", "G"},
        # {"W", "R", "G"},
        # {"Y", "G", "R"},
        # {"O", "G", "Y"},
        # {"B", "R", "W"},
        # {"B", "O", "W"},
        # {"O", "Y", "B"},
        # {"B", "R", "Y"},
        # ]

        # return corners.index(corner)

        # return corners[frozenset(corner)]

        corner_colours, orientations = zip(
            *[self._get_colours_from_corner_id(i) for i in range(8)]
        )
        corner_positions = [
            self.corners_priorities[corner_colour]
            for corner_colour in corner_colours
        ]

        corner_positions = np.array(corner_positions, dtype=np.uint8)
        orientations = np.array(orientations)

        unique_corner_id = corner_positions * 3 + orientations

        one_hot = np.zeros(shape=(8, 24))
        one_hot[range(8), unique_corner_id] = 1

        return one_hot

    def _get_corner_face(self, face, corner, return_colour=True):

        assert corner in ("TL", "TR", "BL", "BR"), "unkown corner specified"

        assert face in self.COLOUR_MAP.keys(), "unknown face specified"

        x = 0 if corner[0] == "T" else 2
        y = 0 if corner[1] == "L" else 2

        if return_colour:
            return self.INVERSE_COLOUR_MAP[
                self._faces[self.COLOUR_MAP[face], x, y]
            ]
        else:
            return self._faces[self.COLOUR_MAP[face], x, y]

    def _get_colours_from_corner_id(self, corner_id):
        if corner_id == 0:
            colours = [
                self._get_corner_face(*args)
                for args in [("G", "TL"), ("W", "BL"), ("O", "TR")]
            ]
        elif corner_id == 1:
            colours = [
                self._get_corner_face(*args)
                for args in [("G", "TR"), ("R", "TL"), ("W", "BR")]
            ]
        elif corner_id == 2:
            colours = [
                self._get_corner_face(*args)
                for args in [("G", "BR"), ("R", "BL"), ("Y", "TR")]
            ]
        elif corner_id == 3:
            colours = [
                self._get_corner_face(*args)
                for args in [("G", "BL"), ("O", "BR"), ("Y", "TL")]
            ]
        elif corner_id == 4:
            colours = [
                self._get_corner_face(*args)
                for args in [("B", "TL"), ("R", "TR"), ("W", "TR")]
            ]
        elif corner_id == 5:
            colours = [
                self._get_corner_face(*args)
                for args in [("B", "TR"), ("W", "TL"), ("O", "TL")]
            ]
        elif corner_id == 6:
            colours = [
                self._get_corner_face(*args)
                for args in [("B", "BR"), ("O", "BL"), ("Y", "BL")]
            ]
        else:
            colours = [
                self._get_corner_face(*args)
                for args in [("B", "BL"), ("R", "BR"), ("Y", "BR")]
            ]

        return (
            frozenset(colours),
            np.argmin([self.COLOUR_MAP[colour] for colour in colours]),
        )

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
