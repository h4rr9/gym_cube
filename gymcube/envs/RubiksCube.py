import numpy as np


class RubiksCube:
    """RubiksCube class used to simulate rubiks cube.

    The cube is held with the white side facing up and green side facing
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

    def __init__(self, scramble_moves=25):
        self.scramble_moves = scramble_moves

        self.COLOUR_MAP = {"W": 0, "G": 1, "R": 2, "O": 3, "B": 4, "Y": 5}
        self.INVERSE_COLOUR_MAP = {
            value: key for key, value in self.COLOUR_MAP.items()
        }

        self._faces = np.empty(shape=(6, 3, 3), dtype=np.uint8)
        self._solved_cube()

    def __getitem__(self, i):
        return self._faces[i]

    def _set_state(self, state):
        self._faces = state

    def _solved_cube(self):
        """cube Creates a normal RubiksCube
        """

        for i in range(len(self.COLOUR_MAP)):
            self._faces[i] = i

        self._faces = self._faces.astype(np.uint8)

    def _solved(self):

        return np.all(
            self._faces.max(axis=(1, 2)) == self._faces.min(axis=(1, 2))
        )

    def render(self):
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

        return value

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
                left[:, 2].copy(),
                np.flipud(top[0, :]).copy(),
                right[:, 0].copy(),
                np.flipud(bottom[2, :]).copy(),
            )

            front = np.rot90(m=front, k=1, axes=(1, 0))
        elif type == 1:

            top[0, :], right[:, 0], bottom[2, :], left[:, 2] = (
                np.flipud(right[:, 0]).copy(),
                bottom[2, :].copy(),
                np.flipud(left[:, 2]).copy(),
                top[0, :].copy(),
            )

            front = np.rot90(m=front, k=1, axes=(0, 1))
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

    def _turn(self, turn):
        """turn Performs turn o f cube
        
        Args:
            turn (str): turn in Rubiks Notation
        """
        turns = {
            "F": {"turn": self._turn_F, "kind": 0},
            "F~": {"turn": self._turn_F, "kind": 1},
            "F2": {"turn": self._turn_F, "kind": 2},
            "B": {"turn": self._turn_B, "kind": 0},
            "B~": {"turn": self._turn_B, "kind": 1},
            "B2": {"turn": self._turn_B, "kind": 2},
            "L": {"turn": self._turn_L, "kind": 0},
            "L~": {"turn": self._turn_L, "kind": 1},
            "L2": {"turn": self._turn_L, "kind": 2},
            "R": {"turn": self._turn_R, "kind": 0},
            "R~": {"turn": self._turn_R, "kind": 1},
            "R2": {"turn": self._turn_R, "kind": 2},
            "U": {"turn": self._turn_U, "kind": 0},
            "U~": {"turn": self._turn_U, "kind": 1},
            "U2": {"turn": self._turn_U, "kind": 2},
            "D": {"turn": self._turn_D, "kind": 0},
            "D~": {"turn": self._turn_D, "kind": 1},
            "D2": {"turn": self._turn_D, "kind": 2},
        }

        if turn in turns.keys():
            turns[turn]["turn"](turns[turn]["kind"])
        else:
            raise KeyError("Unknown turn command.")
