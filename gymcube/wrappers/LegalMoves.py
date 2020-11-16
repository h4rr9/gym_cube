import gym
import numpy as np
from collections import deque


ANTI_MOVE_QUARTER = {i: (i + 6) % 12 for i in range(12)}
ANTI_MOVE_HALF = {i: i for i in range(12, 18)}
ANTI_MOVE_HALF = {**ANTI_MOVE_HALF, **ANTI_MOVE_QUARTER}


class LegalMoves(gym.Wrapper):
    """Wrapper to return a legal moves mask based on current and previous moves.
    
    Quarter moves:
        The move that returns cube to current state is illegal i.e (current_move=F, illegal=F~)

    Half moves:
        Same condition as Quarter as well as if previous and current moves are the same,
        then the half move which will return the cube to the current state is illegal,
        i.e (currenta and previous moves=F, illegal=F2).
    
    """

    def __init__(self, env):
        super(LegalMoves, self).__init__(env)
        self.last_performed_action = None

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        info["legal_moves"] = self._get_legal_moves(action)

        self.last_performed_action = action

        return obs, reward, done, info

    def _get_legal_moves(self, action):

        valid_moves = self.env.unwrapped.VALID_MOVES

        legal_moves = np.full(shape=len(valid_moves), fill_value=True)

        if len(valid_moves) == 12:
            legal_moves[ANTI_MOVE_QUARTER[action]] = False
        else:
            legal_moves[ANTI_MOVE_HALF[action]] = False

            if self.last_performed_action == action and action < 12:
                legal_moves[12 + (action % 6)] = False

        return legal_moves

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

