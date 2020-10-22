import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="RubiksCubeFlat-v0",
    entry_point="gym_cube.envs:RubiksCubeEnv",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 10, "flatten_state": True},
)

register(
    id="RubiksCubeFlat-v1",
    entry_point="gym_cube.envs:RubiksCubeEnv",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 25, "flatten_state": True},
)

register(
    id="RubiksCube-v0",
    entry_point="gym_cube.envs:RubiksCubeEnv",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 10, "flatten_state": False},
)

register(
    id="RubiksCube-v1",
    entry_point="gym_cube.envs:RubiksCubeEnv",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 25, "flatten_state": False},
)
