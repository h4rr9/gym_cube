import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="RubiksCubeChildData-v0",
    entry_point="gymcube.envs:RubiksCubeEnv",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 10, "get_children": True},
)

register(
    id="RubiksCube-v1",
    entry_point="gymcube.envs:RubiksCubeEnv",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 25, "get_children": True},
)
register(
    id="RubiksCube-v0",
    entry_point="gymcube.envs:RubiksCubeEnv",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 10, "get_children": False},
)

register(
    id="RubiksCube-v1",
    entry_point="gymcube.envs:RubiksCubeEnv",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 25, "get_children": False},
)
