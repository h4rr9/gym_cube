import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="RubiksCube-v0",
    entry_point="gymcube.envs:RubiksCubeEnv",
    max_episode_steps=100,
    reward_threshold=1.0,
    kwargs={"scramble_moves": 10, "get_children": False, "half_turns": False},
)

from gymcube import envs
from gymcube import wrappers
