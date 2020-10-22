# RubiksCube Environment for gym

command to install

    pip install -e gym_cube


## RubiksCubeFlat-v0
max_episode_length = 50
scrable_moves = 10

observations shape = (54,)

## RubiksCubeFlat-v1
max_episode_length = 50
scrable_moves = 25

observations shape = (54,)

## RubiksCube-v0
max_episode_length = 50
scrable_moves = 10

observations shape = (6, 3, 3)
## RubiksCube-v1
max_episode_length = 50
scrable_moves = 25

observations shape = (6, 3, 3)


### code to create envs

    import gym
    
    gym.make('gym_cube:RubiksCubeFlat-v0')
    gym.make('gym_cube:RubiksCubeFlat-v1')
    gym.make('gym_cube:RubiksCube-v0')
    gym.make('gym_cube:RubiksCubeFlat-v1')
