# RubiksCube Environment for gym

steps to install

    git clone https://github.com/h4rr9/gymcube.git
    cd gymcube
    python setup.py install

add gymcube to PYTHONPATH
    import os
    os.environ['PYTHONPATH'] += "<path to gymcube>"


## RubiksCubeFlat-v0
max_episode_length = 50

scramble_moves = 10

observations shape = (54,)

## RubiksCubeFlat-v1
max_episode_length = 50

scramble_moves = 25

observations shape = (54,)

## RubiksCube-v0
max_episode_length = 50

scramble_moves = 10

observations shape = (6, 3, 3)
## RubiksCube-v1
max_episode_length = 50

scramble_moves = 25

observations shape = (6, 3, 3)


### creating envs

    import gym
    
    gym.make('gym_cube:RubiksCubeFlat-v0')
    gym.make('gym_cube:RubiksCubeFlat-v1')
    gym.make('gym_cube:RubiksCube-v0')
    gym.make('gym_cube:RubiksCubeFlat-v1')
