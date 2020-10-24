# RubiksCube Environment for gym

steps to install

    git clone https://github.com/h4rr9/gymcube.git
    cd gymcube
    python setup.py install

add gymcube dir to PYTHONPATH

    import os
    os.environ['PYTHONPATH'] += "<path to gymcube>"

## RubiksCube-v0
max_episode_length = 100
observations shape = (480,) # flattened vector of 20 * 24 one_hot values

arguments : half_turns (bool), scramble_moves (int), get_children (bool)


### creating envs

    import gym
    
    args = {'half_turns' : False, 'scramble_moves' : 10, 'get_children' : False}
    gym.make('gym_cube:RubiksCube-v0', *args)
    
    
Implementaion based on [Solving the Rubik's Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470#:~:text=A%20generally%20intelligent%20agent%20must,human%20data%20or%20domain%20knowledge.)