# RubiksCube Environment for gym

steps to install

    git clone https://github.com/h4rr9/gymcube.git
    pip install -e gymcube

## RubiksCube-v0
max_episode_length = 100
observations shape = (480,) # flattened vector of 20 * 24 one_hot values

arguments : half_turns (bool), scramble_moves (int)


### creating envs

    import gym
    import gymcube
    
    env = gym.make('RubiksCube-v0')
    
    """ specifying arguments"""
    args = {'half_turns' : False, 'scramble_moves' : 10}
    env = gym.make('RubiksCube-v0', *args) 
    
use wrapper to get children observations, children information will be passed in the info dictionary

    import gym
    from gymcube.wrappers import getchildren
    
    env = getchildren(gym.make('RubiksCube-v0'))
    
    env.reset()
    
    obs, rew, done, info = env.step(0)
    
    print(info)
    
    
Implementation based on [Solving the Rubik's Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470#:~:text=A%20generally%20intelligent%20agent%20must,human%20data%20or%20domain%20knowledge.){{:w
}}