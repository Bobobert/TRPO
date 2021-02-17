from gym import make
from gym.spaces import Box, Discrete, MultiDiscrete 
import gym

def configEnvMaker(name:str, seed:int = -1, **kwargs):
    
    tEnv = make(name)
    obS = tEnv.observation_space
    acS = tEnv.action_space
    actions = 0
    discrete = False

    def flat(t):
        e = 1
        for i in t:
            e *= i
        return e 

    if isinstance(acS, (Discrete)):
        discrete = True
        actions = acS.n
    elif isinstance(acS, (Box)):
        discrete = False
        actions = flat(acS.shape)

    def envMaker():
        env = make(name, **kwargs)
        env.seed(seed) if seed > 0 else None
        return env

    return envMaker, flat(obS.shape), actions, discrete