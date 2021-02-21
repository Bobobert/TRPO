from TRPO.functions.const import *
from .base import Policy

class randomPolicy(Policy):
    def __init__(self, actionSpace):
        super(randomPolicy, self).__init__()
        self.discrete = True
        self._n_ = actionSpace.n
        self.__dvc__ = DEVICE_DEFT

    def infere(self, obs):
        return self.leRan(), None, None

    def getAction(self, obs):
        return self.leRan().item()
    
    def leRan(self):
        return torch.randint(0, self._n_, size = (1,))

class randomConPolicy(Policy):
    def __init__(self, actionSpace):
        super(randomConPolicy, self).__init__()
        self.discrete = False
        self.actS = actionSpace
        self.__dvc__ = DEVICE_DEFT
        
    def infere(self, obs):
        return self.leRan(), None, None

    def getAction(self, obs):
        return self.leRan().numpy()
        
    def leRan(self):
        return torch.as_tensor(self.actS.sample())
