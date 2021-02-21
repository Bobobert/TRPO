from TRPO.functions.functions import copyStateDict
from TRPO.functions.const import *

class Policy(nn.Module):
    """
    Base class to define a parametric policy for 
    policy methods.
    """
    def __init__(self):
        super(Policy, self).__init__()
        self._name_ = "Parametric policy neural network"
        self.discrete = True
        self.__dvc__ = None

    def forwardSoftmax(self, X):
        out = self.forward(X)
        return F.softmax(out), out

    def forward(self, X):
        raise NotImplementedError

    def getDist(self, output):
        """
        This method should construct a distribution given an output
        of the network. This could be any shape or kind. 
        
        Should expect a batch of outputs too.

        parameters
        ----------
        output: torch.Tensor

        returns
        ----------
        distribution: torch.distributions.distribution.Distribution
        """
        raise NotImplementedError

    def infere(self, obs):
        """
        Given an observation, calculates accordingly the 
        corresponding action. If exploration is in order do
        not implement it here.

        parameters
        -----------
        obs: torch.Tensor

        returns
        -----------
        action: int, log_prob: torch.Tensor, mu: torch.Tensor
        """
        output = self.forward(obs)
        dist = self.getDist(output)
        actions = dist.sample()
        log_actions = dist.log_prob(actions)
        del dist # This should be alright as long as obs is stored
        return actions, log_actions, output

    def getAction(self, obs):
        with no_grad():
            dist = self.getDist(self.forward(obs))
            action = dist.sample()
        if self.discrete:
            return action.item()
        return action.to(DEVICE_DEFT).squeeze(0).numpy()

    def clone(self):
        """
        Returns a new object of the same type of the network 
        with the actual state loaded in

        returns
        --------
        Policy
        """
        return clone(self)
    
    def _new(self):
        """
        Must return the same type of object as the net with the same
        shapes and such.
        """
        raise NotImplementedError
    
    def loadOther(self, stateDict):
        """
        Load new parameters from a state_dict or
        a list of tensor parameters.
        
        This operation erases the grads.
        """
        loadOther(self, stateDict)

    def getState(self, cpu:bool=False, lst: bool = False):
        return getState(self, cpu) if not lst else getListState(self, cpu)

    def none_grad(self):
        for p in self.parameters():
            p.grad = None

    def zero_grad(self):
        for p in self.parameters():
            p.grad = p.new_zeros(p.shape)
    
    def restoreParams(self):
        """
        if available loads the previous state of the network,
        the previous state is stored when loadOther is called.

        This operation erases the grads.
        """
        try:
            self.load_state_dict(self.opParams)
            for p in self.parameters():
                p.grad = p.new_zeros(p.shape)
        except:
            None
    
    @property
    def device(self):
        if self.__dvc__ is None:
            self.__dvc__ =  next(self.parameters()).device
        return self.__dvc__

class Value(nn.Module):
    """
    Base class for a value function approximator.
    """
    def __init__(self):
        super(Value, self).__init__()
        self._name_ = "Parametric value function"
        self.__dvc__ = None

    def forward(self, X):
        raise NotImplementedError

    def clone(self):
        """
        Returns a new object of the same type of the network 
        with the actual state loaded in

        returns
        --------
        Policy
        """
        return clone(self)
    
    def _new(self):
        """
        Must return the same type of object as the net with the same
        shapes and such.
        """
        raise NotImplementedError

    @property
    def device(self):
        if self.__dvc__ is None:
            self.__dvc__ =  next(self.parameters()).device
        return self.__dvc__

    def getState(self, cpu:bool=False, lst:bool = False):
        return getState(self, cpu) if not lst else getListState(self, cpu)

    def loadOther(self, stateDict):
        loadOther(self, stateDict)

def clone(net):
    new = net._new()
    new.load_state_dict(copyStateDict(net), strict = True)
    return new.to(net.device)

def loadOther(net, targetLoad):
    net.opParams = copyStateDict(net)
    if isinstance(targetLoad, dict):
        net.load_state_dict(targetLoad)
    elif isinstance(targetLoad, list):
        for p, pt in zip(targetLoad, net.parameters()):
            pt.requires_grad_(False) # This is a must to change the values properly
            pt.copy_(p).detach_()
            pt.requires_grad_(True)

def getState(net, cpu):
    stateDict = net.state_dict()
    if cpu:
        for key in stateDict.keys():
            stateDict[key] = stateDict[key].to(DEVICE_DEFT)
    return stateDict

def getListState(net, cpu):
    params = []
    for p in net.parameters():
        params += [p if not cpu else p.clone().to(DEVICE_DEFT)]
    return params