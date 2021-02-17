from .base import Policy, Value
from .dist import *
from functions.const import *

class policyNet(Policy):
    _h1_ = 30
    def __init__(self, inputs: int, actions: int):
        assert inputs > 0, "Inputs need to be greater than zero"
        assert actions > 0, "Actions need to be greater than zero"
        super(policyNet, self).__init__()
        self._ins, self._outs = inputs, actions
        self.rectifier = F.relu
        self.fc1 = nn.Linear(inputs, self._h1_)
        self.fc2 = nn.Linear(self._h1_, actions)

    def forward(self, X):
        X = self.rectifier(self.fc1(X))
        X = self.fc2(X)
        return X
    
    def getDist(self, output):
        return Categorical(logits = output)

    def _new(self):
        return policyNet(self._ins, self._outs)

class atariPolicy(Policy):
    def __init__(self, actions: int):
        None

class continuousPolicy(Policy):
    _h1_ = 30
    def __init__(self, inputs: int, dims: int):
        super(continuousPolicy, self).__init__()
        self.discrete = False
        self._ins, self._outs = inputs, dims
        self.rectifier = F.relu
        self.fc1 = nn.Linear(inputs, self._h1_)
        # Net has two heads, one for means, another for scales
        self.fc2_1 = nn.Linear(self._h1_, dims)
        self.fc2_2 = nn.Linear(self._h1_, dims)
        self.dims = dims

    def forward(self, X):
        X = self.rectifier(self.fc1(X))
        X_2 = X.clone()
        X = self.fc2_1(X)
        X_2 = self.fc2_2(X_2)
        return cat([X,X_2], dim=1)

    def getDist(self, output):
        # Zero deviation
        return Normal(output[:,:self.dims], exp(output[:,self.dims:]))

    def _new(self):
        return continuousPolicy(self._ins, self._outs)

class baselineNet(Value):
    _h1_ = 30
    def __init__(self, inputs:int):
        assert inputs > 0, "Inputs need to be greater than zero"
        super(baselineNet, self).__init__()
        self._ins = inputs
        self.rectifier = F.relu
        self.fc1 = nn.Linear(inputs, self._h1_)
        self.fc2 = nn.Linear(self._h1_, 1)

    def forward(self, X):
        X = self.rectifier(self.fc1(X))
        return self.fc2(X)
    
    def _new(self):
        return baselineNet(self._ins)