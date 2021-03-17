from .optimMaker import optimizerMaker
from TRPO.functions.const import *
from .functions import unpackTrayectories

class PG:
    """
    Optimizer Policy gradient

    It does support a baseline to calculate the Advantage function A(s,a) = Q(s,a) - V(s)
    """
    def __init__(self, policy, **kwargs):
        self.policy = policy
        self.device = next(policy.parameters()).device
        optm = optimizerMaker(optimizer=kwargs.get("optimizer", OPT_DEF),
                            learningRate=kwargs.get("learningRate", LEARNING_RATE))
        self.opt = optm(self.policy.parameters())
        self.name = "PG-Vanilla"
        
    def __repr__(self):
        return self.name

    def updateParams(self, *trayectoryBatch):
        states, actions, returns, advantages, oldLogprobs, _, _, N = unpackTrayectories(*trayectoryBatch, device = self.device)
        self.states, self.returns = states, returns

        out = self.policy.forward(states)
        dist = self.policy.getDist(out)
        logActions = dist.log_prob(actions.detach_())
        self.opt.zero_grad()
        logActions = Tsum(logActions, dim = -1)
        lossPolicy = -1.0 * mean(mul(logActions, advantages))
        lossPolicy.backward()
        self.opt.step()

        return None