from .optimMaker import optimizerMaker
from functions.const import *
from .functions import unpackTrayectories

class PG:
    """
    Optimizer Policy gradient
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
        states, actions, returns, oldLogprobs, baselines, N = unpackTrayectories(*trayectoryBatch, device = self.device)
        self.states, self.returns = states, returns

        advantages = returns - baselines
        out = self.policy.forward(states)
        dist = self.policy.getDist(out)
        logActions = dist.log_prob(actions.detach_())
        self.opt.zero_grad()
        lossPolicy = -1.0 * Tsum(mul(logActions, advantages))
        lossPolicy.backward()
        self.opt.step()

        return None