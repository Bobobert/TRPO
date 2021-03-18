from .optimMaker import optimizerMaker
from TRPO.functions.const import *
from .functions import unpackTrajectories

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

    def updateParams(self, *trajectoryBatch):
        states, actions, returns, advantages, oldLogprobs, _, _, N = unpackTrajectories(*trajectoryBatch, device = self.device)
        self.states, self.returns = states, returns

        out = self.policy.forward(states)
        dist = self.policy.getDist(out)
        logActions = dist.log_prob(actions.detach_())
        self.opt.zero_grad()
        logActions = Tsum(logActions, dim = -1)
        lossPolicy = -1.0 * mean(mul(logActions, advantages))
        lossPolicy.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), np.inf)
        self.opt.step()

        return None