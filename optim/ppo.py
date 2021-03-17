
from torch.distributions.kl import kl_divergence
from TRPO.functions.const import *
from .functions import cg, ls, convert2flat, convertFromFlat, unpackTrayectories

# TODO All

class PPO:
    """
    Optimizer Policy Poximal Optimization

    From the paper https://arxiv.org/pdf/1707.06347.pdf
    """
    def __init__(self, policy, **kwagrs):
        self.pi = policy
        self.device = next(policy.parameters()).device
        self.delta = kwagrs.get("delta", MAX_DKL)
        self.states, self.returns = None, None
        self.name = "PPO"
        self.gae = kwagrs.get("gae", False)

    def __repr__(self):
        return self.name

    def updateParams(self, *trayectoryBatch):
        states, actions, returns, advantages, oldLogprobs, baselines, entropies, N = unpackTrayectories(*trayectoryBatch, device = self.device)
        self.states, self.returns = states, returns

        # Calculate new surrogate function
        # This one using the cliped Advantage


