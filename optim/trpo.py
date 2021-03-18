# THIS IS HELL
from torch.distributions.kl import kl_divergence
from TRPO.functions.const import *
from .functions import cg, ls, convert2flat, convertFromFlat, unpackTrajectories

class TRPO:
    """
    Optimizer Trust Region Policy Optimization

    Mostly based on Schulman's implementation on Theano
    https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py 


    """
    def __init__(self, policy, **kwargs):
        self.pi = policy
        self.pi2 = policy.clone() # A copy of the network for surrogate evalution
        self.device = next(policy.parameters()).device
        self.delta = kwargs.get("delta", MAX_DKL)
        self.states, self.returns = None, None
        self.cgDamping = kwargs.get("cg_damping",CG_DAMPING)
        self.name = "TRPO"

    def __repr__(self):
        return self.name

    def updateParams(self, *trajectoryBatch):

        self.pi.none_grad()
        params = [p.clone().detach_() for p in self.pi.parameters()]

        states, actions, returns, advantage, oldLogprobs, _, _, N = unpackTrajectories(*trajectoryBatch, device = self.device)
        self.states, self.returns = states, returns
        
        Ni = 1.0 / N

        #advantage = returns - baselines
        #advantage.detach_()

        def calculateSurrogate(stateDict=None):
            if stateDict is not None:
                pi = self.pi2
                pi.loadOther(stateDict)
                states.detach_().requires_grad_(False)
            else:
                pi = self.pi
                states.requires_grad_(True)
            dist = pi.getDist(pi.forward(states))
            logprobsNew = Tsum(dist.log_prob(actions), dim=-1)
            oldLogprobs_ = Tsum(oldLogprobs.detach_(), dim=-1)
            probsDiff = exp(logprobsNew - oldLogprobs_) 
            surrogate = mean(mul(probsDiff, advantage))
            #surrogate *=  -1.0 if self.gae else 1.0
            return surrogate
        
        def getGrad(loss, c:float = 1.0, detach = True, 
                        graph:bool = False, retainGraph:bool = False):
            g = []
            loss.backward(create_graph=graph, retain_graph=retainGraph)
            for p in self.pi.parameters():
                ax = p.grad.clone().detach_() if detach else p.grad.clone()
                g += [c * ax]
            return g

        # Calculate gradient respect to L(Theta)
        surr = calculateSurrogate()
        pg = getGrad(surr)

        # Fisher-vector product
        def fvp(x, shapes):
            self.pi.none_grad()
            logprobs = self.pi.forward(states)
            logprobsFixed = logprobs.detach()
            dist = self.pi.getDist(logprobs)
            distFix = self.pi.getDist(logprobsFixed)

            klFirstFixed = kl_divergence(distFix, dist) * Ni
            klGrad = getGrad(klFirstFixed, detach = False, graph=True, retainGraph=True)

            xL = convertFromFlat(x.requires_grad_(False), shapes)
            gvpL = [Tsum(mul(v,w)).unsqueeze(0) for v, w in zip(klGrad, xL)]
            gvp = Tsum(cat(gvpL, dim=0))
            Fv, _ = convert2flat(getGrad(gvp))
            Fv += self.cgDamping * x

            #for mem clean
            states.detach_()
            klFirstFixed.detach_()

            return Fv

        # Begin
        ## Solve conjugate gradient for s
        stepDir, stpDirShapes  = cg(fvp, pg) # Consumes memory, gc failling in here
        flatFVP, _ = convert2flat(fvp(stepDir, stpDirShapes)) 
        sHs = 0.5 * dot(stepDir, flatFVP)
        betai = Tsqrt(sHs / self.delta)
        fullStep = stepDir / betai
        flatPG, _ = convert2flat(pg)
        GStepDir = dot(flatPG, stepDir) / betai

        ## line search for the paramaters
        self.pi.zero_grad()
        success, theta = ls(calculateSurrogate, params, fullStep, GStepDir)
        self.pi.loadOther(theta)
        
        # Mem clean
        def destroyArr(*arrs):
            for arr in arrs:
                if isinstance(arr, list):
                    for i in arr:
                        del i
                del arr
        destroyArr(pg, stepDir, flatFVP, fullStep, GStepDir, flatPG)
        return "Success {}, Surrogates: {:.3f}, {:.3f}".format(success, surr.item(), calculateSurrogate().item())
