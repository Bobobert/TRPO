from TRPO.functions.const import *
from .functions import unpackTrajectories

# TODO All

class PPO:
    """
    Optimizer Policy Poximal Optimization

    From the paper https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, policy, **kwargs):
        self.pi = policy
        self.device = next(policy.parameters()).device
        self.delta = kwargs.get("delta", MAX_DKL)
        self.states, self.returns = None, None
        self.name = "PPO"
        self.gae = kwargs.get("gae", False)
        lr = kwargs.get("learningRate", LEARNING_RATE)
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.eps = kwargs.get("epsSurrogate", EPS_SURROGATE)
        self.beta = kwargs.get("entropyLoss", ENTROPY_LOSS)
        self.epochs = kwargs.get("ppoEpochs", PPO_EPOCHS)
        self.miniBatchSize = kwargs.get("miniBatchSize", MINI_BATCH_SIZE)

    def __repr__(self):
        return self.name

    def updateParams(self, *trajectoryBatch):
        states, actions, returns, advantage, oldLogprobs, _, _, N = unpackTrajectories(*trajectoryBatch, device = self.device)
        self.states, self.returns = states, returns
        # Enable gradient for states
        states.requires_grad_(True)
        # Normalize the advantages 
        advantage = div(advantage  - mean(advantage), EPS + std(advantage))

        # Calculate new surrogate function
        # This one using the cliped Advantage
        def calculateSurrogate(batchIdx):
            states_b = states[batchIdx]
            oldLogprobs_b = oldLogprobs[batchIdx]
            actions_b = actions[batchIdx]
            advantage_b = advantage[batchIdx]
            # Calculate surrogate
            dist = self.pi.getDist(self.pi.forward(states_b))
            entropies_b = dist.entropy()
            diffLogs = dist.log_prob(actions_b) - oldLogprobs_b.detach_()
            ratio = exp(Tsum(diffLogs, dim = -1)) 
            lossPolicy = mul(ratio, advantage_b)

            lossCliped = mul(ratio.clamp(1.0 - self.eps, 1.0 + self.eps), advantage_b)

            surrogate = -1.0 * mean(torch.min(lossPolicy, lossCliped))
            # Adding entropy bonus
            surrogate = surrogate - self.beta * mean(entropies_b)

            return surrogate
        
        # Train in batches
        for i in range(self.epochs):
            batchIdx = torch.randperm(N)[:self.miniBatchSize]

            loss = calculateSurrogate(batchIdx)
            self.opt.zero_grad()
            loss.backward()
            # Clamp gradients norm L2
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), np.inf)
            self.opt.step()

            with torch.no_grad():
                dist = self.pi.getDist(self.pi.forward(states))
                logProbs = dist.log_prob(actions)
            kl = mean(oldLogprobs - logProbs)
            if kl.item() > MAX_DKL:
                return "Ended at {} epoch. KL difference reached {}".format(i, kl.item())

        return None
