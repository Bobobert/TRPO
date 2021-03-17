import numba
from numba.typed import List
from tqdm import tqdm
from .const import *
from .utils import Tocker
from TRPO.optim import TRPO

def toT(arr:np.ndarray, device = DEVICE_DEFT, dtype = F_DTYPE_DEFT, grad: bool = False):
        arr = np.squeeze(arr) # Fix due to a problem in Pendulum Observation, seems arbritary when it fails or not.
        return torch.as_tensor(arr, dtype = dtype, device = device).unsqueeze(0).requires_grad_(grad)

def copyStateDict(net, grad:bool = True):
    newSD = dict()
    sd = net.state_dict()
    for i in sd.keys():
        t = sd[i]
        newSD[i] = t.new_empty(t.shape, requires_grad=grad).copy_(t)
    return newSD

def test(env, policy, testSteps:int = - 1):
    """
    Test function, agnostic to optimizer methods or such.

    returns
    -------
    accReward: int or float
        Accumulate reward from all the episode
    steps: int
        Steps developed in the episode
    """
    obs = env.reset()
    done, steps, accReward = False, 0, 0
    while not done:
        obs = toT(obs, device=policy.device)
        action = policy.getAction(obs)
        obs, reward, done, _ = env.step(action)
        accReward += reward
        steps += 1
        if testSteps > 0 and steps >= testSteps:
            done = True
    if isinstance(accReward, (np.ndarray)):
        accReward = accReward[0]
    return accReward, steps

def testRun(env, policy, nTests:int, testSteps:int = -1, prnt: bool = False):
    global LOGR
    meanRunReward, meanC, stepsMean, var = 0, 1 / nTests, 0, []
    for i in range(nTests):
        runRes, steps = test(env, policy, testSteps)
        meanRunReward += runRes * meanC
        stepsMean += steps * meanC
        var += [runRes]
    tVar = 0
    for v in var:
        tVar += meanC * (v - meanRunReward)**2
    if prnt:
        s = "Means: accumulate_reward {:.3f}, variance {:.3f}, steps {:.3f}".format(meanRunReward, tVar, stepsMean)
        print(s)
        if LOGR is not None:
            LOGR.logr(s)
    return meanRunReward, tVar, stepsMean
    
class Memory():
    def __init__(self, gamma: float = GAMMA, lmbd: float = LAMBDA, gae: bool = False):
        assert (gamma >= 0) and (gamma <= 1), 'Gamma must be in [0,1]'
        assert (lmbd >= 0) and (lmbd <= 1), 'lmbd(lambda) must be in [0,1]'
        self.gamma = gamma
        self.lmbd = lmbd
        self.gae = gae
        self.emptys()

    def emptys(self):
        self.states = []
        self.actions, self.probs = [], []
        self.baselines = []
        self.entropies = []
        self.rewards = List()
        self.advantages = List()
        self.notTerminals = List()
        self._i = 0

    def add(self, state, action, prob, reward, advantage, entropy, baseline, done:bool = False):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.advantages.append(advantage)
        self.baselines.append(baseline if baseline is not None else torch.zeros((1,)))
        self.notTerminals.append(0.0 if done else 1.0)
        self._i += 1

    def getBatch(self, size: int = -1, device = DEVICE_DEFT):
        """
        Once available the number of size of experience tuples on the memory
        a batch can be made from the samples.

        returns
        -------
        all torch.Tensors of the same dim0
        states, actions, returns
        """
        if size > len(self):
            return None
        
        returns = toT(self._getReturns_(self.rewards, self.notTerminals, self.gamma, 1.0), 
                            device=device, grad=False).squeeze()
        states = cat(self.states, dim=0).to(device)
        actions = cat(self.actions, dim=0).to(device)
        probs = cat(self.probs, dim=0).to(device)
        baselines = cat(self.baselines, dim=0).to(device)
        entropies = cat(self.entropies, dim = 0).to(device)

        if self.gae:
            advantages = toT(self._getReturns_(self.advantages, self.notTerminals, self.gamma, self.lmbd), 
                                device=device, grad=False).squeeze()
        else:
            advantages = returns - baselines.detach().squeeze()

        if size < 0:
            return {"states": states,
                    "actions":actions,
                    "probs":probs,
                    "baselines":baselines,
                    "returns":returns,
                    "advantages":advantages,
                    "entropies":entropies,
                    "N":len(self)}
            
        
        batchIdx = torch.randperm(len(self))[:size]

        return {"states":states[batchIdx],
                 "actions":actions[batchIdx],
                 "probs":probs[batchIdx],
                 "baselines":baselines[batchIdx],
                 "returns":returns[batchIdx],
                 "advantages":advantages[batchIdx],
                 "entropies":entropies[batchIdx],
                 "N":size}

    def clean(self):
        self.emptys()

    def __len__(self):
        return self._i

    @staticmethod
    @numba.njit
    def _getReturns_(rewards: List, notTerminals: List, gamma:float, lmbd:float):
        n = len(rewards)
        newArr = np.zeros(n, dtype = np.float32)
        gae = gamma * lmbd
        newArr[-1] = rewards[-1]
        for i in range(n - 2, -1, -1):
            newArr[i] = rewards[i] + gae * newArr[i + 1] * notTerminals[i]
        return newArr

class Crawler:
    """
    Crawler for gym environments.
    """
    def __init__(self, envMaker, policy,
                    baseline = None, gamma: float = GAMMA,
                    maxEpisodeLength: int = MAX_LENGTH,
                    batchSize: int = BATCH_SIZE,
                    pBatchMem: float = 1.0,
                    gae: bool = False, 
                    lmbd: float = LAMBDA,
                    vineSampling: bool = False,
                    device = DEVICE_DEFT):
        if gae and (baseline is None):
            raise ValueError("If gae mode is active a baseline must be passed on")
        self.env = envMaker()
        self.pi = policy
        self.baseline = baseline
        self.device = device
        self.maxEpLen = maxEpisodeLength
        self.batchSize = batchSize
        self.reqSteps = batchSize / pBatchMem if batchSize > 0 else 1
        self.mem = Memory(gamma, lmbd, gae)
        self.gae = gae
        self.gamma = gamma
        # TODO Add GAE, done (?)
        # TODO Add Vine method

    def singlePath(self, seed:int = - 1, stateDict = None):

        if stateDict is not None:
            self.pi.loadOther(stateDict)

        self.env.seed(seed if seed > 0 else None) # Reseed the environment
        
        steps = 0
        advantage = 0.0
        obs = toT(self.env.reset(), device = self.device)
        if self.baseline is not None:
            baseline = self.baseline.forward(obs)

        for _ in range(self.maxEpLen):
            steps += 1
            with torch.no_grad():
                action, log_action, entropy = self.pi.infere(obs)
                action_ = action.item() if self.pi.discrete else action.clone().to(DEVICE_DEFT).squeeze(0).numpy()
                nextObs, reward, done, _ = self.env.step(action_)
                nextObs = toT(nextObs, device = self.device)
                if self.baseline is not None:
                    nextBaseline = self.baseline.forward(nextObs) if not done else torch.zeros([])
            if self.gae:
                # Calculate delta_t
                advantage = reward + self.gamma * nextBaseline.item() - baseline.item()
            self.mem.add(obs, action, log_action, reward, advantage, entropy, baseline, done)
            obs = nextObs
            baseline = nextBaseline
            if done:
                break

        if stateDict is not None:
            self.pi.restoreParams()
        
        return steps

    def updatePi(self, stateDict):
        self.pi.loadOther(stateDict)

    def updateBasline(self, stateDict):
        self.baseline.loadOther(stateDict)

    def getBatch(self, seed: int = -1):
        steps = 0
        while steps < self.reqSteps:
            steps += self.singlePath(seed=seed)
        return self.mem.getBatch(self.batchSize)

    def clearMem(self):
        self.mem.clean()

    def getMem(self, device = DEVICE_DEFT):
        return self.mem.getBatch(device = device)

    def runTest(self, nTest:int):
        return testRun(self.env, self.pi, nTest)

def train(envMaker, policy,
            optPolicy,
            baseline = None,
            optBaselineMaker = None,
            testEnvMaker = None,
            saver = None,
            iterations: int = 100,
            batchSize: int = BATCH_SIZE,
            gamma: float = GAMMA,
            lmbd: float = LAMBDA,
            maxDKL : float = MAX_DKL,
            beta:float = BETA,
            maxEpisodeLength: int = MAX_LENGTH,
            pBatchMem: float = 1.0,
            nTests: int = TESTS,
            testFreq: int = TEST_FREQ,
            testSteps: int = MAX_LENGTH,
            device = DEVICE_DEFT,
            nWorkers = NCPUS,
            **kwargs
            ):

    assert (gamma <= 1) and (gamma >= 0), "Gamma must be in the interval [0,1] "
    assert nWorkers > 0, "nWorkers must be greater than 0"
    assert batchSize > 32, "Just 'case"
    global LOGR
    gae = kwargs.get("gae", False)
    print("Gae status", gae)
    if nWorkers > 1:
        try:
            import ray
            RAY = True
            nWorkers = nWorkers if nWorkers <= NCPUS else NCPUS
            ray.init(num_cpus=nWorkers)
        except:
            RAY = False
    else:
        RAY = False

    # Create and load the optimizers
    optPolicy = optPolicy(policy, **kwargs)
    optBaseline = optBaselineMaker(baseline.parameters()) if baseline is not None else None

    # Creating crawler
    if RAY:
        crawler = ray.remote(Crawler)
        batchPerCrw = ceil(batchSize / (nWorkers - 1))
        crawlers = [crawler.remote(envMaker, 
                                    policy.clone().to(DEVICE_DEFT), 
                                    baseline.clone().to(DEVICE_DEFT) if baseline is not None else None,
                                    gamma, maxEpisodeLength, 
                                    batchPerCrw, pBatchMem,
                                    gae = gae, lmbd = lmbd) for _ in range(nWorkers - 1)]
    else:
        crawler = Crawler(envMaker, policy.clone(), baseline.clone() if baseline is not None else None,
                            gamma,
                            maxEpisodeLength, batchSize, pBatchMem, 
                            gae = gae, lmbd = lmbd, device = device)

    testRewardRes, testVar, testStepsRes  = [], [], []
    # iterations loop
    bar = tqdm(range(iterations), unit="updates", desc="Training Policy")
    for it in bar:
        # Checking saver
        if saver is not None:
            saver.check()
        # Checking and executing test
        if it % testFreq == 0:
            envTest = testEnvMaker() if testEnvMaker is not None else envMaker()
            meanAcc, var, meanSteps = testRun(envTest, policy, nTests=nTests ,testSteps = testSteps)
            testRewardRes += [meanAcc]
            testVar += [var]
            testStepsRes += [meanSteps]
            bar.write("Test Results: meanGt {:.3f}, var {:.3f} meanEpSteps {:.3f}".format(meanAcc, var, meanSteps))

        # Produce and get trayectories batches
        if not RAY:
            trayectories = [crawler.getBatch()]
        else:
            trayectories = ray.get([crw.getBatch.remote() for crw in crawlers])
        
        # Update policy parameters
        s = optPolicy.updateParams(*trayectories)
        bar.write(s) if s is not None else None
        if LOGR is not None:
            LOGR.logr(s)

        # Update baseline parameters
        if baseline is not None:
            states, returns = optPolicy.states, optPolicy.returns
            states.detach_().to(device)
            returns = returns.detach_().to(device)
            # Doing mini batches - Information already scrambled
            n = returns.shape[0]
            for i in range(0, n, 32):
                s = i + 32
                s = s if s < n else n
                states_b, returns_b = states[i:s], returns[i:s]
                baseline_b = baseline.forward(states_b).squeeze()
                optBaseline.zero_grad()
                lossBaseline = F.mse_loss(baseline_b, returns_b)
                lossBaseline.backward()
                optBaseline.step()
        
        # Update crawlers
        if RAY:
            sdPi = policy.getState(cpu = True, lst=True)
            ray.get([cwr.updatePi.remote(sdPi) for cwr in crawlers])
            if baseline is not None:
                sdB = baseline.getState(cpu = True, lst=True)
                ray.get([cwr.updateBasline.remote(sdB) for cwr in crawlers])
            ray.get([cwr.clearMem.remote() for cwr in crawlers])
        else:
            crawler.updatePi(policy.getState(lst=True))
            if baseline is not None:
                crawler.updateBasline(baseline.getState(lst=True))
            crawler.clearMem()
        
    # Finishing training
    if RAY: ray.shutdown()
    return (testRewardRes,
            testVar,
            testStepsRes)

def playTest(env,
            policy,
            name:str,
            play_steps = 120,
            frameRate = 15):
    try:
        import imageio
        GIF = True
        bufferFrame = []
    except:
        GIF = False
        print("imageio is missing from the packages. A .gif from the run won't be made.")
    toc = Tocker()
    if play_steps > 0:
        # Start playing sequence
        # --- wait user to watch ----
        _ = input("Press any key to initialize test . . .")
        obs = env.reset()
        print("Test of the agent in {}".format(name))
        episodes, reward = 0, 0
        #I = tqdm(range(0, play_steps), 'Test in progress', unit=' plays')
        I = range(0, play_steps)
        for _ in I:
            if GIF:
                toc.tick
                bufferFrame.append(env.render())
            obs = toT(obs, device = policy.device)
            action = policy.getAction(obs)
            obs, stepRwd, done, _ = env.step(action)
            if done:
                episodes += 1
                obs = env.reset()
            reward += stepRwd
            if GIF:
                toc.lockHz(frameRate)
        env.close()
        print("Test play done. Completed {} episodes and accumulated {} points".format(episodes, reward))
        
        if GIF:
            try:
                imageio.mimsave("./testPlay {} frames {} episodes {} points {}.gif".format(name, play_steps, episodes, reward),
                                    bufferFrame, fps = frameRate)
            except:
                print("GIF creation error")
    else:
        None