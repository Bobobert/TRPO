from functions import testRun, train, Saver, configEnvMaker, playTest, graphResults, getDevice, MiniLogger
from optim import optimizerMaker, TRPO, PG
from nets import policyNet, randomPolicy, randomConPolicy, baselineNet, continuousPolicy

DEVICE = getDevice(False)
### USER VARS
#NAME = "MountainCar-v0" # GREAT BAD PERFOMANCE NEEDS EXPLORATION, PERHAPS MEMORY AMAIZING
#NAME = "Pendulum-v0" # Continium task. need to implement a continuos net.
#NAME = "CartPole-v0"
NAME = "LunarLanderContinuous-v2"
ITERS, TEST_FREQ = 200, 10
LEARNING_RATE = 0.00005
RAND_TESTS = 250
GAMMA = 0.99
MAX_EPISODE_LENGTH = 250
BATCH_SIZE = 5000
PBM = 0.6
BASELINE = True
GRAPH_MOD = "TRPO"
PI_OPT = TRPO
PLAY = False
### END

if __name__ == "__main__":
    name = NAME + "" + "_trpo_nogae"
    name += "_baseline" + "lr_" + str(LEARNING_RATE) if BASELINE else ""

    envmaker, nObs, nAction, discrete = configEnvMaker(NAME, seed=69)
    print("Action number {}, Observation shape {}".format(nAction, nObs))

    # Testing random agent
    print("Random results\n -------------")
    env = envmaker()
    if discrete:
        randpol = randomPolicy(env.action_space)
    else:
        randpol = randomConPolicy(env.action_space)
    meanRnd, varRnd, _ = testRun(env, randpol, RAND_TESTS, prnt=True)
    
    saver = Saver(name)
    LOGR = MiniLogger()
    LOGR.kwLog(NAME = NAME,
                ITERS = ITERS, TEST_FREQ = TEST_FREQ,
                LEARNING_RATE = LEARNING_RATE,
                RAND_TESTS = RAND_TESTS,
                GAMMA = GAMMA,
                MAX_EPISODE_LENGTH = MAX_EPISODE_LENGTH,
                BATCH_SIZE = BATCH_SIZE,
                PBM = PBM,
                BASELINE = BASELINE,
                GRAPH_MOD = GRAPH_MOD,
                PI_OPT = PI_OPT,
                PLAY = PLAY,
                discrete = discrete,
                nObs = nObs, 
                nAction = nAction)
    if discrete:
        policy = policyNet(nObs, nAction).to(DEVICE)
    else:
        policy = continuousPolicy(nObs, nAction).to(DEVICE)
    
    optpolicy = PI_OPT
    saver.addObj(policy, "policy", True, DEVICE) if saver is not None else None

    baseline = baselineNet(nObs).to(DEVICE) if BASELINE else  None
    saver.addObj(baseline, "baseline", True, DEVICE) if saver is not None else None
    optbaseline = optimizerMaker("adam", LEARNING_RATE)

    # Training policy
    print("Training policy\n -------------")
    LOGR.logr("Begining training")
    accRewards, var, steps = train(envmaker, 
                                policy,
                                optpolicy,
                                baseline = baseline,
                                optBaselineMaker=optbaseline,
                                device = DEVICE,
                                iterations = ITERS,
                                testFreq = TEST_FREQ,
                                maxEpisodeLength= MAX_EPISODE_LENGTH,
                                gamma = GAMMA,
                                batchSize = BATCH_SIZE,
                                pBatchMem = PBM,
                                nWorkers=1,
                                saver = saver,
                                learningRate = LEARNING_RATE)
    LOGR.logr("Trainnig Done")
    saver.saveAll() if saver is not None else None
    LOGR.logr("Objects saved")
    graphResults(accRewards, var, meanRnd, varRnd, ITERS, TEST_FREQ, GRAPH_MOD)
    if PLAY: playTest(env, policy, NAME, 200, 20)
    LOGR.logr("Experiment done")
