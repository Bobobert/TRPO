from TRPO.functions.const import *

### FOR TRPO ONLY
def convert2flat(x):
    shapes = []
    flat = []
    for p in x:
        shapes += [p.shape]
        flat += [p.flatten()]
    return cat(flat, dim=0), shapes

def totSize(t):
    tot = 1
    for i in t:
        tot *= i
    return tot

def convertFromFlat(x, shapes):
    newX, iL, iS = [], 0, 0
    for s in shapes:
        iS = iL + totSize(s)
        newX += [x[iL:iS].reshape(s)]
        iL = iS
    return newX

def unpackTrayectories(*trayectories, device):
    N = 0
    # Ingest the trayectories
    if len(trayectories) > 1:
        states, actions, returns, advantages, logprobs, baselines, entropies  = [], [], [], [], [], [], []
        for trayectory in trayectories:
            states += [trayectory["states"]]
            actions += [trayectory["actions"]]
            returns += [trayectory["returns"]]
            advantages += [trayectory["advantages"]]
            logprobs += [trayectory["probs"]]
            baselines += [trayectory["baselines"]]
            entropies += [trayectory["entropies"]]
            N += trayectory["N"]
        states = cat(states, dim=0).to(device)
        actions = cat(actions, dim=0).to(device)
        returns = cat(returns, dim=0).to(device)
        advantages = cat(advantages, dim=0).to(device)
        logprobs = cat(logprobs, dim=0).to(device)
        baselines = cat(baselines, dim=0).to(device)
        entropies = cat(entropies, dim=0).to(device)
    else:
        trayectoryBatch = trayectories[0]
        states = trayectoryBatch["states"].to(device)
        actions = trayectoryBatch["actions"].to(device)
        returns = trayectoryBatch["returns"].to(device)
        advantages = trayectoryBatch["advantages"].to(device)
        logprobs = trayectoryBatch["probs"].to(device)
        baselines = trayectoryBatch["baselines"].to(device)
        entropies = trayectoryBatch["entropies"].do(device)
        N = trayectoryBatch["N"]
    return states, actions, returns, advantages, logprobs, baselines, entropies, N

def cg(mvp, b, iters: int = 10, epsilon: float = 1e-10):
    """
    Conjugated gradietns in pytorch

    based on https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf
    
    parameters
    ----------
    mvp: matrix-vector product function
        Any from the approximation of the fisher-vector
        product
    b: flat torch tensor
        Such as the gradients to solve s = A⁻1 g
    iters: int
        number of iterations to run the algorithm
    epsilon: float
        The threshold to finish early the algorithm when
        sqrt(rho_{k-1}) \leq \epsilon * |b|

    """
    # Init
    b, shapes = convert2flat(b)
    x = b.new_zeros(b.shape)
    r = b.clone().detach_()
    rho = dot(r,r) # Default to the Frobenius norm or L2 for vector type

    epsilon = epsilon * torch.norm(b, 2).item()

    # Iterations
    for k in range(iters):
        if Tsqrt(rho) <= epsilon:
            break
        oldRho = rho
        
        if k == 0:
            p = r
        else:
            p = r + (rhoRatio) * p

        w = mvp(r,shapes)
        alpha = oldRho / dot(p, w)
        x = x + alpha * p
        r = r - alpha * w
        rho = dot(r, r)
        rhoRatio = rho / oldRho

    del b, r, rho, w, alpha
    return x, shapes

def ls(f, x, direction, 
                expectedImproveRate,
                maxBacktracks: int = 10,
                acceptRatio:float = 0.1,
                Min:bool = False):
    """
    Backtracking Line search algorithm with Armijo condition,

    http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/Descent-Line-Search.pdf 
    
    Based on https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
    
    params
    ------
    f: function 
        to evaluate the direction on
    x: tensor
        Start point for evaluation
    direction: tensor
        Direction to search on
    expectedImproveRate: tensor
        The expected amount of improvement, from TRPO this
        amount is the slope dy/dx at the input

    """
    fx = f(x)
    direction, _ = convert2flat(direction)
    x, xShapes = convert2flat(x)
    for stepFrac in 0.5 ** np.arange(maxBacktracks):
        newX =  x + stepFrac * direction
        newfx = f(convertFromFlat(newX, xShapes))
        improvement = newfx - fx # In max mode, expected improvement when positive
        improvement *= -1.0 if Min else 1.0 
        expectedImprovement = expectedImproveRate * stepFrac
        r = improvement / expectedImprovement
        if r > acceptRatio and improvement > 0:
            return True, convertFromFlat(newX, xShapes)
    return False, convertFromFlat(x, xShapes)