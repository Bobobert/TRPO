"""
    File with all the defaults constants and functions
"""
import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn, no_grad, tanh, cat, exp, mul, mean, dot

LOGR = None

# Functions

ceil = math.ceil
Tsum = torch.sum
Tsqrt = torch.sqrt
Tlog = torch.log

# DEFAULTS 

NCPUS = os.cpu_count()
F_DTYPE_DEFT = torch.float32
DEVICE_DEFT = torch.device("cpu")
OPT_DEF = "adam"

def getDevice(cudaTry:bool = True):
    if torch.cuda.is_available() and cudaTry:
        return torch.device("cuda")
    return DEVICE_DEFT

# ALGS

ITER_PER_EPOCH = 10**5
GAMMA = 0.99
LAMBDA = 0.10
LEARNING_RATE = 0.00001
TEST_FREQ = 10**3
TEST_STEPS = -1
TESTS = 20
MAX_LENGTH = 1000
MAX_DKL = 1e-2
EPISODES_PER_ITER = 50
BETA = 0.01
CG_DAMPING = 1e-3
BATCH_SIZE = MAX_LENGTH * EPISODES_PER_ITER

# GX
ALPHA = 0.2
LINEWDT = 2
CLRRDM = "red"
CLRPI = "blue"