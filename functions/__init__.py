"""
    Functions for the algorithm in general. Such as 
    train, test and utilities.
"""


from .functions import test, train, playTest, testRun
from .utils import Saver
from .const import getDevice
from .envMaker import configEnvMaker
from .gx import graphResults
from .logger import MiniLogger