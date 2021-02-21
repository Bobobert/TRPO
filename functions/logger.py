from .utils import  timeFormatedS, timeFormated

class MiniLogger():
    """
    Lazzy logger
    """
    def __init__(self):
        self.FH = open("./log_{}.txt".format(timeFormated()), "wt")
        self.logr("Begining")

    def write(self, s, end:str = "\n"):
        self.FH.write(s + end)

    def kwLog(self, **kwargs):
        s, spc = beginEntry()
        for k in kwargs.keys():
            s += "arg: {} := {}".format(k, kwargs[k]) + " \n"
            s += spc
        self.write(s)

    def logr(self, msg):
        s, spc = beginEntry()
        s += msg
        self.write(s)
    

def beginEntry():
    s = timeFormatedS() + "_log :: "
    return s, spaceWhite(s)

def spaceWhite(s):
    spc = ""
    for a in s:
        spc += " "
    return spc