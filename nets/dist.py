from TRPO.functions.const import *
from torch.distributions.kl import register_kl
from torch.distributions import Normal, Categorical

@register_kl(Normal, Normal)
def kl_normals(dist1, dist2):
    mean1, mean2 = dist1.mean, dist2.mean
    logstd1, logstd2 = Tlog(dist1.stddev), Tlog(dist2.stddev)
    var1, var2 = dist1.variance, dist2.variance

    return Tsum(logstd2 - logstd1  + (var1 + torch.pow(mean1 - mean2, 2)) / (2 * var2) - 0.5)

@register_kl(Categorical, Categorical)
def kl_cats(dist1, dist2):
    ax1 = torch.div(dist1.logits, dist2.logits)
    return Tsum(mul(dist1.probs, ax1))