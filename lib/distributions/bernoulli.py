import torch
import torch.nn.functional as F

from .core import Distribution


class Bernoulli(Distribution):

    def __init__(self, mean):
        super().__init__()

        self.mean = mean

    def sample(self):
        # TODO sampling from Bernoulli currently returns mean
        return self.mean
        
    def log_prob(self, value):
        return F.binary_cross_entropy(self.mean, value, reduction='sum')
