import torch
import torch.nn.functional as F

from .core import Distribution


class Dirac(Distribution):

    def __init__(self, mean):
        super().__init__()

        self.mean = mean

    def sample(self):
        return self.mean
        
    def log_prob(self, value):
        return -F.mse_loss(value, self.mean, reduction='sum')
