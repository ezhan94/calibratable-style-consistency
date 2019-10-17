import torch
import numpy as np

from util.datasets import LabelFunction

from matplotlib.patches import Ellipse


class AverageSpeed(LabelFunction):

    name = 'average_speed'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        
    def label_func(self, states, actions, true_label=None):
        vel = actions.view(actions.size(0), -1, 2)
        speed = torch.norm(vel, dim=-1)
        avg_speed = torch.mean(speed, dim=0)
        return torch.mean(avg_speed)

    def plot(self, ax, states, label, width, length):
        return ax


class Destination(LabelFunction):

    name = 'destination'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

        assert 'xy' in self.config
        self.dest = torch.tensor(self.config['xy']).float()

        self.time_index = self.config['time_index'] if 'time_index' in self.config else -1

    def label_func(self, states, actions, true_label=None):
        diff = states[self.time_index] - self.dest.to(states.device)
        diff = diff.view(-1, 2)
        displacement = torch.norm(diff, dim=-1)
        return torch.mean(displacement)

    def plot(self, ax, states, label, width, length):
        center = self.dest.numpy() * [length, width] + width/2
        # ax.plot(center[0], center[1], marker='*', color='black', markersize=12) # DEFAULT should be uncommented

        # Final positions
        # ax.plot(states[-1,::2], states[-1,1::2], 'D', color='black', markersize=6)
        # ax.plot(states[-1,::2], states[-1,1::2], 'o', color='black', markersize=8)

        if self.categorical:
            radii = self.thresholds.numpy()
        else:
            val = label[0]
            radii = [abs(val-0.1), val, val+0.1]

        for r in radii:
            w = 2*r*length
            h = 2*r*width
            ax.add_patch(Ellipse(xy=center, width=w, height=h, fill=False, color='g', linestyle='-', linewidth=2))

        return ax


class Displacement(LabelFunction):

    name = 'displacement'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        
    def label_func(self, states, actions, true_label=None):
        diff = states[-1] - states[0]
        diff = diff.view(-1, 2)
        displacement = torch.norm(diff, dim=-1)

        return torch.mean(displacement)

    def plot(self, ax, states, label, width, length):
        center = states[0]

        # Final positions
        ax.plot(states[-1,::2], states[-1,1::2], 'o', color='black', markersize=6)

        if self.categorical:
            radii = self.thresholds.numpy()
        else:
            val = label[0]
            radii = [abs(val-0.1), val, val+0.1]

        for r in radii:
            w = 2*r*length
            h = 2*r*width
            ax.add_patch(Ellipse(xy=center, width=w, height=h, fill=False, color='m', linestyle='-'))

        return ax


class Curvature(LabelFunction):

    name = 'curvature'
    eps = 1e-8

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        # Clamp actions to be non-zero
        actions[actions == 0] = self.eps

        v1 = actions[:-1]
        v2 = actions[1:]

        # TODO have to check formula for multi-agent setting
        numer = torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).squeeze()
        denom = torch.norm(v1, dim=1) * torch.norm(v2, dim=1)
        frac = numer/denom

        # Clamp into domain of torch.acos
        frac[frac > 1.0] = 1.0-self.eps
        frac[frac < -1.0] = -1.0+self.eps

        angles = torch.acos(frac)

        return torch.mean(angles)

    def plot(self, ax, states, label, width, length):
        return ax


class Direction(LabelFunction):

    name = 'direction'
    eps = 1e-8

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

        assert 'xy' in self.config
        self.zero_dir = torch.tensor(self.config['xy']).float()
        self.zero_dir[self.zero_dir == 0] = self.eps

    def label_func(self, states, actions, true_label=None):
        # TODO have to check formula for multi-agent setting
        direction = states[-1] - states[0]
        direction[direction == 0] = self.eps

        numer = torch.dot(direction, self.zero_dir)
        denom = torch.norm(direction) * torch.norm(self.zero_dir)
        frac = numer/denom

        # Clamp into domain of torch.acos
        frac[frac > 1.0] = 1.0-self.eps
        frac[frac < -1.0] = -1.0+self.eps

        angle = torch.acos(frac)

        return torch.mean(angle)

    def plot(self, ax, states, label, width, length):
        import pdb; pdb.set_trace()
        return ax
        
