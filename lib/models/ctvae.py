import torch
import torch.nn as nn

from lib.models.core import BaseSequentialModel
from lib.distributions import Normal


class CTVAE(BaseSequentialModel):

    name = 'ctvae' # conditional trajectory VAE policy
    model_args = ['state_dim', 'action_dim', 'z_dim', 'h_dim', 'rnn_dim', 'num_layers']
    requires_labels = True

    def __init__(self, model_config):
        super().__init__(model_config)

    def _construct_model(self):
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        z_dim = self.config['z_dim']
        h_dim = self.config['h_dim']
        enc_rnn_dim = self.config['rnn_dim']
        dec_rnn_dim = self.config['rnn_dim'] if self.is_recurrent else 0
        num_layers = self.config['num_layers']
        label_dim = self.config['label_dim']

        self.enc_birnn = nn.GRU(state_dim+action_dim, enc_rnn_dim, num_layers=num_layers, bidirectional=True)

        # TODO hacky, change this
        if 'mode' in self.config and self.config['mode'] == 'mujoco':
            assert not self.is_recurrent
            
            self.enc_mean = nn.Linear(2*enc_rnn_dim+label_dim, z_dim)
            self.enc_logvar = nn.Linear(2*enc_rnn_dim+label_dim, z_dim)

            self.dec_action_fc = nn.Sequential(
                nn.Linear(state_dim+z_dim+label_dim+dec_rnn_dim, h_dim),
                nn.Tanh(),
                nn.Linear(h_dim, h_dim),
                nn.Tanh())
            self.dec_action_mean = nn.Sequential(
                nn.Linear(h_dim, action_dim),
                nn.Tanh())
            self.dec_action_logvar = nn.Parameter(torch.zeros(action_dim))
        else:
            self.enc_fc = nn.Sequential(
                nn.Linear(2*enc_rnn_dim+label_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU())
            self.enc_mean = nn.Linear(h_dim, z_dim)
            self.enc_logvar = nn.Linear(h_dim, z_dim)

            self.dec_action_fc = nn.Sequential(
                nn.Linear(state_dim+z_dim+label_dim+dec_rnn_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU())
            self.dec_action_mean = nn.Linear(h_dim, action_dim)
            self.dec_action_logvar = nn.Linear(h_dim, action_dim)

        if self.is_recurrent:
            self.dec_rnn = nn.GRU(state_dim+action_dim, dec_rnn_dim, num_layers=num_layers)

    def _define_losses(self):
        self.log.add_loss('kl_div')
        self.log.add_loss('nll')
        self.log.add_metric('kl_div_true')

    def forward(self, states, actions, labels_dict):
        self.log.reset()
        
        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)
        labels = torch.cat(list(labels_dict.values()), dim=-1)

        # Encode
        posterior = self.encode(states[:-1], actions=actions, labels=labels)

        kld = Normal.kl_divergence(posterior, free_bits=0.0).detach()
        self.log.metrics['kl_div_true'] = torch.sum(kld)

        kld = Normal.kl_divergence(posterior, free_bits=1/self.config['z_dim'])
        self.log.losses['kl_div'] = torch.sum(kld)

        # Decode
        self.reset_policy(labels=labels, z=posterior.sample())

        for t in range(actions.size(0)):
            action_likelihood = self.decode_action(states[t])
            self.log.losses['nll'] -= action_likelihood.log_prob(actions[t])
            
            if self.is_recurrent:
                self.update_hidden(states[t], actions[t])

        return self.log
