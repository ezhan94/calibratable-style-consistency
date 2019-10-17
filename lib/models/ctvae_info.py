import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseSequentialModel
from lib.distributions import Normal, Multinomial


class CTVAE_info(BaseSequentialModel):

    name = 'ctvae_info' # conditional trajectory VAE policy w/ information factorization
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
        label_functions = self.config['label_functions']

        self.enc_birnn = nn.GRU(state_dim+action_dim, enc_rnn_dim, num_layers=num_layers, bidirectional=True)

        # TODO hacky, change this
        if 'mode' in self.config and self.config['mode'] == 'mujoco':
            assert not self.is_recurrent
            
            self.enc_mean = nn.Linear(2*enc_rnn_dim, z_dim)
            self.enc_logvar = nn.Linear(2*enc_rnn_dim, z_dim)

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
                nn.Linear(2*enc_rnn_dim, h_dim),
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

        self.aux_fc = nn.ModuleList([nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()) for lf in label_functions])
        self.aux_mean = nn.ModuleList([nn.Linear(h_dim, lf.output_dim) for lf in label_functions])
        self.aux_logvar = nn.ModuleList([nn.Linear(h_dim, lf.output_dim) for lf in label_functions])

        if self.is_recurrent:
            self.dec_rnn = nn.GRU(state_dim+action_dim, dec_rnn_dim, num_layers=num_layers)

    def _define_losses(self):
        self.log.add_loss('kl_div')
        self.log.add_loss('nll')
        self.log.add_metric('kl_div_true')

        for lf in self.config['label_functions']:
            self.log.add_loss('{}_label_pred'.format(lf.name))

    def ctvaep_params(self):
        params = list(self.enc_birnn.parameters()) + list(self.enc_mean.parameters()) + list(self.enc_logvar.parameters()) + \
            list(self.dec_action_fc.parameters()) + list(self.dec_action_mean.parameters())
    
        # TODO hacky
        if 'mode' not in self.config or self.config['mode'] != 'mujoco':
            params += list(self.enc_fc.parameters()) 
            params += list(self.dec_action_logvar.parameters())
        else:
            params += [self.dec_action_logvar]

        if self.is_recurrent:
            params += list(self.dec_rnn.parameters())

        return params

    def aux_params(self):
        return list(self.aux_fc.parameters()) + list(self.aux_mean.parameters()) + list(self.aux_logvar.parameters())

    def init_optimizer(self, lr):
        self.ctvaep_optimizer = torch.optim.Adam(self.ctvaep_params(), lr=lr)
        self.aux_optimizer = torch.optim.Adam(self.aux_params(), lr=lr)

    def optimize(self, losses):
        assert isinstance(losses, dict)

        self.ctvaep_optimizer.zero_grad()
        ctvaep_loss = sum(losses.values())
        ctvaep_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.ctvaep_params(), 10)
        self.ctvaep_optimizer.step()

        self.aux_optimizer.zero_grad()
        label_preds = [ value for key,value in losses.items() if 'label_pred' in key ]
        aux_loss = -sum(label_preds)
        aux_loss.backward()
        nn.utils.clip_grad_norm_(self.aux_params(), 10)
        self.aux_optimizer.step()

    def predict_labels(self, z, lf_idx, categorical):
        aux_h = self.aux_fc[lf_idx](z)
        aux_mean = self.aux_mean[lf_idx](aux_h)
        aux_logvar = self.aux_logvar[lf_idx](aux_h)

        if categorical:
            label_log_prob = F.log_softmax(aux_mean, dim=-1)
            return Multinomial(label_log_prob)
        else:
            return Normal(aux_mean, aux_logvar)

    def forward(self, states, actions, labels_dict):
        self.log.reset()
        
        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)
        labels = torch.cat(list(labels_dict.values()), dim=-1)

        # Encode
        posterior = self.encode(states[:-1], actions=actions)
        z = posterior.sample()

        kld = Normal.kl_divergence(posterior, free_bits=0.0).detach()
        self.log.metrics['kl_div_true'] = torch.sum(kld)

        kld = Normal.kl_divergence(posterior, free_bits=1/self.config['z_dim'])
        self.log.losses['kl_div'] = torch.sum(kld)

        # Train auxiliary networks to maximize label prediction losses
        # Train encoder to minimize label prediction losses
        for lf_idx, lf_name in enumerate(labels_dict):
            lf = self.config['label_functions'][lf_idx]
            lf_labels = labels_dict[lf_name]

            auxiliary = self.predict_labels(z, lf_idx, lf.categorical)
            self.log.losses['{}_label_pred'.format(lf_name)] = auxiliary.log_prob(lf_labels)

        # Decode
        self.reset_policy(labels=labels, z=z)

        for t in range(actions.size(0)):
            action_likelihood = self.decode_action(states[t])
            self.log.losses['nll'] -= action_likelihood.log_prob(actions[t])
            
            if self.is_recurrent:
                self.update_hidden(states[t], actions[t])

        return self.log
  