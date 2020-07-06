import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseSequentialModel
from lib.distributions import Normal


class CTVAE_style(BaseSequentialModel):

    name = 'ctvae_style' # conditional trajectory VAE policy w/ dynamics model and w/ style-consistency
    model_args = ['state_dim', 'action_dim', 'z_dim', 'h_dim', 'rnn_dim', 'num_layers', 'dynamics_h_dim']
    model_args += ['dynamics_h_dim', 'n_collect']
    requires_labels = True
    requires_environment = True

    def __init__(self, model_config):
        super().__init__(model_config)

    def _construct_model(self):
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        z_dim = self.config['z_dim']
        h_dim = self.config['h_dim']
        dynamics_h_dim = self.config['dynamics_h_dim']
        enc_rnn_dim = self.config['rnn_dim']
        dec_rnn_dim = self.config['rnn_dim'] if self.is_recurrent else 0
        label_rnn_dim = self.config['rnn_dim']
        num_layers = self.config['num_layers']
        label_dim = self.config['label_dim']
        label_functions = self.config['label_functions']

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

        self.label_approx_birnn = nn.ModuleList([
            nn.GRU(state_dim+action_dim, label_rnn_dim, num_layers=num_layers, bidirectional=True) for lf in label_functions])

        self.label_approx_fc = nn.ModuleList([nn.Sequential(
            nn.Linear(2*label_rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, lf.output_dim)) for lf in label_functions])

        # Dynamics model
        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim+action_dim, dynamics_h_dim),
            nn.ReLU(),
            nn.Linear(dynamics_h_dim, dynamics_h_dim),
            nn.ReLU(),
            nn.Linear(dynamics_h_dim, state_dim))

        if self.is_recurrent:
            self.dec_rnn = nn.GRU(state_dim+action_dim, dec_rnn_dim, num_layers=num_layers)

    def _define_losses(self):
        self.log.add_loss('kl_div')
        self.log.add_loss('nll')
        self.log.add_loss('state_error')
        self.log.add_metric('kl_div_true')

        for lf in self.config['label_functions']:
            self.log.add_loss(lf.name)
            self.log.add_metric('{}_approx'.format(lf.name))
            self.log.add_metric('{}_true'.format(lf.name))

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

    def label_approx_params(self):
        return list(self.label_approx_birnn.parameters()) + list(self.label_approx_fc.parameters())

    def dynamics_params(self):
        return list(self.dynamics_model.parameters())

    def init_optimizer(self, lr):
        self.ctvaep_optimizer = torch.optim.Adam(self.ctvaep_params(), lr=lr)
        self.label_approx_optimizer = torch.optim.Adam(self.label_approx_params(), lr=lr)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics_params(), lr=1e-3, weight_decay=1e-5)

    def optimize(self, losses):
        assert isinstance(losses, dict)

        if self.stage == 1 or (self.stage >= 3 and self.config['n_collect'] > 0):
            self.dynamics_optimizer.zero_grad()
            dynamics_loss = losses['state_error']
            dynamics_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.dynamics_params(), 10)
            self.dynamics_optimizer.step()

        if self.stage == 2:
            self.label_approx_optimizer.zero_grad()
            label_preds = [ value for key,value in losses.items() if 'LF' in key ]
            label_approx_loss = sum(label_preds)
            label_approx_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.label_approx_params(), 10)
            self.label_approx_optimizer.step()

        if self.stage >= 3:
            self.ctvaep_optimizer.zero_grad()
            ctvaep_losses = [ value for key,value in losses.items() if 'state_error' not in key ]
            ctvaep_loss = sum(ctvaep_losses)
            ctvaep_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.ctvaep_params(), 10)
            self.ctvaep_optimizer.step()

    def label(self, states, actions, lf_idx, categorical):
        assert states.size(0) == actions.size(0)
        hiddens, _ = self.label_approx_birnn[lf_idx](torch.cat([states, actions], dim=-1))
        avg_hiddens = torch.mean(hiddens, dim=0)
        approx_out = self.label_approx_fc[lf_idx](avg_hiddens)

        if categorical:
            approx_out = F.log_softmax(approx_out, dim=-1)
            label_class = torch.argmax(approx_out, dim=1)
            approx_labels = torch.zeros(approx_out.size()).to(approx_out.device)
            approx_labels[:,label_class] = 1.0
            return approx_labels
        else:
            import pdb; pdb.set_trace()

    def compute_label_loss(self, states, actions, labels, lf_idx, categorical):
        assert states.size(0) == actions.size(0)
        hiddens, _ = self.label_approx_birnn[lf_idx](torch.cat([states, actions], dim=-1))
        avg_hiddens = torch.mean(hiddens, dim=0)
        approx_out = self.label_approx_fc[lf_idx](avg_hiddens)
        assert approx_out.size() == labels.size()

        if categorical:
            approx_out = F.log_softmax(approx_out, dim=-1)
            return -torch.sum(approx_out*labels)
        else:
            # TODO can make this a Dirac I think
            return F.mse_loss(approx_out, labels, reduction='sum')

    def propogate_forward(self, state, action):
        state_action_pair = torch.cat([state, action], dim=-1)
        return self.dynamics_model(state_action_pair)

    def compute_dynamics_loss(self, states ,actions):
        for t in range(actions.size(0)):
            state_change = self.propogate_forward(states[t], actions[t])
            self.log.losses['state_error'] += F.mse_loss(state_change, states[t+1]-states[t], reduction='sum')

    def forward(self, states, actions, labels_dict, env):
        self.log.reset()
        
        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)
        labels = torch.cat(list(labels_dict.values()), dim=-1)

        # Pretrain dynamics model
        if self.stage == 1:
            self.compute_dynamics_loss(states, actions)
        
        # Pretrain label approximators
        if self.stage == 2:
            for lf_idx, lf_name in enumerate(labels_dict):
                lf = self.config['label_functions'][lf_idx]
                lf_labels = labels_dict[lf_name]
                self.log.losses[lf_name] = self.compute_label_loss(states[:-1], actions, lf_labels, lf_idx, lf.categorical)

        # Train CTVAE
        elif self.stage >= 3:
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

            # Generate rollout w/ dynamics model
            self.reset_policy(labels=labels)
            rollout_states, rollout_actions = self.generate_rollout_with_dynamics(states, horizon=actions.size(0))

            # Compute label loss
            for lf_idx, lf_name in enumerate(labels_dict):
                lf = self.config['label_functions'][lf_idx]
                lf_labels = labels_dict[lf_name]
                self.log.losses[lf_name] = self.compute_label_loss(rollout_states[:-1], rollout_actions, lf_labels, lf_idx, lf.categorical)
                
                # Compute label loss with approx
                approx_labels = self.label(rollout_states[:-1], rollout_actions, lf_idx, lf.categorical)
                assert approx_labels.size() == lf_labels.size()
                self.log.metrics['{}_approx'.format(lf.name)] = torch.sum(approx_labels*lf_labels)

                # Compute label loss with true LF    
                rollout_lf_labels = lf.label(rollout_states.transpose(0,1).detach().cpu(),
                    rollout_actions.transpose(0,1).detach().cpu(), batch=True)
                assert rollout_lf_labels.size() == lf_labels.size()
                self.log.metrics['{}_true'.format(lf.name)] = torch.sum(rollout_lf_labels*lf_labels.cpu())

            # Update dynamics model with n_collect rollouts from environment
            if self.config['n_collect'] > 0:
                self.reset_policy(labels=labels[:1])
                rollout_states_env, rollout_actions_env = self.generate_rollout_with_env(env, horizon=actions.size(0))
                self.compute_dynamics_loss(rollout_states_env.to(labels.device), rollout_actions_env.to(labels.device))

        return self.log
  
    def generate_rollout_with_env(self, env, horizon):
        T = horizon
        states = torch.zeros(T+1, env.observation_space.shape[0])
        actions = torch.zeros(T, env.action_space.shape[0])

        obs = env.reset()
        states[0] = obs

        for t in range(T):
            with torch.no_grad():
                action = self.act(obs[:,:self.config['state_dim']])
                if 'mode' in self.config and self.config['mode'] == 'mujoco':
                    action = torch.clamp(action, min=-1.0, max=1.0) # TODO hack for mujoco, should be in env
                obs, reward, done, _ = env.step(action)

            states[t+1] = obs
            actions[t] = action

        return states[:,:self.config['state_dim']].unsqueeze(1), actions.unsqueeze(1)

    def generate_rollout_with_dynamics(self, states, horizon):
        rollout_states = [states[0].unsqueeze(0)]
        rollout_actions = []

        for t in range(horizon):
            curr_state = rollout_states[-1].squeeze(0)
            action = self.act(curr_state)
            next_state = curr_state + self.propogate_forward(curr_state, action)

            next_state = torch.clamp(next_state, min=-100, max=100) # TODO prevent compounding error

            rollout_states.append(next_state.unsqueeze(0))
            rollout_actions.append(action.unsqueeze(0))

        rollout_states = torch.cat(rollout_states, dim=0)
        rollout_actions = torch.cat(rollout_actions, dim=0)

        return rollout_states, rollout_actions
        