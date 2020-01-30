import numpy as np
import torch
from torch import optim

from learners.learner import Learner
from learners.core import to_tensor
import learners.core as core

"""
Soft Actor-Critic (SAC)
"""

class SAC(Learner):
    def __init__(self,
                 device,
                 action_space,
                 gamma=0.995,
                 polyak=0.995,
                 pi_lr=1e-3,
                 v_lr=1e-3,
                 alpha=0.2,
                 batch_size=128,
                 ac_kwargs = {},
                 ):
        super(SAC, self).__init__()
        self.device = device
        self.gamma = gamma
        self.polyak = polyak
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.batch_size = batch_size
        self.alpha = alpha

        ac_kwargs['action_space'] = action_space
        self.ac_main = core.ActorCritic(**ac_kwargs).to(device)
        self.ac_target = core.ActorCritic(**ac_kwargs).to(device)

        # Policy train op 
        self.pi_optimizer = optim.Adam(self.ac_main.policy.parameters(), lr=pi_lr)

        # Value train op
        value_params = [*self.ac_main.v.parameters(),
                        *self.ac_main.q1.parameters(),
                        *self.ac_main.q2.parameters()]
        self.v_optimizer = optim.Adam(value_params, lr=v_lr)

        # Initializing targets to match main variables
        self.ac_target.load_state_dict(self.ac_main.state_dict())

    def var_count(self,):
        var_count = {
            "pi":core.count_vars(self.ac_main.policy),
            "q1":core.count_vars(self.ac_main.q1),
            "q2":core.count_vars(self.ac_main.q2),
            "v":core.count_vars(self.ac_main.v),
            "total":core.count_vars(self.ac_main)
        }
        return var_count

    def train(self,):
        self.ac_main.train()

    def eval(self,):
        self.ac_main.eval()

    def learn(self, buffer):
            batch = buffer.sample_batch(self.batch_size)
            x, x2, a, r, d = [to_tensor(batch[k], self.device) for k in
                              ['obs1', 'obs2', 'acts', 'rews', 'done']]
            mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = self.ac_main(x, a)
            v_targ = self.ac_target.v(x2)

            # Min Double-Q:
            min_q_pi = torch.min(q1_pi, q2_pi)

            # Targets for Q and V regression
            q_backup = (r + self.gamma*(1-d)*v_targ).detach()
            v_backup = (min_q_pi - self.alpha * logp_pi).detach()

            # Soft actor-critic losses
            pi_loss = (self.alpha * logp_pi - q1_pi).mean()
            q1_loss = 0.5 * ((q_backup - q1)**2).mean()
            q2_loss = 0.5 * ((q_backup - q2)**2).mean()
            v_loss = 0.5 * ((v_backup - v)**2).mean()
            value_loss = q1_loss + q2_loss + v_loss

            # Q-learning update
            self.v_optimizer.zero_grad()
            value_loss.backward()
            self.v_optimizer.step()

            # Policy update
            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            self.pi_optimizer.step()

            # Polyak averaging for target variables
            # Credits: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
            for ac_target, ac_main in zip(self.ac_target.parameters(),
                                          self.ac_main.parameters()):
                ac_target.data.copy_(ac_main.data * (1.0 - self.polyak) +
                                     ac_target.data * self.polyak)

            return dict(LossPi=pi_loss.item(),
                        LossQ1=q1_loss.item(),
                        LossQ2=q2_loss.item(),
                        LossV=v_loss.item(),
                        Q1Vals=q1.cpu().detach().numpy(),
                        Q2Vals=q2.cpu().detach().numpy(),
                        VVals=v.cpu().detach().numpy(),
                        LogPi=logp_pi.cpu().detach().numpy()
                       )

    def get_action(self, obs, deterministic):
        self.ac_main.eval()
        if obs.ndim == 1: obs = obs.reshape(1, -1)
        mu, pi, _ = self.ac_main(to_tensor(obs, self.device))
        a = mu if deterministic else pi
        a = a.cpu().detach().numpy()
        return np.clip(a, -self.act_limit, self.act_limit)

    def checkpoint(self):
        checkpoints_dict = dict()
        # Log info about epoch
        # min_and_max
        for key in ['Q1Vals','Q2Vals','VVals']:
            checkpoints_dict[key] = 'mm'
        # average
        for key in ['LossPi','LossV','LossQ1','LossQ2']:
            checkpoints_dict[key] = 'avg'
        return checkpoints_dict