import numpy as np

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def store_batch(self, obs, act, rew, next_obs, done):
        n_agents = obs.shape[1]
        n_exps = obs.shape[0] * n_agents
        store_idxs = np.arange(self.ptr,self.ptr+n_exps)%self.max_size
        self.obs1_buf[store_idxs] = obs.reshape(n_exps, -1)
        self.obs2_buf[store_idxs] = next_obs.reshape(n_exps, -1)
        self.acts_buf[store_idxs] = act.reshape(n_exps, -1)
        self.rews_buf[store_idxs] = rew.reshape(n_exps)
        self.done_buf[store_idxs] = done.reshape(n_exps)
        self.ptr = (self.ptr+n_exps) % self.max_size
        self.size = min(self.size+n_exps, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def save(self, filename):
        object_to_save = [self.obs1_buf,
                          self.obs2_buf,
                          self.acts_buf,
                          self.rews_buf,
                          self.done_buf,
                          np.array([self.ptr, self.size, self.max_size])
                         ]
        np.savez(filename, *object_to_save)
    
    def load(self, filename):
        npzfile = np.load(filename)
        self.obs1_buf = npzfile['arr_0']
        self.obs2_buf = npzfile['arr_1']
        self.acts_buf = npzfile['arr_2']
        self.rews_buf = npzfile['arr_3']
        self.done_buf = npzfile['arr_4']
        self.ptr, self.size, self.max_size = npzfile['arr_5']