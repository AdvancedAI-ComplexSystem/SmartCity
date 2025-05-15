import random
import numpy as np
import torch 


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class ReplayBuffer:
    def __init__(self, memory, trajectory_len, device):
        s, a, ns, r = memory
        self.trajectory_len = trajectory_len
        self.batch = int(s.shape[1]/trajectory_len) * s.shape[0]
        self.batch_sample = 64 if self.batch > 64 else self.batch
        self.states = np.zeros(shape=(self.batch, trajectory_len, s.shape[2]))
        self.actions = np.zeros(shape=(self.batch, trajectory_len, a.shape[2]))
        self.next_states = np.zeros(shape=(self.batch, trajectory_len, ns.shape[2]))
        self.rewards = np.zeros(shape=(self.batch, trajectory_len, r.shape[2]))
        for i in range(s.shape[0]):
            for j in range(int(s.shape[1]/trajectory_len)):
                self.states[s.shape[0]*j+i] = s[i,j*trajectory_len:(j+1)*trajectory_len,:]
                self.actions[s.shape[0]*j+i] = a[i,j*trajectory_len:(j+1)*trajectory_len,:]
                self.next_states[s.shape[0]*j+i] = ns[i,j*trajectory_len:(j+1)*trajectory_len,:]
                self.rewards[s.shape[0]*j+i] = r[i,j*trajectory_len:(j+1)*trajectory_len,:]
        self.start = 0
        self.end = trajectory_len - 10
        self.device = device

    def sample_batch(self):
        s,a,ns,r = [],[],[],[]
        idxs = np.random.randint(
            0, self.batch, size=self.batch_sample
        )

        traj_idxs = np.random.randint(
            self.start, self.end, size=self.batch_sample
        )
        for i in range(self.batch_sample):
            ind = idxs[i]
            si0 = traj_idxs[i]
   
            s.append(self.states[ind][si0:si0 + 10,:])
            a.append(self.actions[ind][si0:si0 + 10,:])
            ns.append(self.next_states[ind][si0:si0 + 10,:])
            r.append(self.rewards[ind][si0:si0 + 10,:])
        return (
            torch.Tensor(np.array(s)).to(self.device),
            torch.Tensor(np.array(a)).to(self.device),
            torch.Tensor(np.array(ns)).to(self.device),
            torch.Tensor(np.array(r)).to(self.device),
            idxs,
            traj_idxs
        )

    def replace(self, idxs, traj_idxs, best_actions):
        np.copyto(self.states[idxs,traj_idxs,:], best_actions)
