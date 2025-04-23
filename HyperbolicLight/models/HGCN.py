# -*- codeing = utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
# from .agent import Agent
import random
import torch.nn.init as init
import math
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def build_memory():
    return []

class Agent(object):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id="0"):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.intersection_id = intersection_id

    def choose_action(self):

        raise NotImplementedError





class CoLightAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        super(CoLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.memory = build_memory()
        if cnt_round == 0:
            # initialization
            self.q_network = self.build_network()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
            self.criterion = nn.MSELoss()
            # if os.listdir(self.dic_path["PATH_TO_MODEL"]):
            #     self.q_network.load_state_dict(
            #         torch.load(
            #             os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.pt".format(intersection_id))))
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                "UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                        max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def _cal_len_feature(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                N += 8
            else:
                N += 12
        return N
    

    def adjacency_index2matrix(self, adjacency_index):
        # [batch,agents,neighbors]
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        lab = torch.tensor(adjacency_index_new).to(device)
        lab = torch.nn.functional.one_hot(lab, num_classes=self.num_agents).float()
        return lab

    def convert_state_to_input(self, s):
        """
        s: [state1, state2, ..., staten]
        """
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        feats0 = []
        adj = []
        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            tmp = []
            for feature in used_feature:
                if feature == "cur_phase":
                    if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                        tmp.extend(self.dic_traffic_env_conf['PHASE'][s[i][feature][0]])
                    else:
                        tmp.extend(s[i][feature])
                else:
                    tmp.extend(s[i][feature])

            feats0.append(tmp)
        feats = torch.tensor([feats0]).to(device)
        adj = self.adjacency_index2matrix(np.array([adj]))
        return [feats, adj]

    def choose_action(self, count, states):
        """
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        """
        xs = self.convert_state_to_input(states)
        
        q_values = self.q_network(xs[0],xs[1])
        # TODO: change random pattern
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = np.random.randint(self.num_actions, size=len(q_values[0]))
        else:
            action = np.argmax(q_values[0].detach().cpu().numpy(), axis=1)
        return action

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            tmp += ls[i]
        return [tmp]

    def prepare_Xs_Y(self, memory):
        """
        memory: [slice_data, slice_data, ..., slice_data]
        prepare memory for training
        """
        slice_size = len(memory[0])
        _adjs = []
        # state : [feat1, feat2]
        # feati : [agent1, agent2, ..., agentn]
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]

        for i in range(slice_size):
            _adj = []
            for j in range(self.num_agents):
                state, action, next_state, reward, _, _, _ = memory[j][i]
                _action[j].append(action)
                _reward[j].append(reward)
                _adj.append(state["adjacency_matrix"])
                # TODO
                _state[j].append(self._concat_list([state[used_feature[i]] for i in range(len(used_feature))]))
                _next_state[j].append(
                    self._concat_list([next_state[used_feature[i]] for i in range(len(used_feature))]))
            _adjs.append(_adj)
        # [batch, agent, nei, agent]
        _adjs2 = self.adjacency_index2matrix(np.array(_adjs))

        # [batch, 1, dim] -> [batch, agent, dim]
        _state2 = torch.tensor(np.concatenate([np.array(ss) for ss in _state], axis=1)).to(device)
        _next_state2 = torch.tensor(np.concatenate([np.array(ss) for ss in _next_state], axis=1)).to(device)
        target = self.q_network(_state2, _adjs2)
        next_state_qvalues = self.q_network_bar(_next_state2, _adjs2).detach() 
        # [batch, agent, num_actions]
        final_target = target.clone().detach()
        
        for i in range(slice_size):
            for j in range(self.num_agents):
                final_target[i, j, _action[j][i]] = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                    self.dic_agent_conf["GAMMA"] * torch.max(next_state_qvalues[i, j])

        self.Xs = [_state2, _adjs2]
        self.Y = final_target

    def build_network(self, MLP_layers=[32, 32]):
        
        model = Network(MLP_layers,self.dic_agent_conf,self.dic_traffic_env_conf,self.len_feature) #[batch,agent,action]
        model.to(device)

        

        return model

    def train_network(self):
        # Xs = [(batch, agent, dim),(batch, agent, nei, agent)],Y=[batch, agent, num_actions]
        with torch.no_grad():
            epochs = self.dic_agent_conf["EPOCHS"]
            batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))
        

            min_loss = np.array([2,3,1,4,5],dtype='float')
            min_loss = np.sort(min_loss)

        for epoch in range(epochs):
            perm = np.arange(self.Xs[0].size(0))  # batch
            np.random.shuffle(perm)
            loss_epoch = 0

            for i in range(0, len(perm), batch_size):
                indices = perm[i:i + batch_size]
                batch_Xs = [X[indices].to(device) for X in self.Xs]
                batch_Y = self.Y[indices].to(device)

                q_values = self.q_network(batch_Xs[0], batch_Xs[1])
                loss = self.criterion(q_values, batch_Y)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                loss_epoch += loss.item()

            average_loss = loss_epoch / len(perm)
            print("Epoch: {}/{}, Loss: {:.5f}".format(epoch + 1, epochs, average_loss))

            

            # 检查是否满足早停条件
            if average_loss>min_loss[4]:
                print("验证损失没有改善。停止训练。")
                break
            else:
                min_loss[4] = average_loss
                min_loss = np.sort(min_loss)
    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network = self.build_network()
        network.load_state_dict(network_copy.state_dict())
        # network.optimizer = optim.Adam(network.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
        # network.loss_function = nn.MSELoss()
        return network

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = torch.load(
            os.path.join(file_path, "%s.pt" % file_name),
            map_location=torch.device('cpu'))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
        self.criterion = nn.MSELoss()
        print("Succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.q_network_bar = torch.load(
            os.path.join(file_path, "%s.pt" % file_name),
            map_location=torch.device('cpu'))
        print("Succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        torch.save(self.q_network, os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.pt" % file_name))

    def save_network_bar(self, file_name):
        torch.save(self.q_network_bar, os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.pt" % file_name))


class RepeatVector3D(nn.Module):
    def __init__(self, times):
        super(RepeatVector3D, self).__init__()
        self.times = times

    def forward(self, inputs):
        # [batch, agent, dim] -> [batch, 1, agent, dim]
        # [batch, 1, agent, dim] -> [batch, times, agent, dim]
        return inputs.unsqueeze(1).repeat(1, self.times, 1, 1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def get_config(self):
        config = {'times': self.times}
        return config


############################################################################################################
class Network(nn.Module):
    def __init__(self,MLP_layers=[32, 32],dic_agent_conf=None, dic_traffic_env_conf=None,len_feature=20):
        super(Network,self).__init__()
        self.MLP_layers = MLP_layers
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.CNN_heads = [5] * len(self.CNN_layers)

        self.Poincare = PoincareBall()

        self.c = nn.Parameter(torch.Tensor([1.]))
        self.MultiHeadsAttModel = MultiHeadsAttModel(MLP_layers=self.MLP_layers,dic_agent_conf=self.dic_agent_conf,
                                                     dic_traffic_env_conf=self.dic_traffic_env_conf,manifold=self.Poincare,
                                                     c=self.c)

        
        
        self.HNN = HNN_MLP(self.c,self.Poincare,len_feature)
        self.fc = nn.Linear(32,self.num_actions)
    # In: [batch,agent,dim] [1, agent, dim]
    # In: [batch,agent,neighbors,agents] [1, agent, nei, agent]
    def forward(self,feats,adj):
      
        #"CNN_layers": [[32,32]]
        #feature = self.MLP(In[0], MLP_layers)
        feats = feats.to(torch.float32)
       
        feature = self.HNN(feats)
 
        #feature = self.Poincare.logmap0(feature,c=self.c)
   
        for CNN_layer_index, CNN_layer_size in enumerate(self.CNN_layers):
            if CNN_layer_index == 0:
                h, _ = self.MultiHeadsAttModel(
                    in_feats=feature,
                    in_nei=adj,
                    d_in=self.MLP_layers[-1],
                    h_dim=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    head=self.CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                )
            else:
          
                h, _ = self.MultiHeadsAttModel(
                    h,
                    adj,
                    d_in=self.MLP_layers[-1],
                    h_dim=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    head=self.CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                )
        
        out = self.Poincare.proj_tan0(self.Poincare.logmap0(h,c=self.c),self.c)
        out = self.fc(out)
        return out
    
class HNN_MLP(nn.Module):
    # Input:[batch,agent,32]
    def __init__(self, c, manifold,len_feature):
        super(HNN_MLP,self).__init__()
        self.c = c
        self.manifold = manifold
        self.H_layer1 = HNNLayer(self.manifold,len_feature,32,self.c,0,'relu',1)
        
    def forward(self,x):
        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        x = self.H_layer1(x)
      
        return x
class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
      
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )        

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = getattr(F, act)

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
    

class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError

def artanh(x):
    return Artanh.apply(x)

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)




class MultiHeadsAttModel(nn.Module):
    def __init__(self,MLP_layers=[32, 32],dic_agent_conf=None, dic_traffic_env_conf=None, manifold=None,c=None):
        super(MultiHeadsAttModel,self).__init__()
        self.MLP_layers = MLP_layers
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.CNN_heads = [5] * len(self.CNN_layers)
        self.manifold = manifold
        self.c = c
        self.H_layer1 = HNNLayer(self.manifold,32,32*5,self.c,0,'relu',1)
        self.H_layer2 = HNNLayer(self.manifold,32,32,self.c,0,'relu',1)

    def forward(self,in_feats=None, in_nei=None, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        """
        input: [batch, agent, dim] feature
               [batch, agent, nei, agent] adjacency
        input:[bacth,agent,128]
        output:
              [batch, agent, dim]
        """
        in_feats = self.manifold.proj_tan0(self.manifold.logmap0(in_feats,c=self.c),self.c)
        #in_feats = self.manifold.logmap0(in_feats,self.c)
        # [batch,agent,dim]->[batch,agent,1,dim]
        agent_repr = in_feats.unsqueeze(2)
        

        # [batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        # neighbor_repr = in_feats.unsqueeze(1).repeat(1, 1, self.num_agents, 1)
        neighbor_repr = RepeatVector3D(self.num_agents)(in_feats)
        
        

        # [batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr = torch.matmul(in_nei, neighbor_repr)

        # attention computation
        # [batch, agent, 1, dim]->[batch, agent, 1, h_dim*head]
        
        #agent_repr_head = self.fc1(agent_repr)
        agent_repr = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(agent_repr,self.c),self.c),self.c)
        agent_repr_head = self.H_layer1(agent_repr)
        agent_repr_head = self.manifold.proj_tan0(self.manifold.logmap0(agent_repr_head,c=self.c),self.c)
        #agent_repr_head = self.manifold.logmap0(agent_repr_head,self.c)

        agent_repr_head = agent_repr_head.view(agent_repr_head.size(0), self.num_agents, 1, h_dim, head)
        agent_repr_head = agent_repr_head.permute(0, 1, 4, 2, 3)

        # [batch,agent,neighbor,dim]->[batch,agent,neighbor,h_dim_head]
        #neighbor_repr_head = self.fc1(neighbor_repr)
        neighbor_repr = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(neighbor_repr,self.c),self.c),self.c)
        neighbor_repr_head = self.H_layer1(neighbor_repr)
        neighbor_repr_head = self.manifold.proj_tan0(self.manifold.logmap0(neighbor_repr_head,c=self.c),self.c)
        #neighbor_repr_head = self.manifold.logmap0(neighbor_repr_head,self.c)


        neighbor_repr_head = neighbor_repr_head.view(neighbor_repr_head.size(0), self.num_agents, self.num_neighbors,
                                                     h_dim, head)
        neighbor_repr_head = neighbor_repr_head.permute(0, 1, 4, 2, 3)

        # [batch,agent,head,1,h_dim]x[batch,agent,head,neighbor,h_dim]->[batch,agent,head,1,neighbor]
        att = torch.matmul(agent_repr_head, neighbor_repr_head.permute(0, 1, 2, 4, 3))
        att = torch.softmax(att, dim=-1)

        # [batch,agent,nv,1,neighbor]->[batch,agent,head,neighbor]
        att_record = att.view(att.size(0), self.num_agents, head, self.num_neighbors)

        # self embedding again
        neighbor_hidden_repr_head = self.H_layer1(neighbor_repr)
        neighbor_hidden_repr_head = self.manifold.proj_tan0(self.manifold.logmap0(neighbor_hidden_repr_head,c=self.c),self.c)
        #neighbor_hidden_repr_head = self.manifold.logmap0(neighbor_hidden_repr_head,self.c)


        neighbor_hidden_repr_head = neighbor_hidden_repr_head.view(neighbor_hidden_repr_head.size(0), self.num_agents,
                                                                   self.num_neighbors, h_dim, head)
        neighbor_hidden_repr_head = neighbor_hidden_repr_head.permute(0, 1, 4, 2, 3)

        out = torch.matmul(att, neighbor_hidden_repr_head).mean(dim=2)
        out = out.view(out.size(0), self.num_agents, h_dim)
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out,self.c),self.c),self.c)
        out = self.H_layer2(out)
        #out = self.fc2(out)
        
        return out, att_record

