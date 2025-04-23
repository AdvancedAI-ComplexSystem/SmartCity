# -*- codeing = utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
# from .agent import Agent
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input: [batch, agent, dim] feature   (3,12,128)
# [batch, agent, nei, agent] adjacency  (3,12,5,12)

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
        
        self.MultiHeadsAttModel = MultiHeadsAttModel(MLP_layers=self.MLP_layers,dic_agent_conf=self.dic_agent_conf,
                                                     dic_traffic_env_conf=self.dic_traffic_env_conf)
        self.MLP = MLP(MLP_layers,len_feature)
        self.fc = nn.Linear(32,self.num_actions)
    # In: [batch,agent,dim] [1, agent, dim]
    # In: [batch,agent,neighbors,agents] [1, agent, nei, agent]
    def forward(self,feats,adj):
        #"CNN_layers": [[32,32]]
        #feature = self.MLP(In[0], MLP_layers)
        feature = self.MLP(feats)
        
        for CNN_layer_index, CNN_layer_size in enumerate(self.CNN_layers):
            if CNN_layer_index == 0:
                #self,in_feats=None, in_nei=None, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
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
        # action prediction layer
        # [batch,agent,32]->[batch,agent,action]  
        out = self.fc(h)
        return out
    


# Input(shape=(self.num_agents, self.len_feature)      
class MLP(nn.Module):
    def __init__(self,layers=None,dim=128):
        super(MLP,self).__init__()
        #[32,32]
        self.lay1 = nn.Linear(dim,layers[0])
        # nn.init.normal_(self.lay1.weight)
        # nn.init.normal_(self.lay1.bias) 
        self.lay2 = nn.Linear(layers[0],layers[1])
        # nn.init.normal_(self.lay2.weight)
        # nn.init.normal_(self.lay2.bias) 
    def forward(self,ins):
        """
        Currently, the MLP layer
        -input: [batch,#agents,dim]
        -outpout: [batch,#agents,dim]
        """
        ins = ins.to(torch.float32)
        h = self.lay1(ins)
        h = nn.ReLU()(h)

        h = self.lay2(h)
        h = nn.ReLU()(h)

        return h

class MultiHeadsAttModel(nn.Module):
    def __init__(self,MLP_layers=[32, 32],dic_agent_conf=None, dic_traffic_env_conf=None, 
                 ):
        super(MultiHeadsAttModel,self).__init__()
        self.MLP_layers = MLP_layers
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.CNN_heads = [5] * len(self.CNN_layers)
        self.fc1 = nn.Linear(32,32*5)
        # nn.init.normal_(self.fc1.weight)
        # nn.init.normal_(self.fc1.bias) 
        self.fc2 = nn.Linear(32,32)
        # nn.init.normal_(self.fc2.weight)
        # nn.init.normal_(self.fc2.bias) 
    def forward(self,in_feats=None, in_nei=None, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        """
        input: [batch, agent, dim] feature
               [batch, agent, nei, agent] adjacency
        input:[bacth,agent,128]
        output:
              [batch, agent, dim]
        """
        # [batch,agent,dim]->[batch,agent,1,dim]
        agent_repr = in_feats.unsqueeze(2)
        

        # [batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        # neighbor_repr = in_feats.unsqueeze(1).repeat(1, 1, self.num_agents, 1)
        neighbor_repr = RepeatVector3D(self.num_agents)(in_feats)
        
        

        # [batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr = torch.matmul(in_nei, neighbor_repr)

        # attention computation
        # [batch, agent, 1, dim]->[batch, agent, 1, h_dim*head]
        
        agent_repr_head = self.fc1(agent_repr)
        agent_repr_head = agent_repr_head.view(agent_repr_head.size(0), self.num_agents, 1, h_dim, head) #[batch, agent, 1, h_dim,head]
        agent_repr_head = agent_repr_head.permute(0, 1, 4, 2, 3) #batch, agent, head,1,h_dim

        # [batch,agent,neighbor,dim]->[batch,agent,neighbor,h_dim_head]
        neighbor_repr_head = self.fc1(neighbor_repr)
        neighbor_repr_head = neighbor_repr_head.view(neighbor_repr_head.size(0), self.num_agents, self.num_neighbors,
                                                     h_dim, head)  #batch,agent,neighbor,h_dim,head
        neighbor_repr_head = neighbor_repr_head.permute(0, 1, 4, 2, 3)  #batch,agent,head,neighbor,h_dim

        # [batch,agent,head,1,h_dim]x[batch,agent,head,neighbor,h_dim]->[batch,agent,head,1,neighbor]
        att = torch.matmul(agent_repr_head, neighbor_repr_head.permute(0, 1, 2, 4, 3))
        #[batch, agent, head,1,h_dim ]* [batch,agent,head,h_dim,neighbor]
        #batch, agent, head,1,neighbor
        att = torch.softmax(att, dim=-1)

        # [batch,agent,nv,1,neighbor]->[batch,agent,head,neighbor]
        att_record = att.view(att.size(0), self.num_agents, head, self.num_neighbors)

        # self embedding again
        neighbor_hidden_repr_head = self.fc1(neighbor_repr)
        neighbor_hidden_repr_head = neighbor_hidden_repr_head.view(neighbor_hidden_repr_head.size(0), self.num_agents,
                                                                   self.num_neighbors, h_dim, head)
        neighbor_hidden_repr_head = neighbor_hidden_repr_head.permute(0, 1, 4, 2, 3)

        out = torch.matmul(att, neighbor_hidden_repr_head).mean(dim=2)
        out = out.view(out.size(0), self.num_agents, h_dim)
        out = self.fc2(out)
        
        return out, att_record

