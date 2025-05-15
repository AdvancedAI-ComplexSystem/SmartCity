"""
MPLight agent, based on FRAP model structure.
Observations: [cur_phase, traffic_movement_pressure_num]
Reward: -Pressure
"""

from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda,  Activation, Embedding, Conv2D, concatenate, add,\
    multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent, slice_tensor, relation
from tensorflow.keras import backend as K
import numpy as np
import random
from utils.batch_buffer import ReplayBuffer
from utils.make_mask_noise import make_guassion_noise, make_U_rand_noise
import torch
class MPLightAgent(NetworkAgent):


    """
    optimize the build_network function
    assert the features are [ cur_phase, other feature]
    in this part, the determined feature name are removed
    """
    def build_network(self):
        dic_input_node = {"feat1": Input(shape=(8,), name="input_cur_phase"),
                          "feat2": Input(shape=(12,), name="input_feat2")}

        p = Activation('sigmoid')(Embedding(2, 4, input_length=8)(dic_input_node["feat1"]))
        d = Dense(4, activation="sigmoid", name="num_vec_mapping")
        dic_lane = {}
        dic_index = {
            "WL": 0,
            "WT": 1,
            "EL": 3,
            "ET": 4,
            "NL": 6,
            "NT": 7,
            "SL": 9,
            "ST": 10,
        }
        for i, m in enumerate(self.dic_traffic_env_conf["list_lane_order"]):
            idx = dic_index[m]
            tmp_vec = d(
                Lambda(slice_tensor, arguments={"index": idx}, name="vec_%d" % i)(dic_input_node["feat2"]))
            tmp_phase = Lambda(slice_tensor, arguments={"index": i}, name="phase_%d" % i)(p)
            dic_lane[m] = concatenate([tmp_vec, tmp_phase], name="lane_%d" % i)

        if self.num_actions == 8 or self.num_actions == 4:
            list_phase_pressure = []
            lane_embedding = Dense(16, activation="relu", name="lane_embedding")
            for phase in self.dic_traffic_env_conf["PHASE_LIST"]:
                m1, m2 = phase.split("_")
                list_phase_pressure.append(add([lane_embedding(dic_lane[m1]),
                                                lane_embedding(dic_lane[m2])], name=phase))
            # [batch, 4, 3], initialed zeros
            constant_o = Lambda(relation, arguments={"phase_list": self.dic_traffic_env_conf["PHASE_LIST"]},
                                name="constant_o")(dic_input_node["feat2"])
        relation_embedding = Embedding(2, 4, name="relation_embedding")(constant_o)

        # rotate the phase pressure
        list_phase_pressure_recomb = []
        num_phase = len(list_phase_pressure)

        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    list_phase_pressure_recomb.append(
                        concatenate([list_phase_pressure[i], list_phase_pressure[j]],
                                    name="concat_compete_phase_%d_%d" % (i, j)))

        list_phase_pressure_recomb = concatenate(list_phase_pressure_recomb, name="concat_all")
        if num_phase == 8:
            feature_map = Reshape((8, 7, 32))(list_phase_pressure_recomb)
        else:
            feature_map = Reshape((4, 3, 32))(list_phase_pressure_recomb)
        lane_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1),
                           activation="relu", name="lane_conv")(feature_map)
        relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                               name="relation_conv")(relation_embedding)
        combine_feature = multiply([lane_conv, relation_conv], name="combine_feature")

        # [batch, 4, 3, D_DENSE]
        hidden_layer = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                              name="combine_conv")(combine_feature)
        # [batch, 4, 3, 1 ]
        before_merge = Conv2D(1, kernel_size=(1, 1), activation="linear", name="before_merge")(hidden_layer)
        if self.num_actions == 8:
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(Reshape((8, 7))(before_merge))
        else:
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(Reshape((4, 3))(before_merge))

        network = Model(inputs=[dic_input_node[feature_name] for feature_name in ["feat1", "feat2"]],
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()

        return network

    def convert_state_to_input(self, states):
        dic_state_feature_arrays = {}  # {feature1: [inter1, inter2,..], feature2: [inter1, inter 2...]}
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "cur_phase":
                    dic_state_feature_arrays[feature_name].append(self.dic_traffic_env_conf['PHASE'][s[feature_name][0]])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        state_input = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                       used_feature]
        return state_input

    def choose_action(self, is_test, states):
        dic_state_feature_arrays = {}  # {feature1: [inter1, inter2,..], feature2: [inter1, inter 2...]}
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "cur_phase":
                    dic_state_feature_arrays[feature_name].append(self.dic_traffic_env_conf['PHASE'][s[feature_name][0]])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        state_input = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                       used_feature]

        q_values = self.q_network.predict(state_input)
        # e-greedy
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = np.random.randint(len(q_values[0]), size=len(q_values))
        else:
            action = np.argmax(q_values, axis=1)

        import time
        start_time = time.time()
        if is_test:
            if self.dic_traffic_env_conf['NOISE_LEVEL'] == 0:   
                if self.dic_traffic_env_conf['NOISE_TYPE'] == 2:
                    timestep = int((self.dic_traffic_env_conf['NOISE_SCALE'] + 0.03) / 0.05 + 0.0001)
                    noise_rpt = self.num_agents
                    xs = self.convert_state_to_input(states)
                    actions = []
                    state_real_tensor = torch.tensor(xs[1].reshape(1, self.num_agents, -1)).to(self.dic_traffic_env_conf['device'])
                    rpt_state = torch.repeat_interleave(state_real_tensor, repeats=noise_rpt, dim=0)
                    rpt_noised_state = rpt_state + (2 * self.dic_traffic_env_conf['NOISE_SCALE']  * torch.rand_like(rpt_state) - self.dic_traffic_env_conf['NOISE_SCALE'] )
                    rpt_long_state_con_array = []
                    for k in range(self.num_agents):
                        if self.dic_traffic_env_conf['is_inference']:
                            cur_action = torch.tensor(np.eye(4)[np.array(action)]).to(self.dic_traffic_env_conf['device'])[k]
                            rpt_con_action = torch.repeat_interleave(cur_action.reshape(1, -1), repeats=noise_rpt, dim=0)
                            rpt_long_state_con = torch.repeat_interleave(self.long_state_con[k].reshape(1,self.long_state_con.shape[1], -1), repeats=noise_rpt, dim=0)
                            rpt_state_predict = self.inference_model.denoise_state(rpt_noised_state[:,k,:].type(torch.float32), rpt_con_action.type(torch.float32), rpt_long_state_con.type(torch.float32), timestep, method="mean").cpu()
                            rpt_noised_state[:,k,:] = rpt_state_predict
                            rpt_long_state_con_array.append(rpt_state_predict)
                    state_phase = torch.tensor(xs[0].reshape(1,self.num_agents,-1)).to(self.dic_traffic_env_conf['device'])
                    rpt_state_phase = torch.repeat_interleave(state_phase, repeats=noise_rpt, dim=0)
                    x = torch.tensor(xs[1]).to(self.dic_traffic_env_conf['device'])
                    for k in range(self.num_agents):
                        rpt_noised_action = self.q_network([rpt_state_phase[:,k,:].cpu().numpy(),rpt_noised_state[:,k,:].cpu().numpy()])
                        # 输出action
                        tmp_action = np.argmax(rpt_noised_action, axis=1)
                        # 从重复的噪声中选择q最小的，索引，从tmp_action中根据索引找到action
                        tmp_q = []
                        for j in range(noise_rpt):
                            tmp_q.append(rpt_noised_action[j,tmp_action[j]].cpu().numpy().tolist())
                        idx = tmp_q.index(min(tmp_q))
                        actions.append(tmp_action[idx])
                        if self.dic_traffic_env_conf['is_inference']:
                            x[k] = rpt_long_state_con_array[k][idx]
                    if self.dic_traffic_env_conf['is_inference']:
                        self.long_state_con = torch.cat([self.long_state_con[:, 1:], x.view(self.long_state_con.shape[0],1,-1)], dim=1)

                    action = actions
                if self.dic_traffic_env_conf['NOISE_TYPE'] == 3:
                    timestep = int((self.dic_traffic_env_conf['NOISE_SCALE'] + 0.03) / 0.05 + 0.0001)
                    noise_rpt = self.num_agents
                    xs = self.convert_state_to_input(states)
                    actions = []
                    state_real_tensor = torch.tensor(xs[1].reshape(1, self.num_agents, -1)).to(self.dic_traffic_env_conf['device'])
                    rpt_state = torch.repeat_interleave(state_real_tensor, repeats=noise_rpt, dim=0)
                    rpt_noised_state = rpt_state + (2 * self.dic_traffic_env_conf['NOISE_SCALE']  * torch.rand_like(rpt_state) - self.dic_traffic_env_conf['NOISE_SCALE'] )
                    rpt_long_state_con_array = []
                    for k in range(self.num_agents):
                        if self.dic_traffic_env_conf['is_inference']:
                            cur_action = torch.tensor(np.eye(4)[np.array(action)]).to(self.dic_traffic_env_conf['device'])[k]
                            rpt_con_action = torch.repeat_interleave(cur_action.reshape(1, -1), repeats=noise_rpt, dim=0)
                            rpt_long_state_con = torch.repeat_interleave(self.long_state_con[k].reshape(1,self.long_state_con.shape[1], -1), repeats=noise_rpt, dim=0)
                            rpt_state_predict = self.inference_model.denoise_state(rpt_noised_state[:,k,:].type(torch.float32), rpt_con_action.type(torch.float32), rpt_long_state_con.type(torch.float32), timestep, method="mean").cpu()
                            rpt_noised_state[:,k,:] = rpt_state_predict
                            rpt_long_state_con_array.append(rpt_state_predict)
                    state_phase = torch.tensor(xs[0].reshape(1,self.num_agents,-1)).to(self.dic_traffic_env_conf['device'])
                    rpt_state_phase = torch.repeat_interleave(state_phase, repeats=noise_rpt, dim=0)
                    
                    x = torch.tensor(xs[1]).to(self.dic_traffic_env_conf['device'])
                    
                    for k in range(self.num_agents):
                        rpt_noised_all_q = self.q_network([rpt_state_phase[:,k,:].cpu().numpy(),rpt_noised_state[:,k,:].cpu().numpy()])
                        original_action_all_q = self.q_network([rpt_state_phase[:,k,:].cpu().numpy(), rpt_state[:,k,:].cpu().numpy()])
                        # 输出action
                        rpt_noised_action_q = rpt_noised_all_q
                        original_action_q = np.mean(original_action_all_q, axis=0)
                        rpt_action = np.repeat(original_action_q.reshape(1, -1), repeats=noise_rpt, axis=0)
                        difference = np.sum(np.abs(rpt_action - rpt_noised_action_q), 1)
                        idx = np.argmax(difference)
                        action = np.argmax(rpt_noised_action_q[idx])
                        actions.append(action)
                        if self.dic_traffic_env_conf['is_inference']:
                            x[k] = rpt_long_state_con_array[k][idx]
                    if self.dic_traffic_env_conf['is_inference']:
                        self.long_state_con = torch.cat([self.long_state_con[:, 1:], x.view(self.long_state_con.shape[0],1,-1)], dim=1)

                    action = actions
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000

        print(f"Inference time: {elapsed_time_ms:.2f} ms") 
        return action

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        # used_feature = ["phase_2", "phase_num_vehicle"]
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        _state = [[], []]
        _next_state = [[], []]
        _action = []
        _reward = []
        for i in range(len(sample_slice)):
            state, action, next_state, reward, _, _, _ = sample_slice[i]
            for feat_idx, feat_name in enumerate(used_feature):
                _state[feat_idx].append(state[feat_name])
                _next_state[feat_idx].append(next_state[feat_name])
            _action.append(action)
            _reward.append(reward)
        # well prepared states
        _state2 = [np.array(ss) for ss in _state]
        _next_state2 = [np.array(ss) for ss in _next_state]
        self.replay_buffer = ReplayBuffer((np.swapaxes(_state2[1].reshape(-1, self.num_agents, 12),0,1), np.eye(4)[np.array(_action).reshape(self.num_agents,-1)], np.swapaxes(_next_state2[1].reshape(-1, self.num_agents, 12),0,1), np.array(_reward).reshape(self.num_agents,-1)[:,:, np.newaxis]),
                                    int(self.dic_traffic_env_conf['RUN_COUNTS']/self.dic_traffic_env_conf['MIN_ACTION_TIME']), self.device)

        cur_qvalues = self.q_network.predict(_state2)
        next_qvalues = self.q_network_bar.predict(_next_state2)
        # [batch, 4]
        target = np.copy(cur_qvalues)

        for i in range(len(sample_slice)):
            target[i, _action[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(next_qvalues[i, :])
        self.Xs = _state2
        self.Y = target