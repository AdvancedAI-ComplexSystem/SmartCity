"""
Advanced-MaxPressure agent.
observation: [ traffic_movement_pressure_queue_efficient, enter running part].
"""

import numpy as np
import os
from .agent import Agent
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Layer, Reshape
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from utils.make_mask_noise import make_guassion_noise
from inferences.rl_dynamic_state_filling import Diffusion_Predictor
from utils.batch_buffer import ReplayBuffer
import torch

def build_memory():
    return []

class AdvancedMaxPressureAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):

        super(AdvancedMaxPressureAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.current_phase_time = 0
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.action = None
        self.weight = dic_traffic_env_conf["W"]
        if self.phase_length == 4:
            self.DIC_PHASE_MAP_4 = {  # for 4 phase
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                0: 0
            }
        elif self.phase_length == 8:
            self.DIC_PHASE_MAP = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                0: 0
            }
        self.device = dic_traffic_env_conf['device']
        
        self.timestep = None
        if cnt_round == 0:
            pass
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))
       
        self.inference_model = Diffusion_Predictor(self.len_feature, self.phase_length, self.device, dic_traffic_env_conf['inference_config'], log_writer=False)
        self.inference_model.load_model(os.path.join(self.dic_traffic_env_conf['sota_path'], "round_{0}_int".format(cnt_round+70, self.intersection_id)),  int(self.device[-1]))

    def _cal_len_feature(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                N += 8
            else:
                N += 12
        return N


    def choose_action(self, count, state):
        """
        As described by the definition, use traffic_movement_pressure
        to calcualte the pressure of each phase.
        """

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]

        if state["cur_phase"][0] == -1:
            return self.action

        #  WT_ET
        tr_mo_pr = np.array(state[used_feature[0]])
        l_e_r = np.array(state[used_feature[1]])

        phase_1 = tr_mo_pr[1] + tr_mo_pr[4]
        # NT_ST
        phase_2 = tr_mo_pr[7] + tr_mo_pr[10]
        # WL_EL
        phase_3 = tr_mo_pr[0] + tr_mo_pr[3]
        # NL_SL
        phase_4 = tr_mo_pr[6] + tr_mo_pr[9]

        d1 = l_e_r[1] + l_e_r[4]
        d2 = l_e_r[7] + l_e_r[10]
        d3 = l_e_r[0] + l_e_r[3]
        d4 = l_e_r[6] + l_e_r[9]

        phase_p = [phase_1, phase_2, phase_3, phase_4]
        phase_d = [d1, d2, d3, d4]

        if self.phase_length == 4:

            if self.action is None:

                self.action = np.argmax(phase_p)
            elif phase_d[self.action] * self.weight >= np.max(phase_p):
                pass
            else:
                self.action = np.argmax(phase_p)
        elif self.phase_length == 8:
            #  WL_WT
            phase_5 = tr_mo_pr[0] + tr_mo_pr[1]
            # EL_ET
            phase_6 = tr_mo_pr[3] + tr_mo_pr[4]
            # SL_ST
            phase_7 = tr_mo_pr[9] + tr_mo_pr[10]
            # NL_NT
            phase_8 = tr_mo_pr[6] + tr_mo_pr[7]
            self.action = np.argmax([phase_1, phase_2, phase_3, phase_4, phase_5, phase_6, phase_7, phase_8])

        if state["cur_phase"][0] == self.action:
            self.current_phase_time += 1
        else:
            self.current_phase_time = 0

        return self.action

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.inference_model.load_model(os.path.join(self.dic_path["PATH_TO_MODEL"], file_name[:12]))
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):

        self.inference_model.save_model(os.path.join(self.dic_path["PATH_TO_MODEL"], file_name[:12]))

    def train_network(self):    
        if self.dic_traffic_env_conf['is_test'] is not True:
            self.inference_model.train(self.replay_buffer, self.dic_traffic_env_conf['inference_epoch'], self.replay_buffer.batch_sample, self.dic_traffic_env_conf['log_writer'])
    
    def convert_state_to_input(self, s):
        """
        s: [state1, state2, ..., staten]
        """
        # TODO
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        feats0 = []
        for i in range(self.num_agents):
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
        feats = np.array([feats0])
        return feats

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
        # state : [feat1, feat2]
        # feati : [agent1, agent2, ..., agentn]
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]

        for i in range(slice_size):
            for j in range(self.num_agents):
                state, action, next_state, reward, _, _, _ = memory[j][i]
                _action[j].append(action)
                _reward[j].append(reward)
                # TODO
                _state[j].append(self._concat_list([state[used_feature[i]] for i in range(len(used_feature))]))
                _next_state[j].append(self._concat_list([next_state[used_feature[i]] for i in range(len(used_feature))]))

        # [batch, 1, dim] -> [batch, agent, dim]
        _state2 = np.concatenate([np.array(ss) for ss in _state], axis=1)

        _next_state2 = np.concatenate([np.array(ss) for ss in _next_state], axis=1)

        self.replay_buffer = ReplayBuffer((np.swapaxes(_state2,0,1), np.eye(4)[np.array(_action)], np.swapaxes(_next_state2,0,1), np.array(_reward)[:,:, np.newaxis]),
                                    int(self.dic_traffic_env_conf['RUN_COUNTS']/self.dic_traffic_env_conf['MIN_ACTION_TIME']), self.device)
