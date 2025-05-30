from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from .network_agent import NetworkAgent

class SimpleDQN(NetworkAgent):
    def build_network(self):
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        dic_input_node = {}
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                _shape = (8,)
            else:
                _shape = (12,)
            dic_input_node[feat_name] = Input(shape=_shape, name="input_" + feat_name)
        
        # concatenate features
        list_all_flatten_feature = []
        for feature_name in used_feature:
            list_all_flatten_feature.append(dic_input_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")
        
        # shared dense layer
        shared_dense = Dense(self.dic_agent_conf["D_DENSE"], activation="sigmoid",
                           name="shared_hidden")(all_flatten_feature)

        q_values = Dense(self.num_actions, activation="linear")(shared_dense)

        network = Model(inputs=[dic_input_node[feature_name]
                              for feature_name in used_feature],
                       outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                       loss="mean_squared_error")
        network.summary()

        return network 