from .pipeline import Pipeline
from .oneline import OneLine
from . import config
import os
import json
import shutil
import copy
import torch 
import numpy as np
import random
import tensorflow as tf

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result


def pipeline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_agent_conf=dic_agent_conf,
                   dic_traffic_env_conf=dic_traffic_env_conf,
                   dic_path=dic_path
                   )
    ppl.run(multi_process=False)

    print("pipeline_wrapper end")
    return


def oneline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path):
    oneline = OneLine(dic_agent_conf=dic_agent_conf,
                      dic_traffic_env_conf=merge(config.dic_traffic_env_conf, dic_traffic_env_conf),
                      dic_path=merge(config.DIC_PATH, dic_path)
                      )
    oneline.train()
    return


def setup_seed(seed=1024): # After doing this, the Training results will always be the same for the same seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(42)


