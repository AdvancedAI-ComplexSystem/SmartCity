from utils.utils import pipeline_wrapper, merge, setup_seed
from utils import config
import time
from multiprocessing import Process
import argparse
import os
from utils.config import DMBP_config
import wandb
import torch
import numpy as np 
import random
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo",       type=str,               default='no_Advanced_maxpressure_guassian_4')
    parser.add_argument("-model",       type=str,               default="AdvancedMaxPressure")
    parser.add_argument("-eightphase", action="store_true",     default=False)
    parser.add_argument("-multi_process", action="store_true",  default=False)
    parser.add_argument("-workers",     type=int,               default=3)
    parser.add_argument("-gen",        type=int,            default=1)
    parser.add_argument("-hangzhou",    action="store_true",    default=True)
    parser.add_argument("-jinan",       action="store_true",    default=True)
    parser.add_argument("-new_york",       action="store_true", default=True)
    return parser.parse_args()


def main(in_args=None):
    if in_args.hangzhou:
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json",
                             #"anon_4_4_hangzhou_real_5816.json"
                             ]
        num_rounds = 10
        template = "Hangzhou"
    elif in_args.jinan:
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", 
                          #"anon_3_4_jinan_real_2000.json",
                            #"anon_3_4_jinan_real_2500.json"
                             ]
        num_rounds = 10
        template = "Jinan"
    elif in_args.new_york:
        count = 3600    
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json",
                            #"anon_28_7_newyork_real_triple.json"
                              ]
        num_rounds = 10
        template = "newyork_28_7"

    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)
    process_list = []

    for traffic_file in traffic_file_list:
        dic_traffic_env_conf_extra = {
            "W": 1,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "NUM_ROUNDS": num_rounds,
            "MODEL_NAME": in_args.model,
            "RUN_COUNTS": count,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "NUM_GENERATORS": in_args.gen,
            'is_test': True,
            "log_writer": False,
            "inference_epoch": 0,
            'is_inference': False,
            #"sota_path": 'model/AdvancedMaxPressure/' + traffic_file[:-5],
            "sota_path": 'model/AdvancedMaxPressure/anon_3_4_jinan_real',
            "NOISE_SCALE": 4,
            "NOISE_TYPE": 0, # 每一个level有不同类型的noise，譬如level为0噪声类型有guassion，qmin，uniform，action_diff
            "NOISE_LEVEL": 0, # 0为state上面加噪声，1是mask， 2是 mask掉其他交叉口的
            "device":"cuda:3",
            "index_maps" : {
                "W": [0, 1, 2],
                "E": [3, 4, 5],
                "N": [6, 7, 8],
                "S": [9, 10, 11]
            },
            "NOISE_DIRECTION": ["W"], # 某个方向出错了
            "inference_config": DMBP_config,


            "LIST_STATE_FEATURE": [
                "traffic_movement_pressure_queue_efficient",
                "lane_enter_running_part",
                "cur_phase",
            ],

            "DIC_REWARD_INFO": {
                "pressure": 0
            },
        }
        if in_args.eightphase:
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            }
            dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                        'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']

        dic_agent_conf_extra = {
            "FIXED_TIME": [15, 15, 15, 15],
        }
        dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]
        

     
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file + "_" +
                                          time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file + "_" +
                                                   time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net))
        }
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)
        deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)
        if in_args.multi_process:
            process_list.append(Process(target=pipeline_wrapper,
                                        args=(deploy_dic_agent_conf,
                                              deploy_dic_traffic_env_conf, deploy_dic_path))
                                )
        else:
            pipeline_wrapper(deploy_dic_agent_conf, deploy_dic_traffic_env_conf, deploy_dic_path)

    if in_args.multi_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < in_args.workers:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < in_args.workers:
                continue

        for p in list_cur_p:
            p.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)
