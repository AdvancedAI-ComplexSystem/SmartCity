from .config import DIC_AGENTS
from copy import deepcopy
from .cityflow_env import CityFlowEnv
import json
import os
from .make_mask_noise import *
import torch
from utils.psnr import PSNR


def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d" % cnt_round
    dic_path = {"PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir}
    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    if os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    if dic_traffic_env_conf["MODEL_NAME"] in dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0
        dic_agent_conf["MIN_EPSILON"] = 0

    agents = []
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_traffic_env_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(i)
        )
        agents.append(agent)
    if not dic_traffic_env_conf['is_test']:
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)
    env = CityFlowEnv(
        path_to_log=path_to_log,
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=dic_traffic_env_conf
    )

    done = False

    step_num = 0

    total_time = dic_traffic_env_conf["RUN_COUNTS"]
    state = env.reset()

    xs = agents[i].convert_state_to_input(state)
    
    
    if dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight", "AdvancedColight", "ColightDSI"]:
        long_state_con = torch.zeros([xs[0][0].shape[0], 4, agents[i].len_feature]).to(dic_traffic_env_conf['device'])
    elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMaxPressure", "MaxPressure"]:
        long_state_con = torch.zeros([xs[0].shape[0], 4, agents[i].len_feature]).to(dic_traffic_env_conf['device'])
    elif dic_traffic_env_conf['MODEL_NAME'] in ["EfficientMPLight", "AdvancedMPLight"]:
        long_state_con = torch.zeros([xs[1].shape[0], 4, agents[i].len_feature]).to(dic_traffic_env_conf['device'])
    psnr = PSNR(255.0).to(dic_traffic_env_conf['device'])  
    a = np.zeros((int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]), xs[0].shape[1], xs[0].shape[2]))
    b = a.copy()
    cnt = 0
    while not done and step_num < int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]):
        action_list = []
        import time
        start_time = time.time()
        for i in range(dic_traffic_env_conf["NUM_AGENTS"]):

            if dic_traffic_env_conf["MODEL_NAME"] in ["EfficientPressLight", "EfficientColight", "EfficientMPLight",
                                                        "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "ColightDSI"]:
                one_state = state
                action_list = agents[i].choose_action(True, one_state)

            else:
                one_state = state[i]
                action = agents[i].choose_action(True, one_state)
                action_list.append(action)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"TSC time: {elapsed_time_ms:.2f} ms") 
        next_state, reward, done, _ = env.step(action_list)
        
        a[cnt,:,:] = agents[i].convert_state_to_input(next_state)[0]
        # TODO: 做一些mask（1）对每个交叉口某些维度做mask （2）对交叉口加噪声 (3) 对某些交叉口做mask
        if dic_traffic_env_conf['NOISE_LEVEL'] == 0 and dic_traffic_env_conf['NOISE_TYPE'] != 2 and dic_traffic_env_conf['NOISE_TYPE'] != 3:
            if dic_traffic_env_conf['NOISE_TYPE'] == 0:
                next_state_noise = make_guassion_noise(next_state, dic_traffic_env_conf['NOISE_SCALE'])
            elif dic_traffic_env_conf['NOISE_TYPE'] == 1:
                next_state_noise = make_U_rand_noise(next_state, dic_traffic_env_conf['NOISE_SCALE'])

            xs = agents[i].convert_state_to_input(next_state_noise)
            if dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight", "AdvancedColight", "ColightDSI"]:
                x=torch.tensor(xs[0][0]).to(dic_traffic_env_conf['device'])      
            elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMaxPressure","MaxPressure"]:
                x=torch.tensor(xs[0]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["EfficientMPLight"]:
                x=torch.tensor(xs[1]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMPLight"]:
                tmp_xs = np.concatenate([xs[1], xs[2]], axis=1)
                x=torch.tensor(tmp_xs).to(dic_traffic_env_conf['device'])
            
               

        # 对交叉口某些维度做mask
        elif dic_traffic_env_conf['NOISE_LEVEL'] == 1:
            # 根据ratio决定哪些维度要mask
            if dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedColight", "ColightDSI"]:
                xs = agents[i].convert_state_to_input(next_state)
                mask = np.ones(xs[0][0].shape)
                # 某个方向上的传感器坏了,一共四个方向
                mask_dim = []
                for dire in dic_traffic_env_conf['NOISE_DIRECTION']:
                    mask_dim.extend(dic_traffic_env_conf['index_maps'][dire])
                mask_dim_queue_length = [i+8 for i in mask_dim]
                mask_dim_running_length = [i+20 for i in mask_dim]
                mask[:, mask_dim_queue_length] = 0
                mask[:, mask_dim_running_length] = 0
                
                #compare_to difflight
                # mask[:, 10] = 0
                # mask[:, 24] = 0
                # # 将第0行和第1行中第9列之后的数据设置为0
                # mask[6, 10:] = 0
                # mask[7, 10:] = 0

                mask_tensor = torch.FloatTensor(mask).to(dic_traffic_env_conf['device']).type(torch.float32)
                xs[0][0] = mask * xs[0][0]
                x=torch.tensor(xs[0][0]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMaxPressure"]:
                xs = agents[i].convert_state_to_input(next_state)
                mask = np.ones(xs[0].shape)
                # 某个方向上的传感器坏了,一共四个方向
                mask_dim = []
                for dire in dic_traffic_env_conf['NOISE_DIRECTION']:
                    mask_dim.extend(dic_traffic_env_conf['index_maps'][dire])
                mask_dim_queue_length = [i for i in mask_dim]
                mask_dim_running_length = [i+12 for i in mask_dim ]
                mask[:, mask_dim_queue_length] = 0
                mask[:, mask_dim_running_length] = 0
                mask_tensor = torch.FloatTensor(mask).to(dic_traffic_env_conf['device']).type(torch.float32)
                xs[0] = mask * xs[0]
                x=torch.tensor(xs[0]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["MaxPressure"]:
                xs = agents[i].convert_state_to_input(next_state)
                mask = np.ones(xs[0].shape)
                # 某个方向上的传感器坏了,一共四个方向
                mask_dim = []
                for dire in dic_traffic_env_conf['NOISE_DIRECTION']:
                    mask_dim.extend(dic_traffic_env_conf['index_maps'][dire])
                mask_dim_queue_length = [i for i in mask_dim]
                mask[:, mask_dim_queue_length] = 0
                mask_tensor = torch.FloatTensor(mask).to(dic_traffic_env_conf['device']).type(torch.float32)
                xs[0] = mask * xs[0]
                x=torch.tensor(xs[0]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["EfficientMPLight"]:
                xs = agents[i].convert_state_to_input(next_state)
                mask = np.ones(xs[1].shape)
                # 某个方向上的传感器坏了,一共四个方向
                mask_dim = []
                for dire in dic_traffic_env_conf['NOISE_DIRECTION']:
                    mask_dim.extend(dic_traffic_env_conf['index_maps'][dire])
                mask_dim_queue_length = [i for i in mask_dim]
                mask[:, mask_dim_queue_length] = 0
                mask_tensor = torch.FloatTensor(mask).to(dic_traffic_env_conf['device']).type(torch.float32)
                xs[1] = mask * xs[1]
                x=torch.tensor(xs[1]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight"]:
                xs = agents[i].convert_state_to_input(next_state)
                mask = np.ones(xs[0][0].shape)
                # 某个方向上的传感器坏了,一共四个方向
                mask_dim = []
                for dire in dic_traffic_env_conf['NOISE_DIRECTION']:
                    mask_dim.extend(dic_traffic_env_conf['index_maps'][dire])
                mask_dim_queue_length = [i+8 for i in mask_dim]
                mask[:, mask_dim_queue_length] = 0
                mask_tensor = torch.FloatTensor(mask).to(dic_traffic_env_conf['device']).type(torch.float32)
                xs[0][0] = mask * xs[0][0]
                x=torch.tensor(xs[0][0]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMPLight"]:
                xs = agents[i].convert_state_to_input(next_state)
                tmp_xs = np.concatenate([xs[1], xs[2]], axis=1)
                mask = np.ones((agents[i].num_agents, 24))
                # 某个方向上的传感器坏了,一共四个方向
                mask_dim = []
                for dire in dic_traffic_env_conf['NOISE_DIRECTION']:
                    mask_dim.extend(dic_traffic_env_conf['index_maps'][dire])
                mask_dim_queue_length = [i for i in mask_dim]
                mask_dim_running_length = [i+12 for i in mask_dim ]
                mask[:, mask_dim_queue_length] = 0
                mask[:, mask_dim_running_length] = 0
                mask_tensor = torch.FloatTensor(mask).to(dic_traffic_env_conf['device']).type(torch.float32)
                tmp_xs = mask * tmp_xs
                x=torch.tensor(tmp_xs).to(dic_traffic_env_conf['device'])
        
        import time
        start_time = time.time()
        if dic_traffic_env_conf['is_inference'] and dic_traffic_env_conf['NOISE_TYPE'] != 2 and dic_traffic_env_conf['NOISE_TYPE'] != 3:
            # TODO：做一些predict，针对三种情况进行预测

            xs_recover = agents[i].convert_state_to_input(state)
            if dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight", "AdvancedColight", "ColightDSI"]:
                x_recover=torch.tensor(xs_recover[0][0]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMaxPressure", "MaxPressure"]:
                x_recover=torch.tensor(xs_recover[0]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["EfficientMPLight"]:
                x_recover=torch.tensor(xs_recover[1]).to(dic_traffic_env_conf['device'])
            elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMPLight"]:
                tmp_xs = np.concatenate([xs_recover[1], xs_recover[2]], axis=1)
                x_recover=torch.tensor(tmp_xs).to(dic_traffic_env_conf['device'])

            long_state_con = torch.cat([long_state_con[:, 1:], x_recover.view(long_state_con.shape[0],1,-1)], dim=1)
            action = torch.tensor(np.eye(4)[np.array(action_list)]).to(dic_traffic_env_conf['device'])
            if dic_traffic_env_conf['NOISE_LEVEL'] == 0:
                if dic_traffic_env_conf['NOISE_TYPE'] == 0:
                    timestep = int((dic_traffic_env_conf['NOISE_SCALE'] + 0.02) / 0.04)
                elif dic_traffic_env_conf['NOISE_TYPE'] == 1:
                    timestep = int((dic_traffic_env_conf['NOISE_SCALE'] + 0.04) / 0.06)

                #timestep = 20
                if dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight", "AdvancedColight", "ColightDSI"]:
                    #for k in range(action.shape[0]):
                    #    xs[0][0][k] = agents[i].inference_model.denoise_state(x[k].view(1,-1).type(torch.float32), action[k].view(1,-1).type(torch.float32), long_state_con[k].view(1,4,-1).type(torch.float32), timestep, method="mean").cpu()
                    xs[0][0] = agents[i].inference_model.denoise_state(x.type(torch.float32), action.type(torch.float32), long_state_con.type(torch.float32), timestep, method="mean").cpu()
                
                elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMaxPressure", "MaxPressure"]:
                    xs[0] = agents[i].inference_model.denoise_state(x.type(torch.float32), action.type(torch.float32), long_state_con.type(torch.float32), timestep, method="mean").cpu()
                elif dic_traffic_env_conf['MODEL_NAME'] in ["EfficientMPLight"]:
                    xs[1] = agents[i].inference_model.denoise_state(x.type(torch.float32), action.type(torch.float32), long_state_con.type(torch.float32), timestep, method="mean").cpu()
                elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMPLight"]:
                    tmp_xs = agents[i].inference_model.denoise_state(x.type(torch.float32), action.type(torch.float32), long_state_con.type(torch.float32), timestep, method="mean").cpu()
                
            elif dic_traffic_env_conf['NOISE_LEVEL'] == 1:
                if dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight", "AdvancedColight", "ColightDSI"]:
                    xs[0][0]= agents[i].inference_model.demask_state(x.type(torch.float32), action.type(torch.float32), long_state_con.type(torch.float32), mask_tensor,2)
                elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMaxPressure", "MaxPressure"]:
                    xs[0]= agents[i].inference_model.demask_state(x.type(torch.float32), action.type(torch.float32), long_state_con.type(torch.float32), mask_tensor,2)
                elif dic_traffic_env_conf['MODEL_NAME'] in ["EfficientMPLight"]:
                    xs[1]= agents[i].inference_model.demask_state(x.type(torch.float32), action.type(torch.float32), long_state_con.type(torch.float32), mask_tensor,2)
                elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMPLight"]:
                    tmp_xs = agents[i].inference_model.demask_state(x.type(torch.float32), action.type(torch.float32), long_state_con.type(torch.float32), mask_tensor,2)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000

        #print(f"Inference time: {elapsed_time_ms:.2f} ms") 
        # 为feature赋值
        used_feature = dic_traffic_env_conf["LIST_STATE_FEATURE"]
        if dic_traffic_env_conf['is_inference']  and step_num >= 4 and dic_traffic_env_conf['NOISE_TYPE'] != 2 and dic_traffic_env_conf['NOISE_TYPE'] != 3:
            t = 0
            for s in next_state:
                for feature in used_feature:
                    if feature == "cur_phase" or feature == "adjacency_matrix":
                        continue
                    elif feature == "traffic_movement_pressure_queue_efficient":
                        if dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight", "AdvancedColight", "ColightDSI"]:
                            s[feature] = xs[0][0][t,8:20].tolist()
                        elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMaxPressure"]:
                            s[feature] = xs[0][t,0:12].tolist()
                        elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMPLight"]:
                            s[feature] = tmp_xs[t,0:12].tolist()   

                    elif feature == "lane_enter_running_part":
                        if dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight", "AdvancedColight", "ColightDSI"]:
                            s[feature] = xs[0][0][t,20:].tolist()
                        elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMaxPressure"]:
                            s[feature] = xs[0][t,12:].tolist()
                        elif dic_traffic_env_conf['MODEL_NAME'] in ["AdvancedMPLight"]:
                            s[feature] = tmp_xs[t,12:].tolist()   

                    elif feature == 'traffic_movement_pressure_num':
                        s[feature] = xs[1][t,:].tolist()
                    elif feature == 'lane_num_vehicle':
                        s[feature] = xs[0][0][t,8:20].tolist()
                    elif feature == 'traffic_movement_pressure_queue':
                        s[feature] = xs[0][t,:].tolist()
                        
                t += 1
                
        b[cnt,:,:] = xs[0]
        state = next_state
        step_num += 1
    a, b = torch.tensor(a).to(dic_traffic_env_conf['device']),torch.tensor(b).to(dic_traffic_env_conf['device'])
        
    psnr_value = psnr(postprocess(a), \
                        postprocess(b))
    mae = (torch.sum(torch.abs(a - b)) / torch.sum(a)).float()
    print(f'psnr {psnr_value.item()}')
    print(f'mae {mae}')
    env.batch_log_2()
    env.end_cityflow()
    
    print(f'average waiting time is {env.get_avt()}')


def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0

    return img