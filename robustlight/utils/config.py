from models.fixedtime_agent import FixedtimeAgent
from models.maxpressure_agent import MaxPressureAgent
from models.mplight_agent import MPLightAgent
from models.advanced_mplight_agent import AdvancedMPLightAgent
from models.advanced_maxpressure_agent import AdvancedMaxPressureAgent
from models.colight_agent_dsi import CoLightDSIAgent

DIC_AGENTS = {
    "Fixedtime": FixedtimeAgent,
    "MaxPressure": MaxPressureAgent,
    "AdvancedMaxPressure": AdvancedMaxPressureAgent,
    "EfficientColight": CoLightDSIAgent,
    "EfficientMPLight": MPLightAgent,
    "AdvancedMPLight": AdvancedMPLightAgent,
    "ColightDSI": CoLightDSIAgent
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {

    "LIST_MODEL": ["Fixedtime",  "MaxPressure", "EfficientMaxPressure", "AdvancedMaxPressure",
                   "EfficientPressLight", "EfficientColight", "EfficientMPLight",
                   "AdvancedMPLight", "AdvancedColight", "AdvancedDQN","ColightDSI"],
    "LIST_MODEL_NEED_TO_UPDATE": ["EfficientPressLight", "EfficientColight", "EfficientMPLight",
                                  "AdvancedMPLight", "AdvancedColight", "AdvancedDQN","ColightDSI", "AdvancedMaxPressure", "MaxPressure"],

    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 15,
    "MEASURE_TIME": 15,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle",
        "lane_num_vehicle_downstream",
        "traffic_movement_pressure_num",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "pressure",
        "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 4704,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}


DMBP_config = {
    "algo": "ql", # "bc", "ql"
    "eval_freq" : 5000,
    "max_timestep": int(3e5),
    "start_testing": int(0),
    "checkpoint_start": int(2e5),
    "checkpoint_every": int(1e4),

    # "eval_freq": 200,
    # "max_timestep": int(600),
    # "start_testing": int(400),
    # "checkpoint_start": int(100),
    # "checkpoint_every": int(100),

    "gamma": 0.99,
    "tau": 0.005,
    "eta": 1.0,
    "lr_decay": True,
    "max_q_backup": True,
    "step_start_ema": 1000,
    "ema_decay": 0.995,
    "update_ema_every": 5,
    "T": 100,

    "beta_schedule": 'self-defined2',
    "beta_training_mode": 'all',   # 'all' or 'partial'
    'loss_training_mode': 'no_act2',    # 'normal' or 'noise' or 'no_act' or 'no_act2'
    "predict_epsilon": True,
    "data_usage": 1.0,
    'ms': 'offline',
    'gn': 10.0,

    # Long Term Buffer Parameter Definition
    "condition_length": 4,
    "T-scheme": "same",  # "random" or "same"

    "non_markovian_step": 6,

    # Attention Hyperparameters
    "attn_hidden_layer": 2,
    "attn_hidden_dim": 128,
    "attn_embed_dim": 64,

    "lr": 3e-4,
    "alpha": 0.2,
    "batch_size": 64,
    "hidden_size": 256,
    "embed_dim": 64,
    "reward_tune": "no",
}