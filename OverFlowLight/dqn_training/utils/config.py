DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
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

DIC_TRAFFIC_ENV_CONF = {
    "PHASE": {
        1: [1, 0, 1, 0, 0, 0, 0, 0],
        2: [0, 1, 0, 1, 0, 0, 0, 0],
        3: [0, 0, 0, 0, 1, 0, 1, 0],
        4: [0, 0, 0, 0, 0, 1, 0, 1]
    },
    "LANE": ["W", "E", "N", "S"],
    "LIST_STATE_FEATURE": ["cur_phase", "lane_num_vehicle"],
    "BINARY_PHASE_EXPANSION": True
}

DIC_PATH = {
    "PATH_TO_MODEL": "model",
    "PATH_TO_WORK_DIRECTORY": "records"
} 