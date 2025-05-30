import os
import numpy as np
from models.simple_dqn import SimpleDQN
from utils.config import DIC_BASE_AGENT_CONF, DIC_TRAFFIC_ENV_CONF, DIC_PATH

def create_test_memory():
    memory = []
    for _ in range(100):  # Create 100 test samples
        state = {
            "cur_phase": [np.random.randint(1, 5)],  # Random phase 1-4
            "lane_num_vehicle": np.random.randint(0, 10, size=12)  # Random vehicle counts
        }
        action = np.random.randint(0, 4)  # Random action 0-3
        next_state = {
            "cur_phase": [np.random.randint(1, 5)],
            "lane_num_vehicle": np.random.randint(0, 10, size=12)
        }
        reward = np.random.uniform(-1, 1)  # Random reward
        memory.append((state, action, next_state, reward, None, None, None))
    return memory

def test_dqn():
    # Create necessary directories
    os.makedirs(DIC_PATH["PATH_TO_MODEL"], exist_ok=True)
    os.makedirs(DIC_PATH["PATH_TO_WORK_DIRECTORY"], exist_ok=True)

    # Initialize DQN agent
    agent = SimpleDQN(
        dic_agent_conf=DIC_BASE_AGENT_CONF,
        dic_traffic_env_conf=DIC_TRAFFIC_ENV_CONF,
        dic_path=DIC_PATH,
        cnt_round=0,
        intersection_id="0"
    )

    # Create test memory
    memory = create_test_memory()

    # Test training
    print("Testing DQN training...")
    agent.prepare_Xs_Y(memory)
    agent.train_network()

    # Test action selection
    print("\nTesting action selection...")
    test_state = {
        "cur_phase": [1],
        "lane_num_vehicle": np.random.randint(0, 10, size=12)
    }
    action = agent.choose_action(0, test_state)
    print(f"Selected action: {action}")

    # Save model
    print("\nSaving model...")
    agent.save_network("test_model")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_dqn() 