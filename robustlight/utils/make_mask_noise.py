import numpy as np 

def make_guassion_noise(states, noise_scale):
    for state in states:
        for key in state.keys():
            if key == 'cur_phase' or key == 'time_this_phase' or key == 'adjacency_matrix':
                continue 
            else:
                state[key] = (state[key] + noise_scale * np.random.randn(len(state[key]))).tolist()
    return states


def make_U_rand_noise(states, noise_scale):
    for state in states:
        for key in state.keys():
            if key == 'cur_phase' or key == 'time_this_phase' or key == 'adjacency_matrix':
                continue 
            else:
                state[key] = (state[key] + 2 * noise_scale * np.random.randn(len(state[key]))- noise_scale).tolist()
    return states


