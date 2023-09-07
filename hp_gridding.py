import yaml
import numpy as np
import os
import argparse
from utils import predict_yaml, simulate, visualize_func, visualize_trace

""" Grid pos_space into [-1.2, -1.1], ..., [0.5, 0.6] """
""" Grid vel_space into [-0.07, 0.06], ..., [0.06, 0.07] """
""" Grid ctrl_space into [-1.0, -0.9], ..., [0.9, 1.0] """
""" m = 18 * 14 = 252, n = 20 """
min_pos = -1.2
max_pos = 0.6
step_pos = 0.02
min_speed = -0.07
max_speed = 0.07
step_speed = 0.002
min_ctrl = -1.0
max_ctrl = 1.0
step_ctrl = 0.1

# grid centers of (pos, speed) space
pos_centers = np.arange(start=min_pos + step_pos / 2, stop=max_pos, step=step_pos)
speed_centers = np.arange(start=min_speed + step_speed / 2, stop=max_speed, step=step_speed)
pos_grid, speed_grid = np.meshgrid(pos_centers, speed_centers)

pos_boundaries = np.arange(start=min_pos, stop=max_pos, step=step_pos)
speed_boundaries = np.arange(start=min_speed, stop=max_speed, step=step_speed)
ctrl_boundaries = np.arange(start=min_ctrl, stop=max_ctrl, step=step_ctrl)


# compute approximated ctrl at each input grid center
def compute_ctrl(model):
    approx_function = np.zeros(pos_grid.shape)
    for i in range(len(approx_function)):
        for j in range(len(approx_function[0])):
            coord = np.array([pos_grid[i][j], speed_grid[i][j]])
            ctrl = predict_yaml(model, coord)[0]
            ctrl = (ctrl - min_ctrl) // step_ctrl * step_ctrl + min_ctrl + step_ctrl / 2
            if ctrl > max_ctrl:
                ctrl = max_ctrl - step_ctrl / 2
            elif ctrl <= min_ctrl:
                ctrl = min_ctrl + step_ctrl / 2
            approx_function[i][j] = ctrl
    return approx_function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_pos", help="init pos that the controller fails", default=-0.5)
    parser.add_argument("--init_vel", help="init vel that the controller fails", default=0.0)
    parser.add_argument("--network_dir", help="directory of actual networks (yaml) files", default='networks')
    parser.add_argument("--network_name", help="name of actual network (yaml) file to be repaired", default='sig_8x16.yml')
    parser.add_argument("--approx_network_dir",
                        help="directory of approx networks (npy) files", default='approx_networks')
    args = parser.parse_args()

    init_pos = float(args.init_pos)
    init_vel = float(args.init_vel)

    # Compute the approximate gridded function
    filename_factual = os.path.join(args.network_dir, args.network_name)
    with open(filename_factual, 'rb') as f:
        model_factual = yaml.safe_load(f)
    f_factual = compute_ctrl(model_factual)
    approx_file_name = args.network_name.split('.')[0] + '.npy'
    np.save(os.path.join(args.approx_network_dir, approx_file_name), f_factual)

    # Visualize the grids as a heat map
    visualize_func(f_factual, title="Control Heatmap of Approx Factual f")

    # Simulate the trace with approximate gridded function
    outcome, obs_log = simulate(f_factual, init_pos=init_pos, init_vel=init_vel)

    # Visualize the trace
    visualize_trace(obs_log, title="Trace of Mountain Car Before Repair")
