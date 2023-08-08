import numpy as np
import matplotlib.pyplot as plt
from continuous_mountain_car import Continuous_MountainCarEnv

min_pos = -1.2
max_pos = 0.6
step_pos = 0.1
min_speed = -0.07
max_speed = 0.07
step_speed = 0.01
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


# sigmoid function
def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


# predict ctrl from actual model (yml file) on continuous input = (pos, speed)
def predict_yaml(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 0
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = sigmoid(curNeurons)
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return curNeurons


# predict ctrl from approx model (npy file) on continuous input = (pos, speed)
def predict_grids(approx_function, inputs):
    pos, speed = inputs
    j = int((pos - min_pos) // step_pos)
    i = int((speed - min_speed) // step_speed)
    ctrl = approx_function[i][j]
    return ctrl


# simulate with an approx function (npy file), return property satisfied or not
def simulate(approx_function, init_pos=-0.5, init_vel=0.0, render=True):
    env = Continuous_MountainCarEnv(init_pos=init_pos, init_vel=init_vel)
    env.seed(2)
    observation = env.reset()
    episodes = 110
    total_reward = 0
    observation_log = []

    for e in range(episodes):
        if render:
            env.render()
        action = predict_grids(approx_function, observation)
        action = [action]
        observation_log.append(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    runtime_property = e < (episodes - 1)
    return runtime_property, observation_log


# visualize approx function as heatmap
def visualize_func(approx_f: np.array, title="Control Output Heatmap"):
    ax = plt.axes()
    im = ax.imshow(approx_f, cmap='plasma', interpolation='nearest')
    plt.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1], boundaries=ctrl_boundaries)
    ax.set_yticks(np.arange(-.5, 15, 7))
    ax.set_yticklabels([-0.07, 0, 0.07])
    ax.set_ylabel("Velocity Cells", fontsize=15)
    ax.set_xticks(np.arange(-.5, 18, 6))
    ax.set_xticklabels([-1.2, -0.6, 0, 0.6])
    ax.set_xlabel("Position Cells", fontsize=15)
    ax.set_title(title, fontsize=15)
    plt.show()


# visualize trace from obs_log = [[pos, speed], [pos, speed], ...]
def visualize_trace(obs_log: list, title="Trace"):
    while len(obs_log) < 110:  # pad with last pos and 0 speed, i.e. car stays there if reach target early
        obs_log.append(np.array([obs_log[-1][0], 0]))
    obs_log = np.array(obs_log)
    x = np.arange(110)
    plt.figure(figsize=(1, 1), dpi=100)
    fig, axs = plt.subplots(2)
    axs[0].set_ylabel("Position", fontsize=15)
    axs[1].set_ylabel("Velocity", fontsize=15)
    axs[1].set_xlabel("time step", fontsize=15)
    axs[0].set_xlim(0, 110)
    axs[1].set_xlim(0, 110)
    axs[0].set_ylim(min_pos, max_pos)
    axs[1].set_ylim(min_speed, max_speed)
    axs[0].hlines(y=0.45, linewidth=2, xmin=0, xmax=110, color='r', linestyle='dashed')
    axs[0].plot(x, obs_log[:, 0], c='blue')
    axs[1].plot(x, obs_log[:, 1], c='green')
    axs[0].set_title(title, fontsize=18)
    plt.show()
    return
