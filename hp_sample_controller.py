#!/usr/bin/python3


"""
A script to sample a controller that gives property SAT (on a given init state)

1 command line arguments:
    1: (mandatory) the directory where to save the logs

"""

import numpy as np
import os
import multiprocessing
import argparse

# my model
from continuous_mountain_car import Continuous_MountainCarEnv


min_pos = -1.2
step_pos = 0.02
min_vel = -0.07
step_vel = 0.002


# model is a 70 * 90 random matrix on [-0.9, -0.7, ..., 0.9]
def predict(model, inputs):
    pos, vel = inputs

    j = int((pos - min_pos) // step_pos)
    if j == 90:
        j = 89
    i = int((vel - min_vel) // step_vel)
    if i == 70:
        i = 69

    return model[i][j]


# parallel uniform sampling function
def sample_controller(inputs):

    worker_id, init_pos, init_vel, save_dir = inputs[0], inputs[1], inputs[2], inputs[3]

    for i in range(10000):
        if any(File.startswith("sat_controller_") for File in os.listdir(save_dir)):
            print("Worker " + str(worker_id) + " controller found, exit")
            break
        print("Worker " + str(worker_id) + " sampling controller " + str(i))

        # sample a controller model
        # 70 * 90 matrix in [-0.9, -0.7, ..., 0.9]
        model = np.random.randint(10, size=[70, 90]) * 0.2 - 0.9

        # create gym environment
        env = Continuous_MountainCarEnv(init_pos=init_pos, init_vel=init_vel)
        episodes = 110

        # reset the environment to a random new position
        observation = env.reset()
        total_reward = 0
        num_episodes = 0

        for e in range(episodes):
            action = predict(model, observation)
            observation, reward, done, info = env.step([action])
            total_reward += reward

            if done:
                num_episodes = e
                break

        # decide success or failure
        if num_episodes < episodes - 1:
            print("Worker " + str(worker_id) + 'success!')
            np.save(os.path.join(save_dir, f'sat_controller_{worker_id}.npy'), model)
            break
        else:
            print("Worker " + str(worker_id) + 'Failure!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_pos", help="init pos that the controller fails", default=-0.5)
    parser.add_argument("--init_vel", help="init vel that the controller fails", default=0.0)
    parser.add_argument("--approx_network_dir",
                        help="directory of approx networks (npy) files", default='approx_networks')
    args = parser.parse_args()

    num_parallel = multiprocessing.cpu_count() // 2
    parallel_inputs = [[worker_id, float(args.init_pos), float(args.init_vel), args.approx_network_dir] for worker_id in range(num_parallel)]

    print("Starting parallel sampling ...")
    with multiprocessing.Pool(processes=num_parallel) as pool:
        pool.map(sample_controller, parallel_inputs)
