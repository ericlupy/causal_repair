import numpy as np
import time
import argparse
import os
from utils import simulate


# interpolate between factual and counterfactual to find a midpoint that gives actual cause
def interpolate(factual: np.array, counterfactual: np.array, step_ctrl=0.1):

    start = time.time()
    num_simulator_runs = 0
    num_interpolation_steps = 0
    total_simulator_time = 0
    total_interpolation_time = 0

    assert factual.shape == counterfactual.shape

    solution = np.copy(counterfactual)

    for x1 in range(factual.shape[0]):
        for x2 in range(factual.shape[1]):
            # no need to interpolate
            if solution[x1][x2] == factual[x1][x2]:
                continue
            else:
                diff = solution[x1][x2] - factual[x1][x2]
                temp_solution = np.copy(solution)
                while diff > step_ctrl / 2:  # > 0, but we add step_ctrl / 2 to avoid inprecise data

                    # interpolation step
                    start_interpolation = time.time()
                    temp_solution[x1][x2] = temp_solution[x1][x2] - step_ctrl
                    diff = diff - step_ctrl
                    num_interpolation_steps += 1
                    total_interpolation_time += (time.time() - start_interpolation)

                    # simulator run
                    start_simulation = time.time()
                    outcome, _ = simulate(temp_solution, render=False)
                    num_simulator_runs += 1
                    total_simulator_time += (time.time() - start_simulation)

                    # check
                    if outcome:  # property satisfied
                        solution = temp_solution
                    else:  # property violated
                        break
                while diff < - step_ctrl / 2:

                    # interpolation step
                    start_interpolation = time.time()
                    temp_solution[x1][x2] = temp_solution[x1][x2] + step_ctrl
                    diff = diff + step_ctrl
                    num_interpolation_steps += 1
                    total_interpolation_time += (time.time() - start_interpolation)

                    # simulator run
                    start_simulation = time.time()
                    outcome, _ = simulate(temp_solution, render=False)
                    num_simulator_runs += 1
                    total_simulator_time += (time.time() - start_simulation)

                    # check
                    if outcome:  # property satisfied
                        solution = temp_solution
                    else:  # property violated
                        break

    total_time = time.time() - start
    return solution, total_time, total_simulator_time, total_interpolation_time, num_simulator_runs, num_interpolation_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_pos", help="init pos that the controller fails", default=-0.5)
    parser.add_argument("--init_vel", help="init vel that the controller fails", default=0.0)
    parser.add_argument("--approx_network_dir",
                        help="directory of approx networks (npy) files", default='approx_networks')
    parser.add_argument("--factual_ctrl_name", help="approx broken controller (npy) computed in hp_gridding", default='sig_8x16.npy')
    parser.add_argument("--counterfactual_ctrl_name", help="sampled good controller (npy) computed in hp_sample_controller",
                        default='sat_controller_1.npy')
    parser.add_argument("--solution_ctrl_name", help="final solution name", default="solution.npy")
    args = parser.parse_args()

    f_factual = np.load(os.path.join(args.approx_network_dir, args.factual_ctrl_name))
    f_counterfactual = np.load(os.path.join(args.approx_network_dir, args.counterfactual_ctrl_name))

    f_solution, total_time, total_simulator_time, total_interpolation_time, num_simulator_runs, num_interpolation_steps = interpolate(f_factual, f_counterfactual)
    np.save(os.path.join(args.approx_network_dir, args.solution_ctrl_name), f_solution)
    print('Total time = ' + str(total_time))
    print('Total simulator time = ' + str(total_simulator_time))
    print('Total interpolation time = ' + str(total_interpolation_time))
    print('Num simulator runs = ' + str(num_simulator_runs))
    print('Num interpolation steps = ' + str(num_interpolation_steps))