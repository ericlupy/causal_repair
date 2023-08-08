#!/bin/bash
pip install -r requirements.txt
python hp_gridding.py python hp_gridding.py --init_pos=-0.5 --init_vel=0.0 --network_dir=networks --network_name=sig_8x16.yml --approx_network_dir=approx_networks
# python hp_sample_controller.py --init_pos=-0.5 --init_vel=0.0 --approx_network_dir=approx_networks
python hp_interpolation.py --init_pos=0.5 --init_vel=0.0h --approx_network_dir=approx_networks --factual_ctrl_name=sig_8x16.npy --counterfactual_ctrl_name=sat_controller_1.npy --solution_ctrl_name=sol_controller_1.npy
