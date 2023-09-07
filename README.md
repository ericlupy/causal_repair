# Causal Repair of Learning-enabled Cyber-physical Systems

### Update 09/06/2023

1. Fixed typo in `sample_run.sh`

2. Fixed step size in `hp_gridding.py`, so that now the output numpy control heatmap is the same shape as the example sat controllers.

## Background

This is the code repo of "Causal Repair of Learning-enabled Cyber-physical Systems" by Pengyuan Lu, Ivan Ruchkin, Matthew Cleaveland, Oleg Sokolsky and Insup Lee,
published in ICAA 2023 [link](https://arxiv.org/abs/2304.02813).
The code contains an example repair of mountain car dynamics, including a to-be-repaired controller network and
an example sampled good controller network. Running the code requires Python 3.

### Instructions

After running ``pip install -r requirements.txt``, run the following Python files.

1. From a given to-be-repaired controller network, e.g. ``sig_8x16.yml``, we produce its gridded I/O map by

```
python hp_gridding.py --init_pos=<float in [-1.2, 0.6]> --init_vel=<float in [-0.07, 0.07]> --network_dir=<dir to actual networks> --network_name=<name of to-be-repaired network> --approx_network_dir=<dir to approx networks>
```

Notice that the to-be-repaired network must fail on the given initial pos and vel. One example identified is ``networks/sig_8x16.yml`` failing on (-0.5, 0.0).
This script will also display the heatmap of the I/O map and a trajectory at the end.

2. We uniformly sample alternative I/O maps that succeed in the task as follows.

```
python hp_sample_controller.py --init_pos=<previous init pos> --init_vel=<previous init vel> --approx_network_dir=<previous approx network dir>
```

Warning: this will uniformly sample the I/O space for 10000 times per CPU, on half of all your computer's available CPUs.
Also, this step contains randomness so there is no guarantee that a good controller I/O map can be sampled.
For convenience, we provide two already sampled networks ``sat_controller_1.npy`` and ``sat_controller_2.npy``. 
Please feel free to skip this step and use them in the next step.

3. Interpolate a solution that satisfies the Halpern-Pearl causality condition by interpolation.
This is run by calling the following code.

```
python hp_interpolation.py --init_pos=<previous init pos> --init_vel=<previous init vel> --approx_network_dir=<previous approx network dir> --factual_ctrl_name=<factual ctrl npy name from step 1> --counterfactual_ctrl_name=<counterfactual ctrl npy name from step 2> --solution_ctrl_name=<solution ctrl npy name to save the solution>
```

An alternative interpolation implementation is binary interpolation, which can be called by ``hp_interpolation_binary.py`` with the same arguments.

### Example Scripts

One example is to run the following bash script.

```
./sample_run.sh
```

Notice that this bash script skips the sampling part and directly uses our sampled good controller.
