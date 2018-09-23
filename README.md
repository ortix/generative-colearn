# Generative CoLearn
This repository contains the code for the Generative CoLearn algorithm. Generative CoLearn is a kinodynamic planning system leveraging the power of deep generative models to learn the cost and action space for a pendulum swingup task. This algorithm implements a conditional generative adversarial network with the least squares loss function.

Based on the following paper: https://arxiv.org/abs/1710.10122

### Authors
* Nick Tsutsunava (nicktsutsunava@gmail.com)
* Wouter Wolfslag (wouter.wolfslag@ed.ac.uk)
* Carlos Hernández Corbato (c.h.corbato@tudelft.nl)
* Mukunda Bharatheesha (m.bharatheesha@tudelft.nl)
* Martijn Wisse (m.wisse@tudelft.nl)

For inquiries regarding the code, please contact Nick Tsutsunava or open an Github issue.

## Getting started
1. If you have access to the repo, clone it: `git clone https://github.com/ortix/generative-colearn.git`
2. Ensure your current working directory is the location of this file: `cd generative-colearn`
3. Install the required pip packages: `pip install -r requirements.txt`
4. Download [Julia 0.6.4](https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.4-linux-x86_64.tar.gz) and make sure the extracted `bin/` is in your `PATH`.
5. Download the generated data set for the planar arm [here](https://drive.google.com/open?id=1rzXIwWz_cNUrBcqqu6b71KdswLah5bXD) and place it in the `data` folder. Make sure it's named `2dof_time.csv`.
6. Run `python main.py`

There are several run time arguments that override the JSON settings. Run `python main.py --help` to see what they are.

### Planar arm data generation
The data generation algorithm for the planar arm is written in Julia. Therefore, generating the data set is not as trivial. We have provided the data that we used for our experiments [here](https://drive.google.com/open?id=1rzXIwWz_cNUrBcqqu6b71KdswLah5bXD).

The fastest way to generate the data is running the generation in parallel terminals as we did not write the script with parallelization in mind.

1. Navigate to `simulation/n-dof/`
2. In 6 different terminals run the following commands:
    * `julia generate.jl full`
    * `julia generate.jl full_backwards`
    * `julia generate.jl start`
    * `julia generate.jl start_backwards`
    * `julia generate.jl goal`
    * `julia generate.jl goal_backwards`
3. Move the generated files to `data/merge`
4. Navigate to `data/` and run `2dof_data_merge.py`
5. Make sure the file is named `2dof_time.csv`


## Docker
A docker image is available: `docker pull ortix/generative-colearn`. 

Alternatively, you can build your own image. The `Dockerfile` copies the Julia binaries into the build process so they are required.
1. Download [Julia 0.6.4](https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.4-linux-x86_64.tar.gz).
2. Extract with `tar -xvzf julia-0.6.4-linux-x86_64.tar.gz`
3. Rename the extracted folder to `julia` and make sure it is within the root of the cloned repository
4. Build to image `docker build .`

## Flags
There are some flags to quickly parameterize the experiments. These flags **override** the settings inside the `.json` files.

### Experiment
The `--experiment` flag allows you to define either `pendulum` or `2dof` experiment. 

Usage: `python main.py --experiment=2dof`

Default: `2dof`

### Runs
It is easy to specify how many RRT runs to run with the `--runs` flag.

Usage: `python main.py --runs=100`

Default: `10`

### Reachability
The pendulum experiments only used the reachability parameter whereas for the planar arm we used the discriminator to classify reachable trajectories. You can change the reachable bound for both experiments with `--reach`. The value `-1` corresponds to the discriminator.

Usage: `python main.py --reach=0.1` or `python main.py --reach=-1`

Default: `0.3` for `pendulum` and `-1` for `2dof`
>Note: Reachability is extremely slow for the planar arm.

### Folder
It is useful to store experimental results in separate folder. For example, when running both the planar arm and pendulum experiments, it is a good idea to separate them for the post processing script. The folders are stored within the `tmp` directory. It is not possible to pass nested directories.

Usage: `python main.py --folder=pendulum_results`

Default: `results`

### Post Processing
It is easy to summarize the experimental results with the `--post-process` flag. The post processing script looks in the for the results within `--folder` directory. If not `--folder` flag is passed the default `results` value will be used.

Usage: `python main.py --post-process`

### Learner
Generative CoLearn uses two learning algorithms: `knn` and `clsgan`. The `--learner` flag allows you to select which algorithm to use.

Usage: `python main.py --learner=clsgan`

Default: `clsgan`

### Visualization
We provide the ability to visualize the generated path with `urdf-viz`. 
1. Run `./urdf-viz simulation/n-dof/urdf/2dof.urdf`. The second link will automatically be frozen for pendulum tasks.
2. In a different terminal, run the experiment `python main.py --visualize`. Optionally provide the `--experiment` flag.

## Settings
The settings are contained in a `.json` file, which parameterize the components of this application. There is a `settings/base_settings.json` file containing all available settings. Custom settings specific to an experiment are stored in corresponding files. For example if you run `python main.py --experiment=pendulum` the settings in `pendulum_settings.json` will be loaded and merged with `base_settings.json`. 

### Settings documentation
#### `paths`
The locations of directories for storing training data for the neural network, the trained network itself, temporary data and figures for analysis and post processing. These paths are all relative to the location of `main.py`.
```json
"paths": {
    "training": "data/",
    "models": "models/trained",
    "tmp": "tmp/",
    "figs": "analysis/figures"
}
```
#### `planner`
The planner we use is RRT.
* `debug`: Enable plotting and verbose text at run time
* `plotting`: Save figures of successfully planned paths
* `goal_noise_var`: The variance of the Gaussian noise when goal state is selected
* `threshold`: Euclidean distance to goal for convergence
* `reachability`: See paper. `false` enables the discriminator as a trajectory classifier
```json
"planner": {
    "debug": false,
    "plotting": true,
    "runs": 10,
    "goal_bias": 15,
    "goal_noise_var": 1.571,
    "threshold": 0.15,
    "reachability": 0.3,
    "max_nodes": 300
}
```


#### `simulation`
Settings for the simulation.
* `u_max`: Maximum torque. Only for pendulum
* `load`: Whether to load the generated data or to generate during runtime.
* `split`: What fraction of data to use as a test set

>Note: Data is loaded according to `<system>_<mode>.csv`
```json
"simulation": {
    "system": "2dof",
    "dof": 2,
    "u_max": 0.5,
    "mode": "time",
    "samples": 40000,
    "load": true,
    "split": 0.2
},
```
#### `model`
Settings for the learning model.

* `save`: Whether to save the trained network
* `load`: Whether to load to load an existing trained network
* `use`: What model to use. Either `clsgan` or `knn`.
* `<model>.structure|training`: Model specific settings. See `settings/base_settings.json`
```json
"model": {
    "save": true,
    "load": false,
    "use": "clsgan",
    "clean": false,
    "knn": {"structure":{},"training":{}},
    "clsgan": {"structure":{},"training":{}}
}
```
