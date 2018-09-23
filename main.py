import argparse
import gc
import os

# always import settings first
from settings import settings
from utils.logger import logger

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest="epochs", help="Amount of epochs. Default 1", type=int, default=1)
parser.add_argument("--experiment", dest="experiment", default="2dof",
                    help="Use: pendulum, 2dof")
parser.add_argument("--visualize", dest="visualize", action="store_true",
                    help="Visualize the path in urdf-viz")
parser.add_argument("--reach", dest="reachability", type=float,
                    help="Reachability value. -1 for GAN discriminator")
parser.add_argument("--runs", dest="runs", help="Amount of RRT runs", type=int)
parser.add_argument('--learner', dest="model", help="Which learning algorithm to use",
                    choices=["clsgan", "knn"], default="clsgan")
parser.add_argument('--folder', dest="folder", default="results",
                    help="Subfolder within tmp to store the results in. Usefull for separating experiments")
parser.add_argument('--post-process', dest="post_process", action="store_true",
                    help="Post process the results from folder. Run the experiments first!")
args = parser.parse_args()
cfg = settings(args.experiment)

# Map runtime settings
cfg.planner.runs = args.runs or cfg.planner.runs
cfg.model.use = args.model
cfg.planner.reachability = args.reachability or cfg.planner.reachability

# Post process the results and exit the main application
if args.post_process:
    from utils.post_process import process
    process(args.folder)
    exit()

# If we're not post processing, run experiments and move results to folder
from deep_rrt import DeepRRT
for exp_n in range(args.epochs):
    deepRRT = DeepRRT(exp_n+1)
    deepRRT.run()

    # Move experiment to a separate folder
    loc = os.path.join(cfg.paths.tmp, args.folder)
    run_path = deepRRT.move_run_to(loc)

    # Free up memory
    del deepRRT
    gc.collect()

# We can also visualize the data
if args.visualize:
    from utils.visualizer import visualize

    visualize(run_path)
