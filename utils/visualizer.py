import glob
import json
import os
import time

import numpy as np
import requests

r = requests.get('http://127.0.0.1:7777/get_joint_angles')
model = json.loads(r.text)


def visualize(path):

    pattern = glob.glob(os.path.join(path, "path_*.txt"))

    # Load angles
    trajectory = np.loadtxt(pattern[-1])
    robot = True if trajectory.shape[1] > 2 else False
    # time.sleep(5)
    for i in range(trajectory.shape[0]):
        model["angles"][0] = -trajectory[i, 0] + 3*np.pi/2 if robot else trajectory[i, 0] + np.pi
        if robot:
            model["angles"][1] = trajectory[i, 1] + trajectory[i, 0]

        payload = json.dumps(model, separators=(',', ':'))
        r = requests.post('http://127.0.0.1:7777/set_joint_angles', payload,
                          headers={'Content-type': 'application/json'})
        # time.sleep(0.001)

if __name__ == "__main__":
    visualize(".")