from julia.api import Julia
import os
import numpy as np

jl = Julia(compiled_modules=False)
from julia import Pkg

Pkg.activate("..")
print("activated current package")
print(Pkg.status())
from julia import Main


dirname = os.path.dirname(os.path.abspath(__file__))
colearn = os.path.join(dirname, "ColearnGenerate.jl")
Main.eval('include("{}")'.format(colearn))
print("Imported ColearnGenerate")


class SimulateTimeOptimal:
    def __init__(self, urdf, torque_bounds):
        self.simulate = Main.ColearnGenerate.standardTimeOptimalSim(urdf, torque_bounds)

    def __call__(self, fullState, finalT):
        return self.simulate(fullState, finalT)


if __name__ == "__main__":

    dirname = os.path.dirname(os.path.abspath(__file__))
    kuka = os.path.join(dirname, "urdf", "kuka_iiwa", "model3DOF.urdf")

    sim = SimulateTimeOptimal(kuka, [2.0, 2.0, 2.0])
    sim([0.0] * 12, 0.95)
    for idx in range(10):
        t, q = sim([0.0] * 12, 0.95)
        print(q[-1])
        print(q)
        print(np.linalg.norm(q[-1][6:]))
        print(np.linalg.norm(q[-1][:6]))


