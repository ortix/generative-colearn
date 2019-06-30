# pylint: skip-file
from julia import Julia
import os
import numpy as np


class Robot:
    def __init__(self, dof=3, mode="time", **kwargs):
        self.training_data_file = None
        self.training_data_dir = None
        self.mode = mode
        self.dof = dof
        t_bounds = np.tile([-np.pi / 2, np.pi / 2], (dof, 1))
        o_bounds = np.tile([-1, 1], (dof, 1))
        self.sampling_bounds = np.vstack([t_bounds, o_bounds])

        self.sim = self.init_julia()
        return None

    def init_julia(self):
        print("Waiting for julia to load...")
        Julia(compiled_modules=False)
        from julia import Pkg

        current_dir = os.path.dirname(os.path.abspath(__file__))
        Pkg.activate(os.path.join(current_dir, "n-dof"))
        print("activated current package")
        from julia import Main

        Main.eval('include("{}/n-dof/ColearnGenerate.jl")'.format(current_dir))
        print("Imported ColearnGenerate")

        if self.dof == 3:
            urdf = os.path.join(current_dir, "n-dof", "urdf", "kuka_iiwa", "model3DOF.urdf")
            return Main.ColearnGenerate.standardTimeOptimalSim(urdf, [2.0, 2.0, 2.0])
        else:
            print("WARNING: the URDF in simulation might not match the dynamics of the 2-dof manipulator used in training")
            urdf = os.path.join(current_dir, "n-dof", "urdf", "2dof.urdf")
            return Main.ColearnGenerate.standardTimeOptimalSim(urdf, [1.0, 1.0])



    def set_training_data_dir(self, path):
        """ 
        Absolute path to the directory where training data is stored
        """
        self.training_data_dir = path
        if not os.path.exists(path):
            os.makedirs(path)
        return None

    def simulate(self, samples=1000, dt=0.01):
        exit("Data generation is not supported yet. Please generate separately.")
        return

    def simulate_steer_full(self, s0, u, dt=0.01):

        # For now we don't use dt.. maybe we will incorporate it later
        # The julia simulation takes an array containing the current state
        # and the initial costates. In the legacy code the u variable also
        # contains cost and t_f, we remove that.

        final_time = np.float64(u[-1])
        time_trajectory, state_trajectory = self.sim(np.hstack([s0, u[:-2]]), final_time)
        return state_trajectory[-1], state_trajectory

    def simulate_steer(self, s0, u, dt=0.01):
        final_state, trajectory = self.simulate_steer_full(s0, u)
        theta = final_state[: self.dof]
        omega = final_state[self.dof:(2*self.dof)]
        theta_traj = [state[:self.dof] for state in trajectory]
        omega_traj = [state[self.dof:2*self.dof] for state in trajectory]
        return np.hstack([theta, omega]), np.hstack([theta_traj, omega_traj])

    def get_eom(self):
        pass

    def get_alpha_sampler(self):
        pass

    def validate_costates(self, s0, u):
        pass

    def validate(self, s0, s1, u, dt=0.01):
        theta_hat, omega_hat = self.simulate_steer(s0, u)
        n = int(s0.shape[0] / 2)
        return np.abs(s1[:n] - theta_hat), np.abs(s1[n:] - omega_hat)

    def save(self, path):
        exit("Generation not supported")
        return None

    def load(self, path):
        return None


if __name__ == "__main__":
    r = Robot(3, "time")
    time, states = r.simulate_steer_full(np.array([0.1]*6), np.array([0.3]*8))
    print(np.stack(states, axis=0))
