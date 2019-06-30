# pylint: skip-file
from julia import Julia
import os
from sklearn.metrics import mean_squared_error
import numpy as np


class Robot():
    def __init__(self, dof=3, mode="time", **kwargs):
        self.training_data_file = None
        self.training_data_dir = None
        self.mode = mode
        self.dof = dof
        t_bounds = np.tile([-np.pi/2, np.pi/2], (dof, 1))
        o_bounds = np.tile([-1, 1], (dof, 1))
        self.sampling_bounds = np.vstack([t_bounds, o_bounds])

        self.j = self.init_julia()
        return None

    def init_julia(self):
        print("Waiting for Julia to compile...")
        Julia(compiled_modules=False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        from julia import Main
        Main.eval('include("{}/n-dof/dynamics.jl")'.format(current_dir))
        print("Loaded Dynamics Module")
        return Main.Dynamics

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
        """
        Returns theta, omega, lambda, mu, cost
        """
        n = int(s0.shape[0]/2)

        # Calculate steps necessary for RK4 integration.
        # This is the total integration time divided by dt
        # Integration time (t_f) is the last entry in u
        steps = int(u[-1]/dt) or 1
        return self.j.sim(s0[:n], s0[n:], u[:n], u[n:-2], dt, steps)

    def simulate_steer(self, s0, u, dt=0.01):
        final_state, trajectory = self.simulate_steer_full(s0, u)
        theta, omega, _, _, _ = final_state
        theta_traj, omega_traj, _, _, _ = trajectory
        return np.hstack([theta, omega]), np.hstack([theta_traj, omega_traj])

    def get_eom(self):
        pass

    def get_alpha_sampler(self):
        pass

    def validate_costates(self, s0, u):
        pass

    def validate(self, s0, s1, u, dt=0.01):
        theta_hat, omega_hat = self.simulate_steer(s0, u)
        n = int(s0.shape[0]/2)
        return np.abs(s1[:n] - theta_hat), np.abs(s1[n:] - omega_hat)

    def save(self, path):
        exit("Generation not supported")
        return None

    def load(self, path):
        return None


if __name__ == "__main__":
    from timeit import default_timer as timer
    r = Robot("time")

    def test():
        return j.sim(
            [2.990469062, 0.802295661],  # theta
            [-2.768987789,	-2.668992656],  # omega
            [-0.728761355,	0.01039203],  # lambda
            [-0.615887651,	0.020664921],  # mu
            0.01,  # dt
            10)  # steps

    start = timer()
    a = test()
    print(timer() - start)

    for _ in range(100):
        start = timer()
        a = test()
        print(timer()-start)
