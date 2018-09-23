'''
author: Wouter Wolfslag
'''
import numpy as np


def RK4step(eom, state, dt):
    k1 = eom(state)
    k2 = eom(state + 0.5 * dt * k1)
    k3 = eom(state + 0.5 * dt * k2)
    k4 = eom(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def RK4simulate(eom, state0, dt, tFinal):
    def simstep(s):
        return RK4step(eom, s, dt)

    t = 0
    state = state0
    state_vec = []
    while t < tFinal:
        t += dt
        state = simstep(state)
        state_vec.append(state)

    return t, state


def pendulumSwingUpEOM(fullState, u_max):
    '''
    Theta: angle of pendulum
    Omega: angular velocity of pendulum
    Labda: costate of the position
    Mu: costate of the angular velocity
    u_max: constant pi / 2 - 0.1
    '''
    theta, omega, labda, mu = fullState
    thetaDot = omega
    omegaDot = np.sin(theta) - np.sign(mu) * u_max
    labdaDot = -mu * np.cos(theta)
    muDot = -labda
    return np.array([thetaDot, omegaDot, labdaDot, muDot])


def sampleFullState(u_max):
    theta = np.random.uniform(-np.pi, np.pi)
    omega = np.random.uniform(-3, 3)
    t_final = np.round(np.random.uniform(0, 1.0), 2)

    while True:
        # Generate values for labda and mu and normalize between -1 and 1
        labdaHat = np.random.normal()
        muHat = np.random.normal()
        hatNorm = np.sqrt(labdaHat**2 + muHat**2)
        labdaHat = labdaHat / hatNorm
        muHat = muHat / hatNorm

        # Checks if there is a positive factor alpha, such that costate = alpha*costateHat leads to Hamiltonian = 0
        alpha = -1 / (
            labdaHat * omega + muHat * np.sin(theta) - np.abs(muHat) * u_max)
        if alpha > 0:
            return np.array([theta, omega, labdaHat, muHat]), t_final


def generator(samples, dt):
    from tqdm import tqdm

    u_max = np.pi / 2 - 0.1  # Requires 1 swing to reach 0
    data = np.zeros((samples, 4))
    labels = np.zeros((samples, 4))
    for i in tqdm(range(samples)):
        s0, tf = sampleFullState(u_max)
        tf, s1 = RK4simulate(lambda s: pendulumSwingUpEOM(s, u_max), s0, dt,
                             tf)
        labels[i, :] = np.hstack([s0[0:2], s1[0:2]])
        data[i, :] = np.hstack([s0[2:], [tf, tf]])
    return data, labels


if __name__ == "__main__":
    data, labels = generator(10, 0.01)
    data_labels = np.hstack([data, labels])
    np.savetxt(
        'data.csv',
        data_labels,
        delimiter=',',
        header='lambda0, mu0, t1, theta0, omega0, theta1, omega1')
