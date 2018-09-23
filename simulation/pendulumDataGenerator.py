import numpy as np
from tqdm import tqdm


def RK4step(eom, state, dt):
    k1 = eom(state)
    k2 = eom(state + 0.5 * dt * k1)
    k3 = eom(state + 0.5 * dt * k2)
    k4 = eom(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def RK4simulate(eom, state0, dt, tFinal, alpha=1.0):
    def simstep(s): return RK4step(eom, s, dt)
    t = 0
    steps = int(tFinal/dt)
    state = np.hstack([state0, [0]])
    state[2] = alpha*state[2]
    state[3] = alpha*state[3]
    state = np.tile(state, (steps+1,1))
    for i in range(steps):
        t += dt
        state[i+1, :] = simstep(state[i, :])
    return t, state[-1, :], state


def sampleFullState(computeAlpha):
    theta = np.random.uniform(-3/2*np.pi, 0.5*np.pi)
    omega = np.random.uniform(-np.pi, np.pi)
    # theta = np.random.uniform(-1, 1)
    # omega = np.random.uniform(-1, 1)
    t_final = np.round(np.random.uniform(0, 1.5), 2)
    labdaHat, muHat, alpha = sampleLambda(theta, omega, computeAlpha)
    return np.array([theta, omega, labdaHat, muHat]), alpha, t_final


def sampleLambda(theta, omega, computeAlpha):
    while True:
        try:
            labdaHat, muHat = unitCircleSample(2)
            alpha = computeAlpha(theta, omega, labdaHat, muHat)
            break
        except NegativeAlphaError:
            pass  # A negative alpha should not be in the database, so resample
    return labdaHat, muHat, alpha


def unitCircleSample(n):
    # Samples from n dimensional unit circle/sphere/hypersphere
    point = np.array([np.random.normal() for i in range(n)])
    point = point/np.linalg.norm(point)
    return point


class NegativeAlphaError(Exception):
    pass


def pendulumSwingUpEOM(fullState, u_max):
    theta, omega, labda, mu, cost = fullState
    thetaDot = omega
    omegaDot = np.sin(theta) - np.sign(mu) * u_max
    labdaDot = -mu * np.cos(theta)
    muDot = -labda
    costDot = 1.0
    return np.array([thetaDot, omegaDot, labdaDot, muDot, costDot])


def computeAlphaSwingUp(theta, omega, labdaHat, muHat, u_max):
    alpha = -1 / (labdaHat * omega + muHat *
                  np.sin(theta) - np.abs(muHat) * u_max)
    if alpha < 0:
        raise NegativeAlphaError
    return 1.0


def energyPendulumEOM(fullState, timeweight):
    theta, omega, labda, mu, cost = fullState
    thetaDot = omega
    omegaDot = np.sin(theta) - mu
    labdaDot = -mu*np.cos(theta)
    muDot = -labda
    costDot = timeweight + 1/2*mu**2
    return np.array([thetaDot, omegaDot, labdaDot, muDot, costDot])


def computeAlphaEnergy(theta, omega, labdaHat, muHat, timeweight):
    roots = np.roots([-1/2*muHat**2, labdaHat*omega +
                      muHat*np.sin(theta), timeweight])
    if not np.imag(roots[0]) == 0:
        raise ValueError
    alpha = np.amax(roots)
    if alpha < 0:
        raise ValueError
    return alpha


def boundedEnergyPendulumEOM(fullState, u_max, timeweight):
    theta, omega, labda, mu, cost = fullState
    if np.abs(mu) > u_max:
        return pendulumSwingUpEOM(fullState, u_max)
    else:
        return energyPendulumEOM(fullState, timeweight)


def computeAlphaBoundedEnergy(theta, omega, labdaHat, muHat, u_max, timeweight):
    alpha = computeAlphaEnergy(theta, omega, labdaHat, muHat, timeweight)
    if np.abs(alpha*muHat) > u_max:
        alpha = (-timeweight - 1/2*u_max**2)/(labdaHat * omega + muHat *
                                              np.sin(theta) - np.abs(muHat) * u_max)
        if alpha < 0:
            raise NegativeAlphaError
    return alpha


def runSamples(dt, numberOfSamples, eom, lambdaSampler):
    data = np.zeros((numberOfSamples, 8))
    for i in tqdm(range(numberOfSamples)):
        s0, alpha, tf = sampleFullState(lambdaSampler)
        tf, s1, _ = RK4simulate(eom, s0, dt, tf, alpha)
        data[i, :] = np.hstack([s0[0:2], s1[0:2], s0[2:], [tf, s1[-1]]])
    return data


def generateByMode(mode, samples, dt):
    u_max = np.pi/2 - 0.01  # Requires 1 swing to reach 0
    timeweight = 1.0
    if mode == "energy":
        data = runSamples(dt, samples,
                          lambda s: energyPendulumEOM(s, timeweight),
                          lambda a, b, c, d: computeAlphaEnergy(a, b, c, d, timeweight))
    if mode == "bounded":
        data = runSamples(dt, samples,
                          lambda s: boundedEnergyPendulumEOM(s, u_max, timeweight),
                          lambda th, om, lh, mh: computeAlphaBoundedEnergy(th, om, lh, mh,
                                                                           u_max, timeweight))

    if mode == "time":
        data = runSamples(dt, samples,
                          lambda s: pendulumSwingUpEOM(s, u_max),
                          lambda th, om, lh, mh: computeAlphaSwingUp(th, om, lh, mh, u_max))
        # Return the data and the labels
    return data[:, -4:], data[:, :4]


if __name__ == "__main__":
    dt = 0.01
    numberOfSamples = 200000
    u_max = np.pi/2 - 0.01  # Requires 1 swing to reach 0
    timeweight = 1.0

    # data = runSamples(dt, numberOfSamples,
    #                   lambda s: energyPendulumEOM(s, timeweight),
    #                   lambda a, b, c, d: computeAlphaEnergy(a, b, c, d, timeweight))
    # np.savetxt('dataEnergy.csv', data,  delimiter=',',
    #            header='theta0, omega0, theta1, omega1, lambda0, mu0, t1, cost')

    data = runSamples(dt, numberOfSamples,
                      lambda s: pendulumSwingUpEOM(s, u_max),
                      lambda th, om, lh, mh: computeAlphaSwingUp(th, om, lh, mh, u_max))
    np.savetxt('dataSwingUp.csv', data,  delimiter=',',
               header='theta0, omega0, theta1, omega1, lambda0, mu0, t_f , cost')

    # data = runSamples(dt, numberOfSamples,
    #                   lambda s: boundedEnergyPendulumEOM(s, u_max, timeweight),
    #                   lambda th, om, lh, mh: computeAlphaBoundedEnergy(th, om, lh, mh,
    #                                                                    u_max, timeweight))
    # np.savetxt('dataBoundedEnergy.csv', data, delimiter=',',
    #            header='theta0, omega0, theta1, omega1, lambda0, mu0, t1, cost')
