
function partialStatesZipper(p1, p2, factor)
    return [n[1] + factor * n[2] for n in zip(p1, p2)]
end

const RK4TABLEAU =  zip([0.0, 0.5, 0.5, 1.0], [1.0, 2.0, 2.0, 1.0] / 6.0)

function RK4specialStep(eom, dt, listOfPartialStates)
    ns = copy(listOfPartialStates)
    k = copy(listOfPartialStates) # multiplied by zero in the first round
    for (f_step, f_total) in  RK4TABLEAU
        ps = partialStatesZipper(listOfPartialStates, k, f_step * dt)  
        k = eom(ps...)
        ns = partialStatesZipper(ns, k, f_total * dt) 
    end
    return ns
end

function simulate(eom, theta0, omega0, lambda0, mu0, dt, steps)

    theta, omega, lambda, mu = [repeat(transpose(x), steps) for x in [theta0, omega0, lambda0, mu0]]
    cost = zeros(eltype(theta), steps)
    simulate!(eom, theta, omega, lambda, mu, cost, dt, steps)
    
    return [theta[end,:], omega[end,:], lambda[end,:], mu[end,:], cost[end]], [theta, omega, lambda, mu, cost]
end

function simulate!(eom, theta, omega, lambda, mu, cost, dt, steps)
    result = [theta[1,:], omega[1,:], lambda[1,:], mu[1,:], cost[1]]
    for i in 1:steps
        result =  RK4specialStep(eom, dt, result)
        theta[i,:] = result[1]
        omega[i,:] = result[2]
        lambda[i,:] = result[3]
        mu[i,:] = result[4]
        cost[i] = result[5]
    end
end





