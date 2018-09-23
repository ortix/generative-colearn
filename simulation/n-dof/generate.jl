using ForwardDiff
using Distributions
using ProgressMeter
push!(LOAD_PATH, joinpath(dirname(@__FILE__), "eom"))
using RobotEOM
include("simulate.jl")


function mechanismEom(theta, omega, torque, mechanism::Mechanisms)
    M, bias = mechanismDynamicsTerms(theta, omega, mechanism)
    return M \ (torque - bias)
end


function timeOptimalEom(theta, omega, mu, mechanism::Mechanisms)
    M, bias = mechanismDynamicsTerms(theta, omega, mechanism)
    torque = -sign.(M \ mu) .* mechanism.max_torques
    return M \ (torque - bias)
end

function timeOptimalHamiltonian(theta::AbstractVector{T1},
                                omega::AbstractVector{T2}, lambda,
                                mu, mechanism::Mechanisms) where {T1,T2}
    if !(T2 == Float64)
        result = one(T2)
    else
        result = one(T1)
    end
    alpha = timeOptimalEom(theta, omega, mu, mechanism)
    result += dot(alpha, mu)
    result += dot(lambda, omega)
    return result
end



function timeOptimalFullStateSampling(mechanism::Mechanisms,
                                      switchFactor=nothing, loc=nothing)
    n = robotDOF(mechanism)
    t_steps = rand(1:Int(t_max / dt))
    if loc == nothing
        theta = rand(Uniform(-0.5 * pi, 0.5 * pi), n)
        omega = rand(Uniform(-1, 1), n)
    end
    if loc == "goal"
        theta = [rand(Normal(0.25 * pi, 0.1)),rand(Normal(0, 0.1))]
        omega = rand(Normal(0, 0.1), n)
    end
    if loc == "start"
        theta = [rand(Normal(-0.25 * pi, 0.1)),rand(Normal(0, 0.1))]
        omega = rand(Normal(0, 0.1), n)
    end
    while true
        lambdaHat = rand(Normal(), n)
        if typeof(switchFactor) == Float64
            muBounds = lambdaHat * t_steps * dt
            bounds = [sort([-abs(switchFactor) * bnd, (1 + abs(switchFactor)) * bnd]) for bnd in muBounds]
            if sign(switchFactor) > 0
                muHat = [rand(Truncated(Normal(), bnd[1], bnd[2])) for bnd in bounds]
            else
                muHat = [rand(Truncated(Normal(), -bnd[2], -bnd[1])) for bnd in bounds]
            end
        else
            muHat = rand(Normal(), n)
        end
        coLength = norm(vcat(lambdaHat, muHat))
        lambdaHat = lambdaHat / coLength
        muHat = muHat / coLength
        H1 = timeOptimalHamiltonian(theta, omega, lambdaHat, muHat, mechanism)
        dH = dHda(theta, omega, lambdaHat, muHat, mechanism)
        alpha = -H1 / dH + 1.0
        if alpha >= 0.0
            return t_steps, theta, omega, lambdaHat, muHat, alpha
        end
    end
end

dHdtheta(theta, omega, lambda, mu, mechanism) = ForwardDiff.gradient(t -> timeOptimalHamiltonian(t, omega, lambda, mu, mechanism),
                                                    theta)


function dHdomega(theta, omega, lambda, mu, mechanism)
    return ForwardDiff.gradient(o -> timeOptimalHamiltonian(theta, o, lambda, mu, mechanism),
         omega)
end

function timeOptimalFullStateEOM(theta, omega, lambda, mu, cost, mechanism)
    thetaDot = omega
    omegaDot = timeOptimalEom(theta, omega, mu, mechanism)
    lambdaDot = -dHdtheta(theta, omega, lambda, mu, mechanism) 
  #= dHdtheta(out, theta, omega, lambda, mu, mechanism) =# 
    muDot = -dHdomega(theta, omega, lambda, mu, mechanism)
    costDot = 1.0
    return thetaDot, omegaDot, lambdaDot, muDot, costDot
end


dHda(theta, omega, lambda, mu, mechanism) = ForwardDiff.derivative(a -> timeOptimalHamiltonian(theta, omega, a * lambda,
                                 a * mu, mechanism),
     1.0)

    

function generateDataSet(simulationFunction, samplingFunction, fileName,
                         numberOfSteps, mechanism, switchFactor=nothing, loc=nothing)
    open(fileName, "w") do f
    for name in ["theta0","omega0","lambda0", "mu0","theta1","omega1"]
        for c in 1:robotDOF(mechanism)
            write(f, "$(name)$(c), ")
        end
    end
    write(f, "cost, ")
    write(f, "t_f\n")
    checkSwitch = 0
    @showprogress 1 "Computing ..." for _ = 1:numberOfSteps
        t_steps, theta0, omega0, lambda0, mu0, alpha = samplingFunction(mechanism,
                                                             switchFactor, loc)
        final_state, trajectory = simulationFunction(theta0, omega0, lambda0, mu0,
                                                    sign(switchFactor) * dt, t_steps)
        theta, omega, lambda, mu, cost = final_state                                            
        final_time = t_steps * dt
        if typeof(switchFactor) == Float64 && sign(switchFactor) < 0
            costateNorms = sqrt.(sum(lambda.^2 + mu.^2)) 
            savedata = [theta, omega, lambda / costateNorms, mu / costateNorms, theta0, omega0]
        else 
            savedata = [theta0, omega0, lambda0, mu0, theta, omega]
        end
        for name in savedata
            for c in 1:robotDOF(mechanism)
                write(f, "$(name[c]), ")
            end
        end
        write(f, "$(abs(cost[1])), ")
        write(f, "$(final_time)\n")
    end
  end
end



const dt = 0.01
const t_max = 1.
const switchFactor = 0.5
const numberOfDimensions = 2 
const mechanism = NLinkPlanarMechanism(numberOfDimensions) 
#= const mechanism = Pendulum() =#
# const mechanism = UrdfMechanism(joinpath(dirname(@__FILE__), "urdf/2dof.urdf"), ones(Float64, 2))
                                
out = zeros(Float64, robotDOF(mechanism))
const eom(t, o, l, m, c) =  timeOptimalFullStateEOM(t, o, l, m, c, mechanism)
const simulateTimeOptimal(t, o, l, m, dt, st) = simulate(eom, t, o, l, m, dt, st)
theta, omega, lambda, mu, alpha = timeOptimalFullStateSampling(mechanism)

println("Starting data generation...")

# No time to implement arg parser
if any(x -> x == "full", ARGS)
    generateDataSet(simulateTimeOptimal, timeOptimalFullStateSampling, "2dof_pos_500k.csv", 500000, mechanism, switchFactor)
end
if any(x -> x == "full_backwards", ARGS)
    generateDataSet(simulateTimeOptimal, timeOptimalFullStateSampling, "2dof_neg_500k.csv", 500000, mechanism, -switchFactor)
end
if any(x -> x == "start", ARGS)
    generateDataSet(simulateTimeOptimal, timeOptimalFullStateSampling, "2dof_pos_50k_start.csv", 50000, mechanism, switchFactor, "start")
end
if any(x -> x == "start_backwards", ARGS)
    generateDataSet(simulateTimeOptimal, timeOptimalFullStateSampling, "2dof_neg_50k_start.csv", 50000, mechanism, -switchFactor, "start")
end
if any(x -> x == "goal", ARGS)
    generateDataSet(simulateTimeOptimal, timeOptimalFullStateSampling, "2dof_pos_50k_goal.csv", 50000, mechanism, switchFactor, "goal")
end
if any(x -> x == "goal_backwards", ARGS)
    generateDataSet(simulateTimeOptimal, timeOptimalFullStateSampling, "2dof_neg_50k_goal.csv", 50000, mechanism, -switchFactor, "goal")
end




