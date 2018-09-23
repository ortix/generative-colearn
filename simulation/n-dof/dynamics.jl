# __precompile__()
module Dynamics
    include(joinpath(dirname(@__FILE__), "simulate.jl"))
    push!(LOAD_PATH, joinpath(dirname(@__FILE__), "eom/"))
    using RobotEOM
    using ForwardDiff

    function mechanismEom(theta, omega, torque, mechanism::Mechanisms)
        M, bias = mechanismDynamicsTerms(theta, omega, mechanism)
        return M \ (torque - bias)
    end

    function dHdtheta(theta, omega, lambda, mu, mechanism) 
        return ForwardDiff.gradient(t -> timeOptimalHamiltonian(t, omega, lambda, mu, mechanism), theta)
    end

    function dHdomega(theta, omega, lambda, mu, mechanism)
        return ForwardDiff.gradient(o -> timeOptimalHamiltonian(theta, o, lambda, mu, mechanism), omega)
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

    function timeOptimalFullStateEOM(theta, omega, lambda, mu, cost, mechanism)

        thetaDot = omega
        omegaDot = timeOptimalEom(theta, omega, mu, mechanism)
        lambdaDot = -dHdtheta(theta, omega, lambda, mu, mechanism) 
        muDot = -dHdomega(theta, omega, lambda, mu, mechanism)
        costDot = 1.0
        return thetaDot, omegaDot, lambdaDot, muDot, costDot
    end

    const dt = 0.01
    const t_max = 1.
    # const mechanism = UrdfMechanism(joinpath(dirname(@__FILE__), "urdf/2dof.urdf"), ones(Float64, 2))
    const mechanism = NLinkPlanarMechanism(2) 

    const eom(t, o, l, m, c) =  timeOptimalFullStateEOM(t, o, l, m, c, mechanism)
    const sim(t, o, l, m, dt, st) = simulate(eom, t, o, l, m, dt, st)
    function test()
        return 1
    end
    
    export eom,sim,test

end

# using .Dynamics

# a = sim([2.990469062, 0.802295661, -3.119177084],  # theta
#         [-2.768987789,	-2.668992656,	0.320263734],  # omega
#         [-0.728761355,	0.01039203,	0.149286399],  # lambda
#         [-0.615887651,	0.020664921,	0.258394711],  # mu
#         0.01,  # dt
#         10)  # steps



