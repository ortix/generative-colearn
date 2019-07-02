module ColearnGenerate

using RigidBodyDynamics
using ForwardDiff
using LinearAlgebra
using DifferentialEquations
using Random, Distributions
using DataFrames
using CSV

export ExtendedMechanism, makeTimeOptimalEom!, alphaTimeOptimal,
makeTimeOptimalMonteCarlo, SampleSettings, saveTimeOptimalMonteCarloResults,
StandardTimeOptimalSim

include("indirect_optimal_control_eom.jl")
include("simulation.jl")


function standardTimeOptimalSim(urdf, torque_bounds)
  mechanism = RigidBodyDynamics.parse_urdf(urdf, gravity=[0.0, 0.0, 0.0])
  mech = ExtendedMechanism(mechanism, torque_bounds)
  timeOptimalForMech = makeTimeOptimalEom!(mech)
  function simulate(fullState0, tFinal)
    tspan = (0.0, tFinal)
    prob = ODEProblem(timeOptimalForMech, fullState0, tspan)
    sol = solve(prob, RK4(), dtmin=0.001, force_dtmin=true);
    return (sol.t, sol.u)
  end
  return simulate
end

end # module
