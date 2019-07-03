import RigidBodyDynamics.parse_urdf, RigidBodyDynamics.num_positions
include("ColearnGenerate.jl")
import RigidBodyDynamics
using DifferentialEquations

urdf = "urdf/kuka_iiwa/model3DOF.urdf"
torque_bound = [2.0, 2.0, 1.0] 
state_ub = [0.5pi, 0.35pi, -0.25pi, 1.0, 1.0, 1.0]
state_lb = [-0.5pi, 0.15pi, -0.5pi, -1.0, -1.0, -1.0]
time_ub = 1.0
time_lb = 0.01
numSamples = 100000

min_time_step = 0.001

println("Loaded initial parameters")

mechanism = parse_urdf(urdf, gravity=[0.0, 0.0, 0.0])
println("Parsed URDF")

mech = ColearnGenerate.ExtendedMechanism(mechanism, torque_bound)
println("Loaded mechanism model")

sampleSettings = ColearnGenerate.SampleSettings(state_ub, state_lb, time_ub, time_lb, true, 0.5)
println("Instantiated sample settings")

mcprob = ColearnGenerate.makeTimeOptimalMonteCarlo(mech, sampleSettings)
println("Generated monte carlo simulation")

# JIT-compile
sim = solve(mcprob, RK4(), dtmin=0.1, force_dtmin=true, num_monte=2)
println("JIT compiled simulation")

sim = solve(mcprob, RK4(), dtmin=min_time_step, force_dtmin=true, num_monte=numSamples)
println("Set up compiled simulation")

outFile = "kuka3dof_results_threads.csv"
ColearnGenerate.saveTimeOptimalMonteCarloResults(outFile, sim, mech.size)
println("Done with forward simulation!")


sampleSettings_back = ColearnGenerate.SampleSettings(state_ub, state_lb, time_ub, time_lb, false, 0.5)
println("Instantiated sample settings backwards")

mcprob_back = ColearnGenerate.makeTimeOptimalMonteCarlo(mech, sampleSettings_back)
println("Generated monte carlo simulation")

# JIT-compile
sim_back = solve(mcprob_back, RK4(), dtmin=0.1, force_dtmin=true, num_monte=2)
println("JIT compiled simulation")

sim_back = solve(mcprob_back, RK4(), dtmin=min_time_step, force_dtmin=true, num_monte=numSamples)
println("Set up compiled simulation")

outFile_back = "kuka3dof_results_threads_back.csv"
ColearnGenerate.saveTimeOptimalMonteCarloResults(outFile_back, sim_back, mech.size)
println("Done!")

# @time solve(mcprob, RK4(), dtmin=min_time_step, force_dtmin=true, num_monte=1000)
