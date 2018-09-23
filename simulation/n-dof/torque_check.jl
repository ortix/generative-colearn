using CSV: read
include("dynamics.jl")
using Dynamics: sim
using ProgressMeter

allSwitch(mu0, mu1) = all(sign.(mu0) + sign.(mu1) == zeros(mu0))
singleSwitch(mu0, mu1) = any(x -> x == 0, sign.(mu1) + sign.(mu0))
df = read("2dof_neg.csv")
data = convert(Array, df)
n = size(data)[1]
k = 1000
@printf("Dataset has size of %i \n", n)

dt = 0.01
switchAll = 0
switchSingle = 0
idx_arr = randperm(k)[1:k]
println("Running $k random simulations")
@showprogress 1 "Computing ..."  for j = 1:k
    i = idx_arr[j]
    steps = trunc(Int64, data[i,end] / dt)
    m0 = data[i, 7:8]
    t, o, l, m, cost = sim(data[i, 1:2], data[i, 3:4], data[i, 5:6], data[i, 7:8], dt, steps)
    if allSwitch(m0, m)
        switchAll += 1
    elseif singleSwitch(m0, m)
        switchSingle += 1
    end
end

@printf("Switching Factor all: %f \n", (switchAll / k))
@printf("Switching Factor single: %f \n", (switchSingle / k))
