struct SampleSettings
  state_ub
  state_lb
  time_ub
  time_lb
  forward::Bool
  epsilon
end


function BoolToPM(x::Bool)
  if x
    return 1.0
  else
    return -1.0
  end
end

function sampleCostateForSwitching(m::ExtendedMechanism, s::SampleSettings, tFinal)
  forwardSign = BoolToPM(s.forward)
  lambdaDist = MvNormal(zeros(m.size), ones(m.size))
  lambdaSample = rand(lambdaDist)
  muNominal = forwardSign*tFinal*lambdaSample
  muBounds = [sort([-s.epsilon*mn,(1.0 + s.epsilon)*mn]) for mn in muNominal]
  muSample = [rand(TruncatedNormal(0,1, mb[1], mb[2])) for mb in muBounds]
  costate =  vcat(lambdaSample, muSample)
  costate[:] /= norm(costate)
  return costate
end

function makeTimeOptimalEom(mech, settings) 
  function timeOptimal_for_mech(ds,s, p, t)
      ds[:] = optimalEOM(gradTimeOptimalHamiltonian, mech, s, settings.forward)
  end
end

function makeTimeOptimalEom!(mech, settings) 
  function timeOptimal_for_mech(ds,s, p, t)
      optimalEOM!(ds, gradTimeOptimalHamiltonian!, mech, s, settings.forward)
      return ds
  end
end

function makeTimeOptimalEom!(mech) 
  function timeOptimal_for_mech(ds,s, p, t)
      optimalEOM!(ds, gradTimeOptimalHamiltonian!, mech, s, true)
      return ds
  end
end

function makeTimeOptimalSampler(m::ExtendedMechanism, s::SampleSettings)
  function sample_random_initial_state(t_final) # global mech, state_ub, state_lb
      state = rand(2*m.size).*(s.state_ub-s.state_lb) .+ s.state_lb
      while true
        cs = sampleCostateForSwitching(m, s, t_final)
        cs[:] /= norm(cs)
        fs = vcat(state,cs)
        if alphaTimeOptimal(m, fs) > 0.0
            return fs
        end
      end
  end
end

function makeTimeOptimalMonteCarlo(mech,settings)
  eom = makeTimeOptimalEom!(mech, settings)
  sampler = makeTimeOptimalSampler(mech, settings)

  function prob_func(p, i , repeat)
      t_f = rand()*(settings.time_ub-settings.time_lb) + settings.time_lb
      u0 = sampler(t_f)
      tspan = (0.0, t_f)
      return ODEProblem(eom,u0,tspan)
  end

  function output_func(sol,i)
      rerun = false
      if settings.forward
        out = vcat(sol.u[1], sol.u[end], sol.t[end], sol.t[end])
      else
        out = vcat(sol.u[end], sol.u[1], sol.t[end], sol.t[end]) 
      end
      return  out, rerun
  end

  dummy_prob = ODEProblem(eom, sampler(1.0), (0.0, 1.0))
  return MonteCarloProblem(dummy_prob;output_func=output_func, prob_func=prob_func, u_init = [])
end


function saveTimeOptimalMonteCarloResults(outFile, results, size)
  qNames = [["theta0$n", "omega0$n", "lambda$n", "mu$n", "theta1$n", "omega1$n"]
            for n in 1:size]
  qNames = vcat([[qNames[m][n] for m in 1:size] for n in 1:6]...)
  push!(qNames,"cost")
  push!(qNames,"t_f")

  dataRows = vcat(1:size*6, (size*8 + 1)ones(Int,2))
  selectedData = hcat([u[dataRows] for u in results.u]...)
  df = DataFrame(selectedData')
  names!(df, [Symbol(qn) for qn in qNames])

  CSV.write(outFile, df, delim=',')
  return 0
end
  
