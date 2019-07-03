
struct ExtendedMechanism
  mechanism::Mechanism
  resultsCache::DynamicsResultCache
  stateCache::StateCache
  size::Int
  bounds::Vector
end

ExtendedMechanism(m::Mechanism) = ExtendedMechanism(m, ones(num_positions(m)))
ExtendedMechanism(m::Mechanism, b) = ExtendedMechanism(m, DynamicsResultCache(m),
                                                    StateCache(m),
                                                    num_positions(m),
                                                    b)


equalSplit(a,n) = [a[(m-1)*n+1:m*n] for m = 1:div(length(a),n)] 


function computeTermsForHamiltonian(m, fullState)
  result = m.resultsCache[eltype(fullState)]
  state = m.stateCache[eltype(fullState)]
  q, v, lambda, mu = equalSplit(fullState, m.size)
  set_configuration!(state, q)
  set_velocity!(state, v)
  dynamics!(result, state)
  Mmu = result.massmatrix\mu
  return v, lambda, Mmu, result.dynamicsbias
end

function timeOptimalHamiltonian(m, fullState)
   v, lambda, Mmu, bias = computeTermsForHamiltonian(m, fullState)
  return 1.0  + dot(lambda, v) - dot(Mmu, bias) - dot(abs.(Mmu), m.bounds)
end

function energyOptimalHamiltonian(m, fullState)
  v, lambda, Mmu, bias = computeTermsForHamiltonian(m, fullState)
  return 0.5*dot(Mmu,Mmu) + 1.0  + dot(lambda, v) + dot(Mmu, bias)
end

function alphaTimeOptimal(mech, fullState)
  H1 = timeOptimalHamiltonian(mech, fullState)
  alpha = -1/(H1 - 1)
  return alpha
end


gradEnergyOptimalHamiltonian(m, s) = ForwardDiff.gradient(x ->
                                                          energyOptimalHamiltonian(m, x), s)
gradTimeOptimalHamiltonian(m, s) = ForwardDiff.gradient(x ->
                                                        timeOptimalHamiltonian(m, x), s)

gradTimeOptimalHamiltonian!(r, m, s) =  ForwardDiff.gradient!(r, x ->
                                                        timeOptimalHamiltonian(m, x), s)



function optimalEOM(gradHamiltonian, mech, fullState)
  nablaH = gradHamiltonian(mech, fullState)
  sDot = nablaH[2*mech.size+1:end]
  cDot = -nablaH[1:2*mech.size]
  return vcat(sDot,cDot)
end

function optimalEOM!(stateDot, gradHamiltonian!, mech, fullState)
  gradHamiltonian!(stateDot, mech, fullState)
  stateDot[:] = vcat(stateDot[2*mech.size+1:end], -stateDot[1:2*mech.size])
  return stateDot
end


function optimalEOM!(stateDot, gradHamiltonian!, mech, fullState, forward)
  sd = optimalEOM!(stateDot, gradHamiltonian!, mech, fullState)
  if !(forward)
    stateDot[:] = -stateDot[:]
  end
  return stateDot
end
