using RigidBodyDynamics


struct UrdfMechanism <: Mechanisms
  dof::Integer
  mechanism::RigidBodyDynamics.Mechanism{Float64}
  max_torques::Array{Float64,1}
  state_cache::RigidBodyDynamics.StateCache

  function UrdfMechanism(str, max_torques)
    m = parse_urdf(Float64, str)
    s = MechanismState{Float64}(m)
    v = Array(velocity(s))
    if length(max_torques) == length(v)
      n = length(max_torques)
    else
      print(length(v))
      error("The max_torque dimension does not match the DOF of the mechanism")
    end
    new(n, m, max_torques, StateCache(m))
  end
end

function massMatrix(theta::AbstractVector{T}, m::UrdfMechanism) where T
  state = m.state_cache[T]
  set_configuration!(state, theta)
  setdirty!(state)
  return Array(mass_matrix(state))
end

function biasVector(theta::AbstractVector{T1}, omega::AbstractVector{T2},
                    m::UrdfMechanism) where {T1,T2}
  if !(T2 == Float64)
    T = T2
  else
    T = T1
  end
  state = m.state_cache[T]
  set_configuration!(state, theta)
  set_velocity!(state, omega) 
  setdirty!(state)
  return Array(dynamics_bias(state))
end

function mechanismDynamicsTerms(theta, omega, m::UrdfMechanism)
  return massMatrix(theta, m), biasVector(theta, omega,m)
end

dof(m::UrdfMechanism) = m.dof
