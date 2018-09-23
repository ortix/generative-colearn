
struct NLinkPlanarMechanism <: Mechanisms
    dof::Integer
    link_lengths::Array{Float64,1}
    link_masses::Array{Float64,1}
    max_torques::Array{Float64,1}
  
    function NLinkPlanarMechanism(link_lengths, link_masses, max_torques)
      new(length(link_lengths),link_lengths, link_masses, max_torques)
    end
  
    function NLinkPlanarMechanism(n::Integer)
      new(n, ones(Float64, n), ones(Float64, n), ones(Float64, n))
    end
  end
  
  
  function dof(m::NLinkPlanarMechanism)
      return m.dof
  end
  
  
  function T(theta::AbstractVector{Z}, mechanism::NLinkPlanarMechanism) where Z
    link_lengths = mechanism.link_lengths
    t = zeros(Z,2*mechanism.dof,mechanism.dof)
    for c1 = 1:mechanism.dof
      s = -link_lengths[c1]*sin(theta[c1])
      c = link_lengths[c1]*cos(theta[c1])
      for c2 = (c1*2):2:2*mechanism.dof
        t[c2-1,c1] = s
        t[c2 ,c1] = c
      end
    end
    return t
  end
  
  function convectiveAccelerations(theta::AbstractVector{Z1},
                                   omega::AbstractVector{Z2},
                                   mechanism::NLinkPlanarMechanism) where  {Z1, Z2}
    link_lengths = mechanism.link_lengths
    if !(Z2 == Float64)
      Z = Z2
    else
      Z = Z1
    end
    conv = zeros(Z,2*mechanism.dof)
    conv[1] = -link_lengths[1]*cos(theta[1])*(omega[1]^2)
    conv[2] = -link_lengths[1]*sin(theta[1])*(omega[1]^2)
    for c1 = 2:mechanism.dof
      s = -link_lengths[c1]*cos(theta[c1])*(omega[c1]^2)
      c = -link_lengths[c1]*sin(theta[c1])*(omega[c1]^2)
      conv[c1*2-1] = conv[c1*2 - 3] + s
      conv[c1*2] = conv[c1*2 - 2] + c
    end
    return conv
  end
  
  function xMass(mechanism::NLinkPlanarMechanism)
    link_mass = mechanism.link_masses
    m1 = zeros(2*mechanism.dof)
    n = 1
    for c1 = 1:2:2*mechanism.dof
      m1[c1] = link_mass[n]
      m1[c1+1] = link_mass[n]
      n += 1
    end
    return diagm(m1)
  end
  
  function tTM(jacobian, mechanism::NLinkPlanarMechanism)
    return transpose(jacobian)*xMass(mechanism)
  end
  
  function mechanismMass(tm, jac, mechanism::NLinkPlanarMechanism)
    M = tm*jac
    return M
  end
  
  function mechanismMass(theta, mechanism)
    j = T(theta, mechanism)
    tm = tTM(j, mechanism)
    return mechanismMass(tm, j , mechanism)
  end
  
  function mechanismDynamicsTerms(theta, omega, mechanism::NLinkPlanarMechanism)
    j = T(theta, mechanism)
    tm = tTM(j,mechanism)
    M = mechanismMass(tm,j,mechanism)
    G = tm*convectiveAccelerations(theta,omega,mechanism)
    return M, G
  end
  
  
  