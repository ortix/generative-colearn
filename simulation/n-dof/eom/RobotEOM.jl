module RobotEOM

export Mechanisms, UrdfMechanism, NLinkPlanarMechanism, Pendulum,
       mechanismDynamicsTerms, robotDOF 

abstract type Mechanisms end

include("nlinkeom.jl")
include("urdfeom.jl")
robotDOF(m::Mechanisms) = dof(m)

end
