using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms


urdf = "kuka_iiwa/model3DOF.urdf"
mechanism = parse_urdf(urdf)
vis = Visualizer()
open(vis)
mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf), vis)

set_configuration!(mvis, [-1.4, 1.05, -1.4])

set_configuration!(mvis, [1.4, 0.6, -0.9])


planar = "2dof_z.urdf"
arm = parse_urdf(planar)

vis = Visualizer()
open(vis)
mvis = MechanismVisualizer(arm, URDFVisuals(planar), vis)

set_configuration!(mvis, [-0.785, -0.785])

set_configuration!(mvis, [0.785, 0.785, ])
