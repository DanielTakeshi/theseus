"""Minimum working example for Theseus and robot motion planning w/EE poses."""
import os
import random
import torch
import numpy as np
import theseus as th
from torchkin.forward_kinematics import Robot, get_forward_kinematics_fns
torch.set_printoptions(sci_mode=False, linewidth=150, precision=4)

# Bells and whistles.
dtype = torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
seed = 0
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# -------------------------- Load the robot --------------------------- #

# See examples/inverse_kinematics.py for details.
URDF_REL_PATH = "../tests/theseus_tests/embodied/kinematics/data/ur5_no_translation_joint.urdf"
link_names = ["tool0"]
urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)
robot = Robot.from_urdf_file(urdf_path, dtype)
print(f'Just loaded robot with DOFs: {robot.dof}')

# NOTE: we are NOT using these, should we be using them?
fk, jfk_b, jfk_s = get_forward_kinematics_fns(robot, link_names)

# -------------------------- Theseus optimization --------------------------- #

# Variables for the optimization, based on 04_motion_planning.ipynb.
traj_len = 100
n_steps = traj_len - 1
total_time = 10.0
dt_val = total_time / n_steps
Qc_inv = torch.eye(robot.dof)
boundary_w = 100.0

# Create optimization variables which define a trajectory of EE poses.
ee_poses = []
velocities = []  # Question: do we need velocities here?
for i in range(traj_len):
    ee_pose = th.SE3(name=f'ee_pose_{i}', dtype=dtype)
    ee_poses.append(ee_pose)
    velocity = th.Vector(robot.dof, name=f'vel_{i}', dtype=dtype)
    velocities.append(velocity)

# Targets for pose boundary cost functions in SE(3).
start_point = th.SE3(name="start", dtype=dtype)
goal_point = th.SE3(name="goal", dtype=dtype)

# For GP dynamics cost function
dt = th.Variable(torch.tensor(dt_val).view(1, 1), name="dt")

# Cost weight to use for all GP-dynamics cost functions
gp_cost_weight = th.eb.GPCostWeight(torch.tensor(Qc_inv), dt)

# For all hard-constraints (end points pos/vel) we use a single scalar weight
# with high value
boundary_cost_weight = th.ScaleCostWeight(boundary_w)

# First, create the objective.
objective = th.Objective(dtype=dtype)

# Boundary cost functions -- initial position and velocity.
objective.add(
    th.Difference(ee_poses[0], start_point, boundary_cost_weight, name="ee_pose_0")
)
objective.add(
    th.Difference(
        velocities[0],
        th.Vector(tensor=torch.zeros(1, robot.dof)),
        boundary_cost_weight,
        name="vel_0",
    )
)

# Boundary cost functions -- final position and velocity.
objective.add(
    th.Difference(ee_poses[-1], goal_point, boundary_cost_weight, name="ee_pose_N")
)
objective.add(
    th.Difference(
        velocities[-1],
        th.Vector(tensor=torch.zeros(1, robot.dof)),
        boundary_cost_weight,
        name="vel_N",
    )
)

# -------------------------------------------------------------------------- #
# GP-dynamics cost functions for smooth trajectories.
# Do we need this for EE pose trajectories?
# -------------------------------------------------------------------------- #
# We have to consider all the intermediate poses somehow. My hack is to
# assume we can take a difference between poses but not sure if this is
# correct and naive difference is not ideal right? Or maybe we can still use
# EE pose trajectories because interestingly the GPMotionModel uses LieGroups
# for poses, but Vectors for velocities. Update: well this code amazingly runs
# but again, need to check, if this works.
# -------------------------------------------------------------------------- #
for i in range(1, traj_len):
    objective.add(
        (
            th.eb.GPMotionModel(
                ee_poses[i - 1],
                velocities[i - 1],
                ee_poses[i],
                velocities[i],
                dt,
                gp_cost_weight,
                name=f"gp_{i}",
            )
        )
    )

# Using the standard LM optimizer as in other Theseus examples.
optimizer = th.LevenbergMarquardt(
    objective,
    th.CholeskyDenseSolver,
    max_iterations=50,
    step_size=1.0,
)
motion_planner = th.TheseusLayer(optimizer)
motion_planner.to(device=device, dtype=dtype)

# Now run optimization with random target? Batch size 1.
rand_joints = torch.rand((1,robot.dof), dtype=dtype)
ee_start = th.SE3()
ee_goal = th.SE3(tensor=fk(rand_joints)[0])  # get a 'valid' EE goal pose?
ee_start.to(device)
ee_goal.to(device)
print(f'\nOur starting EE pose:\n{ee_start}')
print(f'\nOur target EE pose:\n{ee_goal}\n')
planner_inputs = {
    "start": ee_start,
    "goal": ee_goal,
}
with torch.no_grad():
    fv, info = motion_planner.forward(
        planner_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": True,
            "damping": 0.1,
        }
    )

# The fv[ee_pose_k] has type tensor, not th.SE3, but it has the 'SE3' info.
print('\nThe solution, EE poses:')
for i in range(traj_len):
    print(f'i={str(i).zfill(3)}\n{fv[f"ee_pose_{i}"]}')
    # eep = fv[f"ee_pose_{i}"]  # (1,3,4)
    # rot_mat = eep[0][:,:3]
    # RTR = torch.t(rot_mat).matmul(rot_mat)
    # print(RTR)  # Sanity check, I do see that this is the identity
print('\nThe solution, velocities:')
for i in range(traj_len):
    print(f'i={str(i).zfill(3)}, {fv[f"vel_{i}"]}')
