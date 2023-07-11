"""Minimum working example for Theseus and robot motion planning w/joint angles."""
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

# Create optimization variables which define a trajectory of joint angles.
poses = []
velocities = []
for i in range(traj_len):
    pose = th.Vector(robot.dof, name=f'pose_{i}', dtype=dtype)
    velocity = th.Vector(robot.dof, name=f'vel_{i}', dtype=dtype)
    poses.append(pose)
    velocities.append(velocity)

# Targets for pose boundary cost functions
start_point = th.Vector(robot.dof, name="start", dtype=dtype)
goal_point = th.Vector(robot.dof, name="goal", dtype=dtype)

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
    th.Difference(poses[0], start_point, boundary_cost_weight, name="pose_0")
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
    th.Difference(poses[-1], goal_point, boundary_cost_weight, name="pose_N")
)
objective.add(
    th.Difference(
        velocities[-1],
        th.Vector(tensor=torch.zeros(1, robot.dof)),
        boundary_cost_weight,
        name="vel_N",
    )
)

# GP-dynamics cost functions for smooth trajectories.
for i in range(1, traj_len):
    objective.add(
        (
            th.eb.GPMotionModel(
                poses[i - 1],
                velocities[i - 1],
                poses[i],
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
start = torch.zeros((1,robot.dof), dtype=dtype).to(device)
goal = torch.rand((1,robot.dof), dtype=dtype).to(device)
print(f'\nOur starting DOFs:\n{start}')
print(f'Our target DOFs:\n{goal}\n')
planner_inputs = {
    "start": start.to(device),
    "goal": goal.to(device),
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

print('\nThe solution, poses:')
for i in range(traj_len):
    print(f'i={str(i).zfill(3)}, {fv[f"pose_{i}"]}')
print('\nThe solution, velocities:')
for i in range(traj_len):
    print(f'i={str(i).zfill(3)}, {fv[f"vel_{i}"]}')
