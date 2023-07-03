"""Minimum working example for Theseus and forward kinematics."""
import os
import torch
torch.set_printoptions(sci_mode=False, linewidth=150)
from torchkin.forward_kinematics import Robot, get_forward_kinematics_fns
dtype = torch.float64

# First we load the URDF file describing the robot and create a `Robot` object to
# represent it in Python. The `Robot` class can be used to build a kinematics tree
# of the robot.
option = 3  # NOTE: pick option in {1, 2, 3}.
if option == 1:
    URDF_REL_PATH = (
        "../tests/theseus_tests/embodied/kinematics/data/panda_no_gripper.urdf"
    )
    link_names = ["panda_virtual_ee_link"]
elif option == 2:
    URDF_REL_PATH = (
        "../tests/theseus_tests/embodied/kinematics/data/ur5_no_translation_joint.urdf"
    )
    link_names = ["tool0"]
elif option == 3:
    URDF_REL_PATH = (
        "../tests/theseus_tests/embodied/kinematics/data/ur5_with_translation_joint.urdf"
    )
    link_names = ["tool0"]

urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)
robot = Robot.from_urdf_file(urdf_path, dtype)

# We can get differentiable forward kinematics functions for specific links
# by using `get_forward_kinematics_fns`. This function creates three differentiable
# functions for evaluating forward kinematics, body jacobian and spatial jacobian of
# the selected links, in that order. The return types of these functions are as
# follows:
#
# - fk: returns a tuple of link poses in the order of link names
# - jfk_b: returns a tuple where the first is a list of link body jacobians, and
#          the second is a tuple of link poses---both are in the order of link names
# - jfk_s: same as jfk_b except returning the spatial jacobians
fk, jfk_b, jfk_s = get_forward_kinematics_fns(robot, link_names)

# Generate target theta, then do forward kinematics to find the EE pose.
target_theta = torch.rand(10, robot.dof, dtype=dtype)
target_pose: torch.Tensor = fk(target_theta)[0]  # shape (10,3,4)
print(target_pose)
