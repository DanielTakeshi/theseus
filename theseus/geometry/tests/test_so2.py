# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import EPS
from theseus.core.tests.common import check_copy_var
from theseus.utils import numeric_jacobian

from .common import (
    check_adjoint,
    check_compose,
    check_exp_map,
    check_inverse,
    check_log_map,
)


def _create_random_so2(batch_size, rng):
    theta = torch.rand(batch_size, 1, generator=rng) * 2 * np.pi - np.pi
    return th.SO2(theta=theta.double())


def test_exp_map():
    for batch_size in [1, 20, 100]:
        theta = torch.from_numpy(np.linspace(-np.pi, np.pi, batch_size)).view(-1, 1)
        check_exp_map(theta, th.SO2)


def test_log_map():
    for batch_size in [1, 2, 100]:
        theta = torch.from_numpy(np.linspace(-np.pi, np.pi, batch_size)).view(-1, 1)
        check_log_map(theta, th.SO2)


def test_compose():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so2_1 = _create_random_so2(batch_size, rng)
        so2_2 = _create_random_so2(batch_size, rng)
        check_compose(so2_1, so2_2)


def test_inverse():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so2 = _create_random_so2(batch_size, rng)
        check_inverse(so2)


def test_rotate_and_unrotate():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in [1, 20, 100]:
            so2 = _create_random_so2(batch_size, rng)
            # Tests that rotate works from tensor. unrotate() would work similarly), but
            # it's also tested indirectly by test_transform_to() for SE2
            point_tensor = torch.randn(batch_size, 2).double()
            jacobians_rotate = []
            rotated_point = so2.rotate(point_tensor, jacobians=jacobians_rotate)
            expected_rotated_data = so2.to_matrix() @ point_tensor.unsqueeze(2)
            jacobians_unrotate = []
            unrotated_point = so2.unrotate(rotated_point, jacobians_unrotate)

            # Check the operation result
            assert torch.allclose(
                expected_rotated_data.squeeze(2), rotated_point.data, atol=EPS
            )
            assert torch.allclose(point_tensor, unrotated_point.data, atol=EPS)

            # Check the jacobians
            # function_dim = 2 because rotate(theta, (x, y)) --> (x_new, y_new)
            expected_jac = numeric_jacobian(
                lambda groups: groups[0].rotate(groups[1]),
                [so2, th.Point2(point_tensor)],
                function_dim=2,
            )
            assert torch.allclose(jacobians_rotate[0], expected_jac[0])
            assert torch.allclose(jacobians_rotate[1], expected_jac[1])
            expected_jac = numeric_jacobian(
                lambda groups: groups[0].unrotate(groups[1]),
                [so2, rotated_point],
                function_dim=2,
            )
            assert torch.allclose(jacobians_unrotate[0], expected_jac[0])
            assert torch.allclose(jacobians_unrotate[1], expected_jac[1])


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so2 = _create_random_so2(batch_size, rng)
        tangent = torch.randn(batch_size, 1).double()
        check_adjoint(so2, tangent)


def test_copy():
    rng = torch.Generator()
    rng.manual_seed(0)
    so2 = _create_random_so2(1, rng)
    check_copy_var(so2)