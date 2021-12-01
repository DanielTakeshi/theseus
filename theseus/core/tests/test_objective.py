# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th

from .common import (
    MockCostFunction,
    MockCostWeight,
    MockVar,
    NullCostWeight,
    check_another_theseus_tensor_is_copy,
    create_mock_cost_functions,
    create_objective_with_mock_cost_functions,
)


def test_add():
    objective, cost_functions, names, *_ = create_objective_with_mock_cost_functions()

    for cost_function, name in zip(cost_functions, names):
        assert name in objective.cost_functions
        assert objective.cost_functions[name] == cost_function

    # --------- All the code that follows test for handling of errors ---------
    def _create_cost_function_with_n_vars_and_m_aux(
        cost_function_name, var_names, aux_names, cost_weight
    ):
        the_vars = [MockVar(1, name=name_) for name_ in var_names]
        the_aux = [MockVar(1, name=name_) for name_ in aux_names]
        return MockCostFunction(
            the_vars,
            the_aux,
            cost_weight,
            name=cost_function_name,
        )

    # ###  Testing no duplicate names allowed ###
    # duplicate cost function name
    new_cost_function = _create_cost_function_with_n_vars_and_m_aux(
        cost_functions[0].name, ["none1", "none2"], [], NullCostWeight()
    )
    with pytest.raises(ValueError):
        objective.add(new_cost_function)
    # duplicate variable name
    new_cost_function = _create_cost_function_with_n_vars_and_m_aux(
        "new", [cost_functions[0].optim_var_0.name], [], NullCostWeight()
    )
    with pytest.raises(ValueError):
        objective.add(new_cost_function)

    # This for loop tests two failure cases:
    # i) duplicate aux. variable name and
    # ii) same name used both for optimization var and aux. variable
    for aux_name in [
        cost_functions[0].aux_var_0.name,
        cost_functions[0].optim_var_0.name,
    ]:
        new_cost_function = _create_cost_function_with_n_vars_and_m_aux(
            f"another_{aux_name}", ["some_name"], [aux_name], NullCostWeight()
        )
        with pytest.raises(ValueError):
            objective.add(new_cost_function)

    # -------------- Test similar stuff but for cost weight --------
    cost_weight = MockCostWeight(
        th.Variable(torch.ones(1), name="cost_weight_aux"),
        add_dummy_var_with_name="cost_weight_var_1",
    )
    yet_another_cost_function = _create_cost_function_with_n_vars_and_m_aux(
        "yet_another", ["yet_var_1"], ["yet_aux_1"], cost_weight
    )
    with pytest.warns(RuntimeWarning):  # optim var associated to weight
        objective.add(yet_another_cost_function)  # no conflict here

    cost_weight_with_conflict_in_aux_var = MockCostWeight(
        th.Variable(torch.ones(1), name="cost_weight_aux")
    )
    cost_function_with_cost_weight_conflict = (
        _create_cost_function_with_n_vars_and_m_aux(
            "yet_another_2",
            ["yet_var_2"],
            "yet_aux_2",
            cost_weight_with_conflict_in_aux_var,
        )
    )
    with pytest.raises(ValueError):
        objective.add(cost_function_with_cost_weight_conflict)

    cost_weight_with_conflict_in_var = MockCostWeight(
        th.Variable(torch.ones(1), name="cost_weight_aux"),
        add_dummy_var_with_name="cost_weight_var_1",
    )
    another_cost_function_with_cost_weight_conflict = (
        _create_cost_function_with_n_vars_and_m_aux(
            "yet_another_3",
            ["yet_var_3"],
            "yet_aux_3",
            cost_weight_with_conflict_in_var,
        )
    )
    with pytest.raises(ValueError):
        objective.add(another_cost_function_with_cost_weight_conflict)


def test_getters():
    (
        objective,
        cost_functions,
        names,
        var_to_cost_functions,
        aux_to_cost_functions,
    ) = create_objective_with_mock_cost_functions()
    for cost_function, name in zip(cost_functions, names):
        assert id(objective.get_cost_function(name)) == id(cost_function)

    for var in var_to_cost_functions:
        assert id(var) == id(objective.get_optim_var(var.name))

    for aux in aux_to_cost_functions:
        assert id(aux) == id(objective.get_aux_var(aux.name))

    assert objective.get_cost_function("bad-name") is None
    assert objective.get_optim_var("bad-name") is None
    assert objective.get_aux_var("bad-name") is None


def test_has_cost_function_and_has_var():
    objective, cost_functions, names, *_ = create_objective_with_mock_cost_functions()

    for name in names:
        assert objective.has_cost_function(name)

    for cost_function in cost_functions:
        for var in cost_function.optim_vars:
            assert objective.has_optim_var(var.name)

    for cost_function in cost_functions:
        for aux in cost_function.aux_vars:
            assert objective.has_aux_var(aux.name)


def test_add_and_erase_step_by_step():
    var1 = MockVar(1, data=None, name="var1")
    var2 = MockVar(1, data=None, name="var2")
    var3 = MockVar(1, data=None, name="var3")
    aux1 = MockVar(1, data=None, name="aux1")
    aux2 = MockVar(1, data=None, name="aux2")
    cw1 = MockCostWeight(aux1, name="cw1", add_dummy_var_with_name="ignored_optim_var")
    cw2 = MockCostWeight(aux2, name="cw2", add_optim_var=var1)

    cf1 = MockCostFunction([var1, var2], [aux1, aux2], cw1, name="cf1")
    cf2 = MockCostFunction([var1, var3], [aux1], cw2, name="cf2")
    cf3 = MockCostFunction([var2, var3], [aux2], cw2, name="cf3")

    objective = th.Objective()
    for cost_function in [cf1, cf2, cf3]:
        if cost_function is not cf3:
            with pytest.warns(RuntimeWarning):
                # a warning should emit the first time cw1/cw2 are added
                objective.add(cost_function)
        else:
            objective.add(cost_function)

    for name in ["var1", "ignored_optim_var"]:
        assert name in objective.cost_weight_optim_vars
    for name in ["var1", "var2", "var2"]:
        assert name in objective.optim_vars
    for name in ["aux1", "aux2"]:
        assert name in objective.aux_vars

    # Checks that the objective is maintaining variable -> fn correctly, against
    # a list of expected functions
    def _check_funs_for_variable(
        var_, expected_fn_list_, optim_var=True, is_cost_weight_optim=False
    ):
        # If var should have been deleted, check that no trace of it remains
        if not expected_fn_list_:
            if optim_var:
                assert var_ not in objective.functions_for_optim_vars
                if is_cost_weight_optim:
                    assert var_ not in objective.cost_weight_optim_vars
                else:
                    assert var_ not in objective.optim_vars
            else:
                assert var_ not in objective.functions_for_aux_vars
                assert var_ not in objective.aux_vars
            return

        # Otherwise check that list of connected functions match the expected
        if optim_var:
            obj_container = objective.functions_for_optim_vars[var_]
        else:
            obj_container = objective.functions_for_aux_vars[var_]
        assert len(obj_container) == len(expected_fn_list_)
        for fun_ in expected_fn_list_:
            assert fun_ in obj_container

    # Runs the above check for all variables in this graph, given expected lists
    # for each
    def _check_all_vars(v1_lis_, v2_lis_, v3_lis_, cw1_opt_lis_, a1_lis_, a2_lis_):
        _check_funs_for_variable(var1, v1_lis_)
        _check_funs_for_variable(var2, v2_lis_)
        _check_funs_for_variable(var3, v3_lis_)
        _check_funs_for_variable(
            cw1.optim_var_at(0), cw1_opt_lis_, is_cost_weight_optim=True
        )
        _check_funs_for_variable(aux1, a1_lis_, optim_var=False)
        _check_funs_for_variable(aux2, a2_lis_, optim_var=False)

    v1_lis = [cw2, cf1, cf2]
    v2_lis = [cf1, cf3]
    v3_lis = [cf2, cf3]
    cw1o_lis = [cw1]
    a1_lis = [cw1, cf1, cf2]
    a2_lis = [cw2, cf1, cf3]

    _check_all_vars(v1_lis, v2_lis, v3_lis, cw1o_lis, a1_lis, a2_lis)

    objective.erase("cf1")
    v1_lis = [cw2, cf2]
    v2_lis = [cf3]
    cw1o_lis = []
    a1_lis = [cf2]  # cf1 and cw1 are deleted, since cw1 not used by any other cost fn
    a2_lis = [cw2, cf3]
    _check_all_vars(v1_lis, v2_lis, v3_lis, cw1o_lis, a1_lis, a2_lis)
    assert cw1 not in objective.cost_functions_for_weights

    objective.erase("cf2")
    v1_lis = [cw2]  # cw2 still used by cf3
    v3_lis = [cf3]
    a1_lis = []
    _check_all_vars(v1_lis, v2_lis, v3_lis, cw1o_lis, a1_lis, a2_lis)
    assert cw2 in objective.cost_functions_for_weights

    objective.erase("cf3")
    v1_lis = []
    v2_lis = []
    v3_lis = []
    a2_lis = []
    _check_all_vars(v1_lis, v2_lis, v3_lis, cw1o_lis, a1_lis, a2_lis)
    assert cw2 not in objective.cost_functions_for_weights


def test_objective_error():
    vars = []
    # Create var_i with data = i, for i in {0, 1, 2}
    for i in range(3):
        data = i * torch.ones(1, 1)
        var = MockVar(1, data=data, name=f"var{i}")
        var.update(data)
        vars.append(var)

    # Adds all three possible cost functions with var_i and var_j to the objective
    cov = NullCostWeight()
    objective = th.Objective()
    expected_objective_error = []
    for i, v1 in enumerate(vars):
        for j, v2 in enumerate(vars):
            if i == j:
                continue
            cost_function = MockCostFunction([v1, v2], [], cov)
            objective.add(cost_function)
            cost_function_error = cost_function.error().numpy()
            expected_objective_error.append(cost_function_error)
            # Mock cost function computes error as (i + j)[1, 1] for var_i, var_j
            np.testing.assert_almost_equal(cost_function_error, (i + j) * np.ones(2))
    # Given these cost functionss,
    # error is just the concatenation of the cost function errors above
    expected_objective_error = np.concatenate(expected_objective_error)

    # batch_size is set by update, but for this test we just want to check
    # that the error method works properly (no pass through update)
    objective._batch_size = 1
    error = objective.error().squeeze().numpy()
    np.testing.assert_almost_equal(error, expected_objective_error)

    # Test the squared error function
    squared_error = np.sum(expected_objective_error ** 2)
    np.testing.assert_almost_equal(
        objective.error_squared_norm().numpy(), squared_error
    )


def test_get_cost_functions_connected_to_vars():
    (
        objective,
        _,
        _,
        optim_var_to_cost_functions,
        aux_to_cost_functions,
    ) = create_objective_with_mock_cost_functions()

    def _check_connections(var_to_cost_fns, objective_get_var_method):
        for variable, expected_cost_functions in var_to_cost_fns.items():

            def _check_cost_functions(cost_functions):
                assert len(cost_functions) == len(expected_cost_functions)
                for cost_function in cost_functions:
                    assert cost_function in expected_cost_functions

            _check_cost_functions(objective_get_var_method(variable))
            _check_cost_functions(objective_get_var_method(variable.name))

            with pytest.raises(ValueError):
                objective_get_var_method("var_not_in_objective")
                objective_get_var_method(MockVar(None, name="none"))

    _check_connections(
        optim_var_to_cost_functions, objective.get_functions_connected_to_optim_var
    )
    _check_connections(
        aux_to_cost_functions, objective.get_functions_connected_to_aux_var
    )


def test_copy():
    objective, cost_functions, *_ = create_objective_with_mock_cost_functions(
        torch.ones(1, 1), MockCostWeight(th.Variable(torch.ones(1)))
    )
    new_objective = objective.copy()
    assert new_objective is not objective

    assert objective.size_cost_functions() == new_objective.size_cost_functions()
    for cost_function, new_cost_function in zip(
        cost_functions, new_objective.cost_functions.values()
    ):
        assert cost_function.num_optim_vars() == new_cost_function.num_optim_vars()
        assert cost_function.num_aux_vars() == new_cost_function.num_aux_vars()
        assert cost_function.name == new_cost_function.name
        assert cost_function is not new_objective
        for var, new_var in zip(cost_function.optim_vars, new_cost_function.optim_vars):
            check_another_theseus_tensor_is_copy(var, new_var)
        for aux, new_aux in zip(cost_function.aux_vars, new_cost_function.aux_vars):
            check_another_theseus_tensor_is_copy(aux, new_aux)


def test_copy_no_duplicate_cost_weights():
    objective = th.Objective()
    v1 = MockVar(1, name="v1")
    cw1 = th.ScaleCostWeight(1.0)
    cw2 = th.ScaleCostWeight(2.0)
    objective.add(MockCostFunction([v1], [], cw1, "cf1"))
    objective.add(MockCostFunction([v1], [], cw1, "cf2"))
    objective.add(MockCostFunction([v1], [], cw2, "cf3"))
    objective.add(MockCostFunction([v1], [], cw2, "cf4"))
    objective.add(MockCostFunction([v1], [], cw2, "cf5"))

    objective_copy = objective.copy()
    seen_cw1 = set()
    seen_cw2 = set()
    for cf in objective_copy.cost_functions.values():
        if cf.name in ["cf1", "cf2"]:
            scale = 1.0
            original_weight = cw1
            set_to_add = seen_cw1
        else:
            scale = 2.0
            original_weight = cw2
            set_to_add = seen_cw2

        assert isinstance(cf.weight, th.ScaleCostWeight)
        assert cf.weight.scale.data.item() == scale
        assert cf.weight is not original_weight
        set_to_add.add(cf.weight)
    assert len(seen_cw1) == 1
    assert len(seen_cw2) == 1


def test_update_updates_properly():
    (
        objective,
        _,
        _,
        var_to_cost_functions,
        aux_to_cost_functions,
    ) = create_objective_with_mock_cost_functions(
        torch.ones(1, 1),
        MockCostWeight(th.Variable(torch.ones(1), name="cost_weight_aux")),
    )

    input_data = {}
    for var in var_to_cost_functions:
        input_data[var.name] = 2 * var.data.clone()
    for aux in aux_to_cost_functions:
        input_data[aux.name] = 2 * aux.data.clone()

    objective.update(input_data=input_data)
    assert objective.batch_size == 1

    for var_name, data in input_data.items():
        if var_name in [v.name for v in var_to_cost_functions]:
            var_ = objective.get_optim_var(var_name)
        if var_name in [aux.name for aux in aux_to_cost_functions]:
            var_ = objective.get_aux_var(var_name)
        assert data is var_.data


def test_update_raises_batch_size_error():
    (
        objective,
        _,
        _,
        var_to_cost_functions,
        aux_to_cost_functions,
    ) = create_objective_with_mock_cost_functions(
        torch.ones(1, 1),
        MockCostWeight(th.Variable(torch.ones(1), name="cost_weight_aux")),
    )

    input_data = {}
    batch_size = 2
    # first check that we can change the current batch size (doubling the size)s
    for var in var_to_cost_functions:
        new_data = torch.ones(batch_size, 1)
        input_data[var.name] = new_data
    for aux in aux_to_cost_functions:
        new_data = torch.ones(batch_size, 1)
        input_data[aux.name] = new_data
    objective.update(input_data=input_data)
    assert objective.batch_size == batch_size

    # change one of the variables, no error since batch_size = 1 is broadcastable
    input_data["var1"] = torch.ones(1, 1)
    objective.update(input_data=input_data)
    assert objective.batch_size == batch_size

    # change another variable, this time throws errors since found batch size 2 and 3
    input_data["var2"] = torch.ones(batch_size + 1, 1)
    with pytest.raises(ValueError):
        objective.update(input_data=input_data)

    # change back before testing the aux. variable
    input_data["var2"] = torch.ones(batch_size, 1)
    objective.update(input_data=input_data)  # shouldn't throw error

    # auxiliary variables should also throw error
    input_data["aux1"] = torch.ones(batch_size + 1, 1)
    with pytest.raises(ValueError):
        objective.update(input_data=input_data)


def test_iterator():
    cost_functions, *_ = create_mock_cost_functions()

    objective = th.Objective()
    for f in cost_functions:
        objective.add(f)

    idx = 0
    for f in objective:
        assert f == cost_functions[idx]
        idx += 1


def test_to_dtype():
    objective, *_ = create_objective_with_mock_cost_functions()
    for dtype in [torch.float32, torch.float64, torch.long]:
        objective.to(dtype=dtype)
        for _, cf in objective.cost_functions.items():
            for var in cf.optim_vars:
                assert var.dtype == dtype
            for aux in cf.aux_vars:
                assert aux.dtype == dtype