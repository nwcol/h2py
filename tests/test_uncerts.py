"""Test uncerts module."""

import numpy as np
import os
import pytest

import h2py
from h2py import uncerts


class TestGetScore:

    def test_basic_function(self):
        func = lambda v: 2 * v[0] + 3 * v[1]
        params = np.array([1, 1])
        expected = np.array([2, 3])
        result = uncerts._get_score(params, func)
        assert np.all(np.isclose(result, expected))

    def test_bounded_evaluation(self):
        func = lambda v: 2 * v[0] + 3 * v[1]
        params = np.array([1, 1])
        bounds = (np.array([1, 0]), np.array([2, 1]))
        expected = np.array([2, 3])
        result = uncerts._get_score(params, func, bounds=bounds)
        assert np.all(np.isclose(result, expected))

    def test_step_size_override(self):
        func = lambda v: 2 * v[0] + 3 * v[1]
        params = np.array([1, 1])
        steps = np.array([1e-6, 1e-6])
        expected = np.array([2, 3])
        result = uncerts._get_score(params, func, steps=steps)
        assert np.all(np.isclose(result, expected))

        result = uncerts._get_score(params, func, delta=1e-6)
        assert np.all(np.isclose(result, expected))

    def test_args_input(self):
        def func(x, m, c, args):
            # Arbitrary function with more complex inputs
            fac = m * c + np.sum(args)
            return fac * (2 * x[0] + 3 * x[1])

        params = np.array([1, 1])
        func_args = [2, 2, (3, 4)]
        expected = np.array([22, 33])
        result = uncerts._get_score(params, func, args=func_args)
        assert np.all(np.isclose(result, expected))


class TestGetHessianElemBasic:

    def test_non_mixed_derivative(self):
        func = lambda v: v[0] ** 2 + 3 * v[1] ** 2 + 5 * v[0] * v[1]
        params = np.array([1, 1])
        bounds = (np.zeros(2), np.full(2, 10))
        steps = params * 0.01

        # 0, 0
        result = uncerts._get_hessian_elem(params, 0, 0, func, [], steps,
                                           bounds)
        expected = 2
        assert np.isclose(result, expected)

        # 1, 1
        result = uncerts._get_hessian_elem(params, 1, 1, func, [], steps,
                                           bounds)
        expected = 6
        assert np.isclose(result, expected)

    def test_mixed_derivatives(self):
        func = lambda v: v[0] ** 2 + 3 * v[1] ** 2 + 5 * v[0] * v[1]
        params = np.array([1, 1])
        bounds = (np.zeros(2), np.full(2, 10))
        steps = params * 0.01

        # 0, 1
        result = uncerts._get_hessian_elem(params, 0, 1, func, [], steps,
                                           bounds)
        expected = 5
        assert np.isclose(result, expected)

        # 1, 0
        _result = uncerts._get_hessian_elem(params, 1, 0, func, [], steps,
                                            bounds)
        assert np.isclose(result, expected)


class TestGetHessianElem:

    # Simple bivariate, analytic function as an example
    @staticmethod
    def f(v):
        x, y = v
        return 2 * x ** 3 + y ** 3 + x ** 2 * y ** 2

    @staticmethod
    def f_dxdx(v):
        x, y = v
        return 12 * x + 2 * y ** 2

    @staticmethod
    def f_dydy(v):
        x, y = v
        return 6 * y + 2 * x ** 2

    @staticmethod
    def f_dxdy(v):
        x, y = v
        return 4 * x * y

    def test_bounded_partials(self):
        bounds = (np.zeros(2), np.full(2, np.inf))
        v = np.array([4.55, 3.895])
        delta = 0.01
        steps = delta * v

        # 0, 0 without tight bounds
        expected = self.f_dxdx(v)
        result = uncerts._get_hessian_elem(v, 0, 0, self.f, [], steps, bounds,
                                           return_form=True)
        assert result[1] == "central"
        assert np.isclose(result[0], expected)

        # 1, 1 without tight bounds
        result = uncerts._get_hessian_elem(v, 1, 1, self.f, [], steps, bounds,
                                           return_form=True)
        assert result[1] == "central"
        assert np.isclose(result[0], expected)

    def test_mixed_partials(self):
        bounds = (np.zeros(2), np.full(2, np.inf))
        x = np.array([1.45, 1.76])
        delta = 0.01
        steps = delta * x

        # 0, 1
        expected = self.f_dxdy(x)
        result = uncerts._get_hessian_elem(x, 0, 1, self.f, [], steps, bounds)
        assert np.isclose(result, expected)

        # 1, 0
        _result = uncerts._get_hessian_elem(x, 1, 0, self.f, [], steps, bounds)
        assert np.isclose(result, _result)

    def test_bounded_mixed_partials(self):
        pass


class TestGetHessianMatrix:
    pass





