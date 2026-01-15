"""
Numerical Methods Package
Contains implementations of various numerical analysis methods
"""

from .equation_solvers import bisection_method, newton_raphson_method, secant_method
from .differentiation import symbolic_derivative, numerical_derivative
from .integration import trapezoidal_rule, simpson_one_third, simpson_three_eighth
from .interpolation import (
    newton_forward_interpolation,
    newton_backward_interpolation,
    divided_differences_interpolation,
    lagrange_interpolation
)

__all__ = [
    'bisection_method',
    'newton_raphson_method',
    'secant_method',
    'symbolic_derivative',
    'numerical_derivative',
    'trapezoidal_rule',
    'simpson_one_third',
    'simpson_three_eighth',
    'newton_forward_interpolation',
    'newton_backward_interpolation',
    'divided_differences_interpolation',
    'lagrange_interpolation'
]