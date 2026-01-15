"""
Differentiation Module
Implements symbolic and numerical differentiation methods
"""

import sympy as sp
import numpy as np
from typing import Callable, Dict


def symbolic_derivative(expression_str: str, variable: str = 'x', order: int = 1) -> Dict:
    """
    Symbolic differentiation using SymPy
    
    Args:
        expression_str: String representation of the function
        variable: Variable to differentiate with respect to
        order: Order of derivative (1 for first derivative, 2 for second, etc.)
    
    Returns:
        Dictionary containing derivative expression and LaTeX representation
    """
    try:
        x = sp.Symbol(variable)
        expr = sp.sympify(expression_str)
        
        derivative = expr
        for _ in range(order):
            derivative = sp.diff(derivative, x)
        
        return {
            'success': True,
            'original': str(expr),
            'derivative': str(derivative),
            'derivative_latex': sp.latex(derivative),
            'order': order
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def numerical_derivative(func: Callable, x: float, h: float = 1e-5, method: str = 'central') -> Dict:
    """
    Numerical differentiation using finite differences
    
    Args:
        func: Function to differentiate
        x: Point at which to compute derivative
        h: Step size
        method: 'forward', 'backward', or 'central'
    
    Returns:
        Dictionary containing derivative value and method details
    """
    try:
        if method == 'forward':
            # Forward difference: f'(x) ≈ [f(x+h) - f(x)] / h
            derivative = (func(x + h) - func(x)) / h
            formula = "f'(x) ≈ [f(x+h) - f(x)] / h"
            
        elif method == 'backward':
            # Backward difference: f'(x) ≈ [f(x) - f(x-h)] / h
            derivative = (func(x) - func(x - h)) / h
            formula = "f'(x) ≈ [f(x) - f(x-h)] / h"
            
        elif method == 'central':
            # Central difference: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
            derivative = (func(x + h) - func(x - h)) / (2 * h)
            formula = "f'(x) ≈ [f(x+h) - f(x-h)] / (2h)"
            
        else:
            return {
                'success': False,
                'error': 'Invalid method. Choose from: forward, backward, central'
            }
        
        return {
            'success': True,
            'derivative': derivative,
            'method': method,
            'formula': formula,
            'x': x,
            'h': h,
            'f(x)': func(x),
            'f(x+h)': func(x + h) if method in ['forward', 'central'] else None,
            'f(x-h)': func(x - h) if method in ['backward', 'central'] else None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }