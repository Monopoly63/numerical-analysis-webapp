"""
Equation Solvers Module
Implements numerical methods for finding roots of equations
"""

import numpy as np
from typing import Callable, Tuple, List, Dict


def bisection_method(func: Callable, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> Dict:
    """
    Bisection Method for finding roots
    
    Args:
        func: Function to find root of
        a: Left endpoint of interval
        b: Right endpoint of interval
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        Dictionary containing root, iterations, and step-by-step results
    """
    steps = []
    
    if func(a) * func(b) >= 0:
        return {
            'success': False,
            'error': 'Function must have opposite signs at endpoints a and b',
            'steps': []
        }
    
    iteration = 0
    while iteration < max_iter:
        c = (a + b) / 2
        fc = func(c)
        fa = func(a)
        fb = func(b)
        
        steps.append({
            'iteration': iteration + 1,
            'a': a,
            'b': b,
            'c': c,
            'f(a)': fa,
            'f(b)': fb,
            'f(c)': fc,
            'error': abs(b - a)
        })
        
        if abs(fc) < tol or abs(b - a) < tol:
            return {
                'success': True,
                'root': c,
                'iterations': iteration + 1,
                'steps': steps,
                'final_error': abs(b - a)
            }
        
        if fa * fc < 0:
            b = c
        else:
            a = c
        
        iteration += 1
    
    return {
        'success': False,
        'error': 'Maximum iterations reached',
        'root': (a + b) / 2,
        'iterations': max_iter,
        'steps': steps
    }


def newton_raphson_method(func: Callable, derivative: Callable, x0: float, 
                         tol: float = 1e-6, max_iter: int = 100) -> Dict:
    """
    Newton-Raphson Method for finding roots
    
    Args:
        func: Function to find root of
        derivative: Derivative of the function
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        Dictionary containing root, iterations, and step-by-step results
    """
    steps = []
    x = x0
    
    for iteration in range(max_iter):
        fx = func(x)
        dfx = derivative(x)
        
        if abs(dfx) < 1e-12:
            return {
                'success': False,
                'error': 'Derivative too close to zero',
                'steps': steps
            }
        
        x_new = x - fx / dfx
        error = abs(x_new - x)
        
        steps.append({
            'iteration': iteration + 1,
            'x': x,
            'f(x)': fx,
            "f'(x)": dfx,
            'x_new': x_new,
            'error': error
        })
        
        if error < tol:
            return {
                'success': True,
                'root': x_new,
                'iterations': iteration + 1,
                'steps': steps,
                'final_error': error
            }
        
        x = x_new
    
    return {
        'success': False,
        'error': 'Maximum iterations reached',
        'root': x,
        'iterations': max_iter,
        'steps': steps
    }


def secant_method(func: Callable, x0: float, x1: float, 
                 tol: float = 1e-6, max_iter: int = 100) -> Dict:
    """
    Secant Method for finding roots
    
    Args:
        func: Function to find root of
        x0: First initial guess
        x1: Second initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        Dictionary containing root, iterations, and step-by-step results
    """
    steps = []
    
    for iteration in range(max_iter):
        fx0 = func(x0)
        fx1 = func(x1)
        
        if abs(fx1 - fx0) < 1e-12:
            return {
                'success': False,
                'error': 'Division by zero encountered',
                'steps': steps
            }
        
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x2 - x1)
        
        steps.append({
            'iteration': iteration + 1,
            'x0': x0,
            'x1': x1,
            'f(x0)': fx0,
            'f(x1)': fx1,
            'x2': x2,
            'error': error
        })
        
        if error < tol:
            return {
                'success': True,
                'root': x2,
                'iterations': iteration + 1,
                'steps': steps,
                'final_error': error
            }
        
        x0, x1 = x1, x2
    
    return {
        'success': False,
        'error': 'Maximum iterations reached',
        'root': x1,
        'iterations': max_iter,
        'steps': steps
    }