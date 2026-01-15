"""
Integration Module
Implements numerical integration methods
"""

import numpy as np
from typing import Callable, Dict


def trapezoidal_rule(func: Callable, a: float, b: float, n: int = 100) -> Dict:
    """
    Trapezoidal Rule for numerical integration
    
    Args:
        func: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of subintervals
    
    Returns:
        Dictionary containing integral value and computation details
    """
    try:
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([func(xi) for xi in x])
        
        # Trapezoidal rule: I ≈ h/2 * [f(x0) + 2*f(x1) + 2*f(x2) + ... + 2*f(xn-1) + f(xn)]
        integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
        
        return {
            'success': True,
            'integral': integral,
            'method': 'Trapezoidal Rule',
            'formula': 'I ≈ h/2 * [f(x₀) + 2*Σf(xᵢ) + f(xₙ)]',
            'a': a,
            'b': b,
            'n': n,
            'h': h,
            'subintervals': list(zip(x, y))
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def simpson_one_third(func: Callable, a: float, b: float, n: int = 100) -> Dict:
    """
    Simpson's 1/3 Rule for numerical integration
    
    Args:
        func: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of subintervals (must be even)
    
    Returns:
        Dictionary containing integral value and computation details
    """
    try:
        if n % 2 != 0:
            n += 1  # Make n even
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([func(xi) for xi in x])
        
        # Simpson's 1/3: I ≈ h/3 * [f(x0) + 4*f(x1) + 2*f(x2) + 4*f(x3) + ... + f(xn)]
        integral = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))
        
        return {
            'success': True,
            'integral': integral,
            'method': "Simpson's 1/3 Rule",
            'formula': 'I ≈ h/3 * [f(x₀) + 4*Σf(x₂ᵢ₋₁) + 2*Σf(x₂ᵢ) + f(xₙ)]',
            'a': a,
            'b': b,
            'n': n,
            'h': h,
            'note': 'Number of intervals adjusted to be even' if (b - a) / n != h else ''
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def simpson_three_eighth(func: Callable, a: float, b: float, n: int = 99) -> Dict:
    """
    Simpson's 3/8 Rule for numerical integration
    
    Args:
        func: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of subintervals (must be multiple of 3)
    
    Returns:
        Dictionary containing integral value and computation details
    """
    try:
        # Make n a multiple of 3
        n = ((n // 3) + 1) * 3
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([func(xi) for xi in x])
        
        # Simpson's 3/8: I ≈ 3h/8 * [f(x0) + 3*f(x1) + 3*f(x2) + 2*f(x3) + ... + f(xn)]
        integral = 3 * h / 8 * (
            y[0] + y[-1] +
            3 * np.sum(y[1:-1:3]) +
            3 * np.sum(y[2:-1:3]) +
            2 * np.sum(y[3:-1:3])
        )
        
        return {
            'success': True,
            'integral': integral,
            'method': "Simpson's 3/8 Rule",
            'formula': 'I ≈ 3h/8 * [f(x₀) + 3*f(x₁) + 3*f(x₂) + 2*f(x₃) + ...]',
            'a': a,
            'b': b,
            'n': n,
            'h': h,
            'note': 'Number of intervals adjusted to be multiple of 3'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }