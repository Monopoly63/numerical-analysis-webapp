"""
Interpolation Module
Implements various interpolation methods with difference tables
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def create_forward_difference_table(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Create forward difference table"""
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]
    
    columns = ['y'] + [f'Δ^{i}y' for i in range(1, n)]
    df = pd.DataFrame(table, columns=columns)
    df.insert(0, 'x', x)
    return df


def create_backward_difference_table(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Create backward difference table"""
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    
    for j in range(1, n):
        for i in range(j, n):
            table[i, j] = table[i, j - 1] - table[i - 1, j - 1]
    
    columns = ['y'] + [f'∇^{i}y' for i in range(1, n)]
    df = pd.DataFrame(table, columns=columns)
    df.insert(0, 'x', x)
    return df


def create_divided_difference_table(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Create divided difference table"""
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    
    columns = ['f[x]'] + [f'f[x,...,x+{i}]' for i in range(1, n)]
    df = pd.DataFrame(table, columns=columns)
    df.insert(0, 'x', x)
    return df


def newton_forward_interpolation(x: np.ndarray, y: np.ndarray, x_interp: float) -> Dict:
    """
    Newton Forward Interpolation
    Best for interpolation near the beginning of the table
    
    Args:
        x: Array of x values (equally spaced)
        y: Array of y values
        x_interp: Value to interpolate at
    
    Returns:
        Dictionary containing interpolated value, polynomial, and difference table
    """
    try:
        n = len(x)
        h = x[1] - x[0]
        
        # Create forward difference table
        diff_table = create_forward_difference_table(x, y)
        
        # Calculate u = (x_interp - x[0]) / h
        u = (x_interp - x[0]) / h
        
        # Newton forward interpolation formula
        result = y[0]
        u_term = u
        factorial = 1
        
        terms = [f"{y[0]:.6f}"]
        
        for i in range(1, n):
            factorial *= i
            coeff = diff_table.iloc[0, i + 1]  # +1 because x column is at index 0
            term = coeff * u_term / factorial
            result += term
            
            # Build polynomial term
            u_str = " * ".join([f"(u - {j})" for j in range(i)])
            terms.append(f"({coeff:.6f} * {u_str}) / {factorial}")
            
            u_term *= (u - i)
        
        polynomial = " + ".join(terms)
        polynomial = f"P(u) = {polynomial}, where u = (x - {x[0]}) / {h}"
        
        return {
            'success': True,
            'interpolated_value': result,
            'polynomial': polynomial,
            'difference_table': diff_table,
            'method': 'Newton Forward Interpolation',
            'u': u,
            'x_interp': x_interp
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def newton_backward_interpolation(x: np.ndarray, y: np.ndarray, x_interp: float) -> Dict:
    """
    Newton Backward Interpolation
    Best for interpolation near the end of the table
    
    Args:
        x: Array of x values (equally spaced)
        y: Array of y values
        x_interp: Value to interpolate at
    
    Returns:
        Dictionary containing interpolated value, polynomial, and difference table
    """
    try:
        n = len(x)
        h = x[1] - x[0]
        
        # Create backward difference table
        diff_table = create_backward_difference_table(x, y)
        
        # Calculate u = (x_interp - x[-1]) / h
        u = (x_interp - x[-1]) / h
        
        # Newton backward interpolation formula
        result = y[-1]
        u_term = u
        factorial = 1
        
        terms = [f"{y[-1]:.6f}"]
        
        for i in range(1, n):
            factorial *= i
            coeff = diff_table.iloc[-1, i + 1]  # +1 because x column is at index 0
            term = coeff * u_term / factorial
            result += term
            
            # Build polynomial term
            u_str = " * ".join([f"(u + {j})" for j in range(i)])
            terms.append(f"({coeff:.6f} * {u_str}) / {factorial}")
            
            u_term *= (u + i)
        
        polynomial = " + ".join(terms)
        polynomial = f"P(u) = {polynomial}, where u = (x - {x[-1]}) / {h}"
        
        return {
            'success': True,
            'interpolated_value': result,
            'polynomial': polynomial,
            'difference_table': diff_table,
            'method': 'Newton Backward Interpolation',
            'u': u,
            'x_interp': x_interp
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def divided_differences_interpolation(x: np.ndarray, y: np.ndarray, x_interp: float) -> Dict:
    """
    Newton Divided Differences Interpolation
    Works for both equally and unequally spaced data
    
    Args:
        x: Array of x values
        y: Array of y values
        x_interp: Value to interpolate at
    
    Returns:
        Dictionary containing interpolated value, polynomial, and difference table
    """
    try:
        n = len(x)
        
        # Create divided difference table
        diff_table = create_divided_difference_table(x, y)
        
        # Newton divided difference formula
        result = y[0]
        product_term = 1
        
        terms = [f"{y[0]:.6f}"]
        
        for i in range(1, n):
            product_term *= (x_interp - x[i - 1])
            coeff = diff_table.iloc[0, i + 1]  # +1 because x column is at index 0
            term = coeff * product_term
            result += term
            
            # Build polynomial term
            x_str = " * ".join([f"(x - {x[j]:.4f})" for j in range(i)])
            terms.append(f"{coeff:.6f} * {x_str}")
        
        polynomial = " + ".join(terms)
        polynomial = f"P(x) = {polynomial}"
        
        return {
            'success': True,
            'interpolated_value': result,
            'polynomial': polynomial,
            'difference_table': diff_table,
            'method': 'Newton Divided Differences',
            'x_interp': x_interp
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def lagrange_interpolation(x: np.ndarray, y: np.ndarray, x_interp: float) -> Dict:
    """
    Lagrange Interpolation
    Direct method without difference tables
    
    Args:
        x: Array of x values
        y: Array of y values
        x_interp: Value to interpolate at
    
    Returns:
        Dictionary containing interpolated value and polynomial
    """
    try:
        n = len(x)
        result = 0
        
        terms = []
        
        for i in range(n):
            # Calculate Li(x)
            Li = 1
            Li_str_num = []
            Li_str_den = []
            
            for j in range(n):
                if i != j:
                    Li *= (x_interp - x[j]) / (x[i] - x[j])
                    Li_str_num.append(f"(x - {x[j]:.4f})")
                    Li_str_den.append(f"({x[i]:.4f} - {x[j]:.4f})")
            
            result += y[i] * Li
            
            # Build polynomial term
            num_str = " * ".join(Li_str_num)
            den_str = " * ".join(Li_str_den)
            terms.append(f"{y[i]:.6f} * [{num_str}] / [{den_str}]")
        
        polynomial = " + ".join(terms)
        polynomial = f"P(x) = {polynomial}"
        
        # Create a simple data table
        data_table = pd.DataFrame({'x': x, 'y': y})
        
        return {
            'success': True,
            'interpolated_value': result,
            'polynomial': polynomial,
            'data_table': data_table,
            'method': 'Lagrange Interpolation',
            'x_interp': x_interp
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }