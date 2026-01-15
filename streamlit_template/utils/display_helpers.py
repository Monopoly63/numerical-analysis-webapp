"""
Display Helper Functions
Utilities for formatting and displaying results in Streamlit
"""

import pandas as pd
import streamlit as st
from typing import List, Dict


def format_steps_table(steps: List[Dict]) -> pd.DataFrame:
    """
    Format step-by-step results into a pandas DataFrame
    
    Args:
        steps: List of dictionaries containing step information
    
    Returns:
        Formatted DataFrame
    """
    if not steps:
        return pd.DataFrame()
    
    df = pd.DataFrame(steps)
    
    # Round numeric columns to 6 decimal places
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].round(6)
    
    return df


def display_polynomial(polynomial: str):
    """
    Display polynomial equation in a formatted way
    
    Args:
        polynomial: String representation of polynomial
    """
    st.markdown("### Polynomial Equation")
    
    # Split long polynomials for better readability
    if len(polynomial) > 100:
        st.text_area("Polynomial", polynomial, height=150, disabled=True)
    else:
        st.code(polynomial, language=None)


def display_difference_table(table: pd.DataFrame, title: str = "Difference Table"):
    """
    Display difference table with proper formatting
    
    Args:
        table: DataFrame containing the difference table
        title: Title for the table
    """
    st.markdown(f"### {title}")
    
    # Format numeric columns
    formatted_table = table.copy()
    numeric_columns = formatted_table.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_columns:
        formatted_table[col] = formatted_table[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) and x != 0 else "")
    
    st.dataframe(formatted_table, use_container_width=True)


def create_function_from_string(func_str: str, variable: str = 'x'):
    """
    Create a callable function from a string expression
    
    Args:
        func_str: String representation of function (e.g., "x**2 + 2*x + 1")
        variable: Variable name (default: 'x')
    
    Returns:
        Callable function
    """
    import sympy as sp
    
    try:
        x = sp.Symbol(variable)
        expr = sp.sympify(func_str)
        func = sp.lambdify(x, expr, 'numpy')
        return func
    except Exception as e:
        raise ValueError(f"Invalid function expression: {str(e)}")


def display_method_info(method_name: str, description: str, formula: str = None):
    """
    Display information about a numerical method
    
    Args:
        method_name: Name of the method
        description: Description of the method
        formula: Mathematical formula (optional)
    """
    st.markdown(f"### {method_name}")
    st.info(description)
    
    if formula:
        st.markdown("**Formula:**")
        st.latex(formula)