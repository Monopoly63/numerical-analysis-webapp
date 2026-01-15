"""
Numerical Analysis Application
A comprehensive tool for university students studying Numerical Analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
from numerical_methods import (
    bisection_method, newton_raphson_method, secant_method,
    symbolic_derivative, numerical_derivative,
    trapezoidal_rule, simpson_one_third, simpson_three_eighth,
    newton_forward_interpolation, newton_backward_interpolation,
    divided_differences_interpolation, lagrange_interpolation
)
from utils.display_helpers import (
    format_steps_table, display_polynomial, display_difference_table,
    create_function_from_string, display_method_info
)


# Page configuration
st.set_page_config(
    page_title="Numerical Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
    }
    h2 {
        color: #2ca02c;
        margin-top: 2rem;
    }
    h3 {
        color: #ff7f0e;
    }
    .stDataFrame {
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Numerical Analysis Tool")
st.markdown("*A comprehensive application for university students studying Numerical Analysis*")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Section:",
    ["Home", "Equation Solvers", "Differentiation", "Integration", "Interpolation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application implements various numerical methods with step-by-step solutions "
    "and clear visualizations for educational purposes."
)


# ============================================================================
# HOME SECTION
# ============================================================================
if section == "Home":
    st.header("Welcome to the Numerical Analysis Tool")
    
    st.markdown("""
    This application provides implementations of fundamental numerical methods used in 
    computational mathematics and engineering. Each method includes:
    
    - 📝 Step-by-step solutions
    - 📊 Detailed computation tables
    - 🔢 Polynomial equations (where applicable)
    - 📈 Educational explanations
    
    ### Available Methods:
    
    #### 🎯 Equation Solvers
    - **Bisection Method**: Finds roots by repeatedly bisecting an interval
    - **Newton-Raphson Method**: Fast convergence using derivatives
    - **Secant Method**: Similar to Newton-Raphson but without derivatives
    
    #### 📐 Differentiation
    - **Symbolic Differentiation**: Exact derivatives using SymPy
    - **Numerical Differentiation**: Forward, Backward, and Central differences
    
    #### ∫ Integration
    - **Trapezoidal Rule**: Approximates area using trapezoids
    - **Simpson's 1/3 Rule**: Higher accuracy using parabolic approximations
    - **Simpson's 3/8 Rule**: Alternative Simpson's method
    
    #### 📈 Interpolation
    - **Newton Forward Interpolation**: Best for data near the beginning
    - **Newton Backward Interpolation**: Best for data near the end
    - **Divided Differences**: Works with unequally spaced data
    - **Lagrange Interpolation**: Direct polynomial construction
    
    ### How to Use:
    1. Select a section from the sidebar
    2. Choose a method
    3. Input your parameters
    4. View step-by-step results
    
    ---
    *Built with Python, NumPy, SymPy, and Streamlit*
    """)


# ============================================================================
# EQUATION SOLVERS SECTION
# ============================================================================
elif section == "Equation Solvers":
    st.header("🎯 Equation Solvers")
    
    method = st.selectbox(
        "Select Method:",
        ["Bisection Method", "Newton-Raphson Method", "Secant Method"]
    )
    
    st.markdown("---")
    
    if method == "Bisection Method":
        display_method_info(
            "Bisection Method",
            "The Bisection Method finds roots by repeatedly dividing an interval in half and "
            "selecting the subinterval where the function changes sign.",
            r"c = \frac{a + b}{2}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Function f(x):",
                value="x**3 - x - 2",
                help="Enter function using Python syntax (e.g., x**2, sin(x), exp(x))"
            )
            a = st.number_input("Left endpoint (a):", value=1.0)
            b = st.number_input("Right endpoint (b):", value=2.0)
        
        with col2:
            tol = st.number_input("Tolerance:", value=1e-6, format="%.2e")
            max_iter = st.number_input("Max iterations:", value=100, min_value=1)
        
        if st.button("Solve", key="bisection"):
            try:
                func = create_function_from_string(func_str)
                result = bisection_method(func, a, b, tol, max_iter)
                
                if result['success']:
                    st.success(f"✅ Root found: **{result['root']:.8f}**")
                    st.metric("Iterations", result['iterations'])
                    st.metric("Final Error", f"{result['final_error']:.2e}")
                    
                    st.markdown("### Step-by-Step Results")
                    df = format_steps_table(result['steps'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error(f"❌ {result['error']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif method == "Newton-Raphson Method":
        display_method_info(
            "Newton-Raphson Method",
            "The Newton-Raphson Method uses the derivative to find roots with fast convergence.",
            r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Function f(x):",
                value="x**3 - x - 2",
                help="Enter function using Python syntax"
            )
            deriv_str = st.text_input(
                "Derivative f'(x):",
                value="3*x**2 - 1",
                help="Enter derivative using Python syntax"
            )
            x0 = st.number_input("Initial guess (x₀):", value=1.5)
        
        with col2:
            tol = st.number_input("Tolerance:", value=1e-6, format="%.2e")
            max_iter = st.number_input("Max iterations:", value=100, min_value=1)
        
        if st.button("Solve", key="newton"):
            try:
                func = create_function_from_string(func_str)
                derivative = create_function_from_string(deriv_str)
                result = newton_raphson_method(func, derivative, x0, tol, max_iter)
                
                if result['success']:
                    st.success(f"✅ Root found: **{result['root']:.8f}**")
                    st.metric("Iterations", result['iterations'])
                    st.metric("Final Error", f"{result['final_error']:.2e}")
                    
                    st.markdown("### Step-by-Step Results")
                    df = format_steps_table(result['steps'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error(f"❌ {result['error']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif method == "Secant Method":
        display_method_info(
            "Secant Method",
            "The Secant Method approximates the derivative using two points, avoiding the need for explicit derivatives.",
            r"x_{n+1} = x_n - f(x_n) \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Function f(x):",
                value="x**3 - x - 2",
                help="Enter function using Python syntax"
            )
            x0 = st.number_input("First initial guess (x₀):", value=1.0)
            x1 = st.number_input("Second initial guess (x₁):", value=2.0)
        
        with col2:
            tol = st.number_input("Tolerance:", value=1e-6, format="%.2e")
            max_iter = st.number_input("Max iterations:", value=100, min_value=1)
        
        if st.button("Solve", key="secant"):
            try:
                func = create_function_from_string(func_str)
                result = secant_method(func, x0, x1, tol, max_iter)
                
                if result['success']:
                    st.success(f"✅ Root found: **{result['root']:.8f}**")
                    st.metric("Iterations", result['iterations'])
                    st.metric("Final Error", f"{result['final_error']:.2e}")
                    
                    st.markdown("### Step-by-Step Results")
                    df = format_steps_table(result['steps'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error(f"❌ {result['error']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ============================================================================
# DIFFERENTIATION SECTION
# ============================================================================
elif section == "Differentiation":
    st.header("📐 Differentiation")
    
    method = st.selectbox(
        "Select Method:",
        ["Symbolic Differentiation", "Numerical Differentiation"]
    )
    
    st.markdown("---")
    
    if method == "Symbolic Differentiation":
        display_method_info(
            "Symbolic Differentiation",
            "Computes exact derivatives using symbolic mathematics (SymPy)."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            expr_str = st.text_input(
                "Expression:",
                value="x**3 + 2*x**2 - 5*x + 1",
                help="Enter expression using SymPy syntax (e.g., sin(x), exp(x), log(x))"
            )
            variable = st.text_input("Variable:", value="x")
        
        with col2:
            order = st.number_input("Derivative order:", value=1, min_value=1, max_value=5)
        
        if st.button("Compute Derivative", key="symbolic"):
            result = symbolic_derivative(expr_str, variable, order)
            
            if result['success']:
                st.success("✅ Derivative computed successfully")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Expression:**")
                    st.latex(result['original'])
                
                with col2:
                    st.markdown(f"**{order}{'st' if order == 1 else 'nd' if order == 2 else 'rd' if order == 3 else 'th'} Derivative:**")
                    st.latex(result['derivative'])
                
                st.markdown("**Simplified Form:**")
                st.code(result['derivative'], language=None)
            else:
                st.error(f"❌ {result['error']}")
    
    elif method == "Numerical Differentiation":
        display_method_info(
            "Numerical Differentiation",
            "Approximates derivatives using finite differences."
        )
        
        diff_method = st.radio(
            "Difference Method:",
            ["Forward Difference", "Backward Difference", "Central Difference"],
            horizontal=True
        )
        
        method_map = {
            "Forward Difference": "forward",
            "Backward Difference": "backward",
            "Central Difference": "central"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            func_str = st.text_input(
                "Function f(x):",
                value="x**3 + 2*x**2 - 5*x + 1",
                help="Enter function using Python syntax"
            )
            x = st.number_input("Point x:", value=2.0)
        
        with col2:
            h = st.number_input("Step size h:", value=1e-5, format="%.2e")
        
        if st.button("Compute Derivative", key="numerical"):
            try:
                func = create_function_from_string(func_str)
                result = numerical_derivative(func, x, h, method_map[diff_method])
                
                if result['success']:
                    st.success(f"✅ f'({x}) ≈ **{result['derivative']:.8f}**")
                    
                    st.markdown("### Method Details")
                    st.info(f"**Formula:** {result['formula']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("f(x)", f"{result['f(x)']:.6f}")
                    with col2:
                        if result['f(x+h)'] is not None:
                            st.metric("f(x+h)", f"{result['f(x+h)']:.6f}")
                    with col3:
                        if result['f(x-h)'] is not None:
                            st.metric("f(x-h)", f"{result['f(x-h)']:.6f}")
                else:
                    st.error(f"❌ {result['error']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ============================================================================
# INTEGRATION SECTION
# ============================================================================
elif section == "Integration":
    st.header("∫ Integration")
    
    method = st.selectbox(
        "Select Method:",
        ["Trapezoidal Rule", "Simpson's 1/3 Rule", "Simpson's 3/8 Rule"]
    )
    
    st.markdown("---")
    
    if method == "Trapezoidal Rule":
        display_method_info(
            "Trapezoidal Rule",
            "Approximates the integral by dividing the area into trapezoids.",
            r"\int_a^b f(x)dx \approx \frac{h}{2}[f(x_0) + 2\sum_{i=1}^{n-1}f(x_i) + f(x_n)]"
        )
    elif method == "Simpson's 1/3 Rule":
        display_method_info(
            "Simpson's 1/3 Rule",
            "Uses parabolic approximations for higher accuracy (requires even number of intervals).",
            r"\int_a^b f(x)dx \approx \frac{h}{3}[f(x_0) + 4\sum_{i=odd}f(x_i) + 2\sum_{i=even}f(x_i) + f(x_n)]"
        )
    else:
        display_method_info(
            "Simpson's 3/8 Rule",
            "Alternative Simpson's method using cubic approximations (requires intervals divisible by 3).",
            r"\int_a^b f(x)dx \approx \frac{3h}{8}[f(x_0) + 3f(x_1) + 3f(x_2) + 2f(x_3) + ...]"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        func_str = st.text_input(
            "Function f(x):",
            value="x**2",
            help="Enter function using Python syntax"
        )
        a = st.number_input("Lower limit (a):", value=0.0)
        b = st.number_input("Upper limit (b):", value=1.0)
    
    with col2:
        n = st.number_input("Number of intervals (n):", value=100, min_value=2)
    
    if st.button("Compute Integral", key="integration"):
        try:
            func = create_function_from_string(func_str)
            
            if method == "Trapezoidal Rule":
                result = trapezoidal_rule(func, a, b, n)
            elif method == "Simpson's 1/3 Rule":
                result = simpson_one_third(func, a, b, n)
            else:
                result = simpson_three_eighth(func, a, b, n)
            
            if result['success']:
                st.success(f"✅ Integral ≈ **{result['integral']:.8f}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Method", result['method'])
                with col2:
                    st.metric("Intervals Used", result['n'])
                with col3:
                    st.metric("Step Size (h)", f"{result['h']:.6f}")
                
                st.info(f"**Formula:** {result['formula']}")
                
                if result.get('note'):
                    st.warning(f"ℹ️ {result['note']}")
            else:
                st.error(f"❌ {result['error']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ============================================================================
# INTERPOLATION SECTION
# ============================================================================
elif section == "Interpolation":
    st.header("📈 Interpolation")
    
    method = st.selectbox(
        "Select Method:",
        [
            "Newton Forward Interpolation",
            "Newton Backward Interpolation",
            "Divided Differences",
            "Lagrange Interpolation"
        ]
    )
    
    st.markdown("---")
    
    if method == "Newton Forward Interpolation":
        display_method_info(
            "Newton Forward Interpolation",
            "Best suited for interpolation near the beginning of the data table. "
            "Uses forward differences and requires equally spaced data points."
        )
    elif method == "Newton Backward Interpolation":
        display_method_info(
            "Newton Backward Interpolation",
            "Best suited for interpolation near the end of the data table. "
            "Uses backward differences and requires equally spaced data points."
        )
    elif method == "Divided Differences":
        display_method_info(
            "Newton Divided Differences",
            "Works with both equally and unequally spaced data points. "
            "Uses divided difference table for polynomial construction."
        )
    else:
        display_method_info(
            "Lagrange Interpolation",
            "Direct method that constructs interpolating polynomial without difference tables. "
            "Works with any spacing of data points."
        )
    
    # Input section
    st.markdown("### Input Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_points = st.number_input(
            "Number of data points:",
            value=5,
            min_value=2,
            max_value=20,
            help="Enter the number of (x, y) pairs"
        )
    
    with col2:
        x_interp = st.number_input(
            "Value to interpolate at:",
            value=2.5,
            help="Enter the x value where you want to find y"
        )
    
    # Data input method
    input_method = st.radio(
        "Data input method:",
        ["Manual Entry", "Use Example Data"],
        horizontal=True
    )
    
    if input_method == "Manual Entry":
        st.markdown("**Enter your data points:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**X values:**")
            x_values = []
            for i in range(n_points):
                x_val = st.number_input(
                    f"x[{i}]:",
                    value=float(i),
                    key=f"x_{i}"
                )
                x_values.append(x_val)
        
        with col2:
            st.markdown("**Y values:**")
            y_values = []
            for i in range(n_points):
                y_val = st.number_input(
                    f"y[{i}]:",
                    value=float(i**2),
                    key=f"y_{i}"
                )
                y_values.append(y_val)
        
        x = np.array(x_values)
        y = np.array(y_values)
    
    else:
        # Example data
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        
        st.info("**Example Data:** f(x) = x²")
        data_df = pd.DataFrame({'x': x, 'y': y})
        st.dataframe(data_df, use_container_width=True)
    
    if st.button("Interpolate", key="interpolate"):
        try:
            # Perform interpolation based on selected method
            if method == "Newton Forward Interpolation":
                result = newton_forward_interpolation(x, y, x_interp)
            elif method == "Newton Backward Interpolation":
                result = newton_backward_interpolation(x, y, x_interp)
            elif method == "Divided Differences":
                result = divided_differences_interpolation(x, y, x_interp)
            else:  # Lagrange
                result = lagrange_interpolation(x, y, x_interp)
            
            if result['success']:
                # Display result
                st.success(f"✅ Interpolated value at x = {x_interp}: **y ≈ {result['interpolated_value']:.8f}**")
                
                # Display polynomial
                display_polynomial(result['polynomial'])
                
                # Display difference table or data table
                if 'difference_table' in result:
                    display_difference_table(
                        result['difference_table'],
                        f"{result['method']} - Difference Table"
                    )
                elif 'data_table' in result:
                    st.markdown("### Input Data Table")
                    st.dataframe(result['data_table'], use_container_width=True)
                
                # Additional information
                if 'u' in result:
                    st.info(f"**Normalized parameter:** u = {result['u']:.6f}")
            
            else:
                st.error(f"❌ {result['error']}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Numerical Analysis Tool v1.0 | Built with Python, NumPy, SymPy, and Streamlit</p>
        <p>For educational purposes | © 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)