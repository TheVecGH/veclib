import sympy as sp
from .tensor import Tensor 

dim = None
metric = None
metric_inv = None
coords = None
christoffel = None

def set_metric_and_coordinates(g, coords_input):
    """
    Set the global metric and coordinates for the package.
    
    Args:
        g (Matrix): The metric tensor (2D sympy matrix).
        coords (tuple): A tuple of sympy symbols representing the coordinates.
    """

    global metric, metric_inv, coords, dim, christoffel  # Use the global variables

    g_components = sp.MutableDenseNDimArray(g.components)
    if g_components.rank() != 2:
        raise ValueError(
                f"Metric tensor must have rank 2, not {g_components.rank}"
            )

    metric = sp.Matrix(g_components)
    metric_inv = metric.inv()
    metric = sp.MutableDenseNDimArray(metric)
    metric_inv = sp.MutableDenseNDimArray(metric_inv)
    coords = coords_input
    dim = len(coords)
    christoffel = None #clearing christoffel symbols

def ensure_christoffel():
    """
        Checks whether Christoffel symbols have already been computed, if not, computes them and stores in spacetime.christoffel.
    """
    global christoffel

    if metric == None:
        raise ValueError("Metric tensor not set.")
    if christoffel == None:
        dg = Tensor("g", metric, [-1,-1]).partial_gradient()
        christoffel = Tensor(r"Γ", (sp.Rational(1,2) * (dg.swap_indices(0,2) + dg.swap_indices(0,1) - dg)).raise_index(0))