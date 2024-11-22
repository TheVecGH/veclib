import sympy as sp

dim = None
metric = None
metric_inv = None
coords = None

def set_metric_and_coordinates(g, coords_input):
    """
    Set the global metric and coordinates for the package.
    
    Args:
        g (Matrix): The metric tensor (2D sympy matrix).
        coords (tuple): A tuple of sympy symbols representing the coordinates.
    """

    global metric, metric_inv, coords, dim  # Use the global variables

    g_components = sp.MutableDenseNDimArray(g.get_components())
    if g_components.rank() != 2:
        raise ValueError(
                f"Metric tensor must have rank 2, not {g_components.rank()}"
            )

    metric = sp.Matrix(g_components)
    metric_inv = metric.inv()
    coords = coords_input
    dim = len(coords)