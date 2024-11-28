import sympy as sp
import time as tt

#general
dim = None
metric = None
metric_inv = None
coords = None

#connection
christoffel = None

#curvature
riemann = None
ricci_t = None
ricci_s = None
einstein = None
kretschmann = None

def set_metric_and_coordinates(g, coords_input):
    """
    Set the global metric and coordinates for the package.
    
    Args:
        g (Matrix): The metric tensor (2D sympy matrix).
        coords (tuple): A tuple of sympy symbols representing the coordinates.
    """

    global metric, metric_inv, coords, dim, christoffel, riemann, ricci_t, ricci_s, einstein, kretschmann  # Use the global variables

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
    riemann = None
    ricci_t = None
    ricci_s = None
    einstein = None
    kretschmann = None

def ensure_christoffel():
    """
        Checks whether Christoffel symbols have already been computed, if not, computes them and stores in spacetime.christoffel.
    """
    global christoffel
    from .tensor import Tensor 

    if metric == None:
        raise ValueError("Metric tensor not set.")
    if christoffel == None:
        dg = Tensor("g", metric, [-1,-1]).partial_gradient()
        christoffel = Tensor(r"Γ", (sp.Rational(1,2) * (dg.swap_indices(0,2) + dg.swap_indices(0,1) - dg)).raise_index(0))

def ensure_curvature():
    """
    if not computed, computes the riemann, ricci and einstein tensors as well as the ricci scalar.
    kretschmann scalar is omitted for computation time reasons. 
    for kretschmann scalar, call ensure_kretschmann()
    """
    global riemann, ricci_t, ricci_s, einstein
    from .tensor import Tensor 

    ensure_christoffel()

    if riemann == None:
        start_time = tt.time()
        Gamma = christoffel
        dGamma = Gamma.partial_gradient()

        riemann = (dGamma.reorder_indices([1,2,0,3]) + (Gamma * Gamma).contract(2,3).reorder_indices([0,2,1,3])).antisymmetrise_pair(2,3, normalise = False)
        riemann.name = "R"
        riemann = riemann.trigsimp()
        riemann = riemann.simplify()

        ricci_t = riemann.contract(0,2)
        #ricci_t = ricci_t.trigsimp()
        #ricci_t = ricci_t.simplify()

        ricci_s = ricci_t.contract()

        einstein = (ricci_t - sp.Rational(1,2) * ricci_s * Tensor("g", metric, [-1,-1])).trigsimp().simplify()
        einstein.name = "G"
        elapsed = tt.time() - start_time
        print(f"Riemann tensor etc. computed in {elapsed:.3f}s")

def ensure_kretschmann():
    """
    if not computed, computes the kretschmann scalar
    """
    global kretschmann
    from .tensor import Tensor 

    ensure_curvature()

    if kretschmann == None:
        start_time = tt.time()
        #kretschmann = (riemann * riemann).contract(0,4).contract(0,3).contract(0,2).contract()


        #generating reduced complete index set
        pairs = list()
        independent_indices = list()

        for i in range(1,dim):
            for j in range(i):
                pairs.append((i,j))

        for i,a in enumerate(pairs):
            for j,b in enumerate(pairs):
                if i <= j:
                    independent_indices.append((a[0],a[1],b[0],b[1]))

        #computing fully contra- and covariant riemann tensors
        riemann_upper = riemann.raise_index(1).raise_index(2).raise_index(3)
        riemann_lower = riemann.lower_index(0)


        #summing over independent indices with appropriate factors
        kretschmann = 0
        factor_sum = 0
        for i in independent_indices:
            factor = 4
            if (i[0],i[1]) != (i[2],i[3]):
                factor *= 2
            kretschmann += factor * (riemann_upper.components[i] * riemann_lower.components[i]).trigsimp().simplify()
            factor_sum += factor
        #print("Factor sum = " + str(factor_sum))

        kretschmann = Tensor("K", kretschmann).trigsimp().simplify()
        elapsed = tt.time() - start_time
        print(f"Kretschmann scalar computed in {elapsed:.3f}s")