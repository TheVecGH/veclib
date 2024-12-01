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
weyl = None

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
        start_time = tt.time()
        dg = Tensor("g", metric, [-1,-1]).partial_gradient()
        christoffel = Tensor(r"Γ", (sp.Rational(1,2) * (dg.swap_indices(0,2) + dg.swap_indices(0,1) - dg)).raise_index(0))
        elapsed = tt.time() - start_time
        print(f"Christoffel symbols computed in {elapsed:.3f}s")

def ensure_curvature():
    """
    if not computed, computes the riemann, ricci and einstein tensors as well as the ricci scalar.
    kretschmann scalar is omitted for computation time reasons. 
    for kretschmann scalar, call ensure_kretschmann()
    """
    global riemann, ricci_t, ricci_s
    from .tensor import Tensor 

    ensure_christoffel()

    if riemann == None:
        start_time = tt.time()
        Gamma = christoffel
        dGamma = Gamma.partial_gradient()

        # riemann = (dGamma.reorder_indices([1,2,0,3]) + (Gamma * Gamma).contract(2,3).reorder_indices([0,2,1,3])).antisymmetrise_pair(2,3, normalise = False).trigsimp().simplify()
        riemann = Tensor("R", sp.MutableDenseNDimArray.zeros(dim,dim,dim,dim), [-1,-1,-1,-1])
        Gamma_ = Gamma.lower_index(0)
        dGamma_ = dGamma.lower_index(1)
        for idx in riemann_independent_indices():
            a = idx[0]
            b = idx[1]
            c = idx[2]
            d = idx[3]
            component = dGamma_[c,a,d,b] - dGamma_[d,a,c,b] + sum(Gamma_[a,c,k] * Gamma[k,d,b] - Gamma_[a,d,k] * Gamma[k,c,b] for k in range(dim))
            component = component.trigsimp().simplify()
            riemann[a,b,c,d] = riemann[b,a,d,c] = component
            riemann[b,a,c,d] = riemann[a,b,d,c] = -component
            if (a,b) != (c,d):
                riemann[c,d,a,b] = riemann[d,c,b,a] = component
                riemann[c,d,b,a] = riemann[d,c,a,b] = -component

        riemann = riemann.raise_index(0)

        ricci_t = Tensor("R", sp.MutableDenseNDimArray.zeros(dim,dim), [-1,-1])
        ricci_s_component = 0

        for i in range(dim):
            for j in range(i + 1):
                component = sum(riemann.components[c,i,c,j] for c in range(dim)).trigsimp().simplify()
                ricci_t.components[i,j] = component
                if i != j:
                    ricci_t.components[j,i] = component
                    component *= 2
                ricci_s_component += component * metric_inv[i,j]

        ricci_s = Tensor("R", ricci_s_component, []).trigsimp().simplify()

        elapsed = tt.time() - start_time                
        print(f"Riemann tensor etc. computed in {elapsed:.3f}s")

        
def ensure_einstein():

    global einstein
    from .tensor import Tensor

    ensure_curvature()
    if einstein == None:    
        start_time = tt.time()
        einstein = (ricci_t - sp.Rational(1,2) * ricci_s * Tensor("g", metric, [-1,-1])).expand()
        einstein.name = "G"
        elapsed = tt.time() - start_time                
        print(f"Einstein tensor etc. computed in {elapsed:.3f}s")



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

        #computing fully contra- and covariant riemann tensors
        riemann_upper = riemann.raise_index(1).raise_index(2).raise_index(3)
        riemann_lower = riemann.lower_index(0)

        #summing over independent indices with appropriate factors
        kretschmann = 0
        factor_sum = 0
        for i in riemann_independent_indices():
            factor = 4
            if (i[0],i[1]) != (i[2],i[3]):
                factor *= 2
            kretschmann += factor * (riemann_upper.components[i] * riemann_lower.components[i]).trigsimp().simplify()
            factor_sum += factor
        #print("Factor sum = " + str(factor_sum))

        kretschmann = Tensor("K", kretschmann).trigsimp().simplify()
        elapsed = tt.time() - start_time
        print(f"Kretschmann scalar computed in {elapsed:.3f}s")

def ensure_weyl():
    """
    computes the Weyl curvature tensor
    """
    global weyl
    from .tensor import Tensor

    ensure_curvature()

    if dim < 4:
        weyl = Tensor("C", sp.MutableDenseNDimArray.zeros(dim,dim,dim,dim), [-1,-1,-1,-1])
        return

    g = Tensor("g", metric, [-1,-1])

    #weyl = Tensor("C", sp.MutableDenseNDimArray.zeros(dim,dim,dim,dim), [-1,-1,-1,-1])
    ricci_t_correction = -sp.Rational(2, dim - 2) * (
        (g * ricci_t)
        .antisymmetrise_pair(1, 2)
        .antisymmetrise_pair(0, 3, normalise=False)
        .reorder_indices([0, 3, 1, 2])
    )
    ricci_s_correction = sp.Rational(2, (dim - 1) * (dim - 2)) * (
        ricci_s * (g * g)
        .antisymmetrise_pair(1, 2)
        .reorder_indices([0, 3, 1, 2])
    )

    #ricci_t_correction.show_components()
    #ricci_s_correction.show_components()

    weyl = riemann.lower_index(0) + ricci_t_correction + ricci_s_correction
    weyl = weyl.trigsimp().simplify()
    weyl.name = "C"

def riemann_independent_indices():
    """
    returns (almost) independent indices of the Riemann tensor.
    """
    if metric == None:
        raise ValueError("Metric not set.")

    pairs = list()
    independent_indices = list()

    for i in range(1,dim):
        for j in range(i):
            pairs.append((i,j))

    for i,a in enumerate(pairs):
        for j,b in enumerate(pairs):
            if i <= j:
                independent_indices.append((a[0],a[1],b[0],b[1]))
    return independent_indices